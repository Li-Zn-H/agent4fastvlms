"""Observe native MoE routing and weighted post-SwiGLU moments without patches."""

from skills.deepseek_vl2_pruning import dimensions, modules, units
from skills.structured_pruning import STRATA


class RoutingMoments:
    """Equal-stratum, unconditional moments of routing_weight * activation.

An expert's denominator includes *all* tokens of a stratum, including tokens
not routed to that expert. Separate observed-hit counts prevent an unobserved
expert from being treated as a cheap pruning target.
"""

    def __init__(self, model):
        import torch
        self.torch = torch
        self.dims = dimensions(model.config.language_config)
        self.domain = units(self.dims)
        self.branches = modules(model, self.dims)
        self.states = []
        self.counts = []
        self.handles = []
        self.active = None
        self.calls = [0] * self.dims['num_hidden_layers']
        self.contexts = [{} for _ in self.calls]
        for unit, experts in zip(self.domain, self.branches):
            self.counts.append({name: 0 for name in STRATA})
            self.states.append([{name: {
                'hits': 0,
                'sums': torch.zeros(unit['original_width'], dtype=torch.float32, device=e.down_proj.weight.device),
                'squares': torch.zeros(unit['original_width'], dtype=torch.float32, device=e.down_proj.weight.device),
            } for name in STRATA} for e in experts])
        try:
            for layer_index, layer in enumerate(model.language.model.layers):
                self.handles.append(layer.mlp.register_forward_pre_hook(self._before(layer_index)))
                self.handles.append(layer.mlp.register_forward_hook(self._after(layer_index)))
                if hasattr(layer.mlp, 'experts'):
                    self.handles.append(layer.mlp.gate.register_forward_hook(self._routing(layer_index)))
            for index, (unit, experts) in enumerate(zip(self.domain, self.branches)):
                for expert_index, expert in enumerate(experts):
                    if unit['kind'] == 'routed':
                        self.handles.append(expert.register_forward_pre_hook(self._check_order(index, expert_index)))
                    self.handles.append(expert.down_proj.register_forward_pre_hook(self._capture(index, expert_index)))
        except BaseException:
            self.close()
            raise

    def start_sample(self, image_mask):
        if image_mask.ndim != 1 or not image_mask.any() or image_mask.all():
            raise ValueError('Calibration requires one native sequence with image and text positions')
        self.active = image_mask.detach().bool()
        self.calls = [0] * len(self.calls)

    def _before(self, layer_index):
        def hook(module, args):
            value = args[0]
            if self.active is None or value.ndim != 3 or value.shape[0] != 1:
                raise ValueError('DeepSeek calibration requires batch one and start_sample')
            if self.calls[layer_index] == 0:
                if value.shape[1] != self.active.numel():
                    raise ValueError('Native prefill and image mask disagree')
                labels = self.active.to(dtype=self.torch.long).neg().add(1)
            else:
                if value.shape[1] != 1:
                    raise ValueError('Calibration decoding must use one cached token')
                labels = self.torch.full((1,), 2, device=value.device, dtype=self.torch.long)
            self.calls[layer_index] += 1
            counts = self.torch.bincount(labels, minlength=3).tolist()
            self.contexts[layer_index] = {'labels': labels}
            for index, unit in enumerate(self.domain):
                if unit['layer'] == layer_index:
                    for name, count in zip(STRATA, counts):
                        self.counts[index][name] += count
        return hook

    def _routing(self, layer_index):
        def hook(module, args, output):
            ids, weights, _ = output
            context = self.contexts[layer_index]
            if ids.shape != weights.shape or ids.shape != (len(context['labels']), self.dims['num_experts_per_tok']):
                raise ValueError('Native gate returned an unexpected routing shape')
            if (ids.sort(dim=-1).values.diff(dim=-1) == 0).any():
                raise ValueError('Duplicate expert routes within a token')
            # This is exactly the native single-rank moe_infer ordering.
            order = ids.reshape(-1).argsort()
            positions = order // ids.shape[1]
            context['routed_labels'] = context['labels'][positions]
            context['routed_weights'] = weights.reshape(-1)[order].detach().float()
            context['routed_inputs'] = args[0].reshape(-1, args[0].shape[-1])[positions].detach()
            counts = self.torch.bincount(ids.reshape(-1), minlength=self.dims['n_routed_experts']).tolist()
            offset = 0
            context['slices'] = []
            for count in counts:
                context['slices'].append(slice(offset, offset + count))
                offset += count
        return hook

    def _check_order(self, index, expert_index):
        def hook(module, args):
            context = self.contexts[self.domain[index]['layer']]
            expected = context['routed_inputs'][context['slices'][expert_index]]
            if not self.torch.equal(args[0], expected):
                raise ValueError('Observed expert inputs disagree with native routing order')
        return hook

    def _capture(self, index, expert_index):
        def hook(module, args):
            unit = self.domain[index]
            context = self.contexts[unit['layer']]
            value = args[0].detach().float().reshape(-1, unit['original_width'])
            if unit['kind'] == 'routed':
                section = context['slices'][expert_index]
                labels = context['routed_labels'][section]
                value = value * context['routed_weights'][section, None]
            else:
                labels = context['labels']
            if len(value) != len(labels):
                raise ValueError('Expert activations and routed strata disagree')
            for label, name in enumerate(STRATA):
                chunk = value[labels == label]
                state = self.states[index][expert_index][name]
                state['hits'] += len(chunk)
                if chunk.numel():
                    state['sums'].add_(chunk.sum(dim=0))
                    state['squares'].add_(chunk.square().sum(dim=0))
        return hook

    def _after(self, layer_index):
        def hook(module, args, output):
            self.contexts[layer_index].clear()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for context in self.contexts:
            context.clear()
