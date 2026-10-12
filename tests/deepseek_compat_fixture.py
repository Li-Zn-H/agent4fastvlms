"""Cross-interpreter native decoder/Token equivalence fixture; CPU only.

Write with the original Transformers 4 Worker, then check exactly the same
weights/inputs with the isolated quantization Worker's Transformers 5 bridge.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('write', 'check'))
    parser.add_argument('path')
    args = parser.parse_args()
    import torch
    from transformers import BatchFeature
    from adapters.deepseek_vl2_dense import native_model_class, load_processor, prepare_image
    from skills.deepseek_vl2_token import DeepSeekTokenController, image_layout
    from PIL import Image

    torch.set_num_threads(2)
    descriptor = json.loads((ROOT / 'configs/model-deepseek-vl2.json').read_text())
    model_class = native_model_class(descriptor)
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    config = dict(tile_tag='2D', global_view_pos='head',
        vision_config={'image_size': 56, 'patch_size': 14, 'width': 32, 'layers': 1, 'heads': 4, 'mlp_ratio': 2},
        projector_config={'input_dim': 32, 'n_embed': 32},
        language_config={'vocab_size': 128, 'hidden_size': 32, 'intermediate_size': 64, 'moe_intermediate_size': 32,
            'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2,
            'kv_lora_rank': 16, 'q_lora_rank': None, 'qk_rope_head_dim': 8, 'qk_nope_head_dim': 8, 'v_head_dim': 8,
            'n_routed_experts': 4, 'n_shared_experts': 1, 'num_experts_per_tok': 2, 'first_k_dense_replace': 1,
            'n_group': 1, 'topk_group': 1, 'topk_method': 'noaux_tc', 'scoring_func': 'sigmoid',
            'max_position_embeddings': 128, 'bos_token_id': 0, 'eos_token_id': 1})
    torch.manual_seed(123)
    model = model_class(DeepseekVLV2Config(**config)).float().eval()

    class TinyVision(torch.nn.Module):
        def forward(self, pixels):
            return torch.arange(512, dtype=pixels.dtype).reshape(1, 16, 32).expand(len(pixels), -1, -1) / 512

    model.vision = TinyVision()
    model.language.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
    path = Path(args.path)
    if args.mode == 'check':
        fixture = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(fixture['weights'], strict=True)
    count = image_layout(2, 2, 1)[2]
    ids = torch.tensor([[3] + [98] * count + [4, 5]])
    def inputs():
        return BatchFeature({'input_ids': ids.clone(), 'attention_mask': torch.ones_like(ids, dtype=torch.bool),
            'images': torch.zeros(1, 3, 3, 56, 56), 'images_seq_mask': ids == 98,
            'images_spatial_crop': torch.tensor([[[2, 1]]])})
    options = {'max_new_tokens': 5, 'do_sample': False, 'pad_token_id': 2, 'eos_token_id': None,
               'return_dict_in_generate': False}
    selection = {'score_dtype': 'float32', 'coverage_grid': {'rows': 1, 'columns': 1}}
    with torch.inference_mode():
        embeds = model.prepare_inputs_embeds(**{k: v for k, v in inputs().items() if k != 'attention_mask'})
        logits = model.language(inputs_embeds=embeds, attention_mask=inputs()['attention_mask'], use_cache=False).logits
        dense = model.generate(**inputs(), **options)
        controller = DeepSeekTokenController(model, selection, .5)
        token = controller.generate(inputs(), options)
    processor = load_processor(descriptor)
    native_inputs = prepare_image(processor, Image.new('RGB', (768, 384), (64, 100, 200)), 'Describe this image.')
    actual = {'logits': logits, 'dense': dense, 'token': token, 'token_metrics': controller.last_metrics,
              'processor_inputs': dict(native_inputs)}
    if args.mode == 'write':
        torch.save({'weights': model.state_dict(), 'reference': actual}, path)
    else:
        expected = fixture['reference']
        torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)
        for key in ('dense', 'token'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual['token_metrics'] == expected['token_metrics']
        for key, value in actual['processor_inputs'].items():
            torch.testing.assert_close(value, expected['processor_inputs'][key], rtol=0, atol=0)
    print(json.dumps({'status': 'passed', 'mode': args.mode, 'scope': 'CPU native processor, FP32 decoder logits, greedy Dense/Token generation and actual cache lengths',
                      'dense': dense.tolist(), 'token': token.tolist(), 'metrics': controller.last_metrics}), flush=True)


if __name__ == '__main__':
    main()
