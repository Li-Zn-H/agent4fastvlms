"""Collect native DeepSeek MoE calibration; this explicitly runs the Dense GPU model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', default='configs/model-deepseek-vl2.json')
    parser.add_argument('--evaluation-config', default='configs/evaluation-deepseek-vl2.json')
    parser.add_argument('--config', default='configs/structured_pruning-deepseek-vl2.json')
    args = parser.parse_args()
    import torch
    from search.context import require_stages, validate_result_scope
    from search.evaluate import load_samples
    from search.model import load_model, load_processor, prepare_inputs
    from scripts.calibrate_llava_sp import validate_heldout, _write_moments, _write_scores
    from skills.deepseek_vl2_calibration import RoutingMoments
    from skills.deepseek_vl2_pruning import SCORE, dimensions, units, validate_config, load_calibration
    from skills.structured_pruning import STRATA, contribution_scores, decode_layer

    model_config, evaluation, config = (json.loads((PROJECT_ROOT / p).read_text())
                                       for p in (args.model_config, args.evaluation_config, args.config))
    if model_config.get('adapter') != 'deepseek_vl2':
        raise ValueError('This collector requires the original DeepSeek-VL2 descriptor')
    require_stages(model_config, ('dense',))
    root = validate_result_scope(model_config, evaluation)
    output = (PROJECT_ROOT / config['calibration']['source']).resolve()
    if not output.is_relative_to(root) or output == root or output.exists():
        raise ValueError('Calibration requires a new directory inside this experiment scope')
    checkpoint = json.loads((PROJECT_ROOT / model_config['path'] / 'config.json').read_text())
    dims = dimensions(checkpoint['language_config'])
    validate_config(config, dims)
    data = PROJECT_ROOT / config['calibration']['data']
    media = PROJECT_ROOT / evaluation['data']['media_root']
    samples = load_samples(data, media_root=media)
    heldout = [row for split in ('search', 'dev') for row in load_samples(
        PROJECT_ROOT / evaluation['data'][split], media_root=media)]
    validate_heldout(samples, heldout)
    processor = load_processor(model_config)
    context = []
    for sample in samples:
        inputs = prepare_inputs(processor, sample, media)
        prompt = inputs['input_ids'].shape[1]
        if prompt + evaluation['generation']['max_new_tokens'] > checkpoint['language_config']['max_position_embeddings']:
            raise ValueError('Native calibration input exceeds context capacity; no truncation')
        context.append({'sample_id': sample['stable_row_id'], 'prompt_tokens': prompt})
    model, processor = load_model(model_config)
    collector = RoutingMoments(model)
    try:
        with torch.inference_mode():
            for index, sample in enumerate(samples, 1):
                inputs = prepare_inputs(processor, sample, media).to(model.device)
                collector.start_sample(inputs['images_seq_mask'][0])
                options = dict(evaluation['generation'])
                if options.pop('ignore_eos', False):
                    options.update(eos_token_id=None, forced_eos_token_id=None)
                model.generate(**inputs, **options, return_dict_in_generate=False)
                if index == 1 or index % 16 == 0 or index == len(samples):
                    print(f'Calibration {index}/{len(samples)}', flush=True)
    finally:
        collector.close()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('moments', 'scores'):
        (output / name).mkdir()
    routing_counts = []
    for index, experts in enumerate(collector.branches):
        coverage = []
        for expert_index, expert in enumerate(experts):
            state = collector.states[index][expert_index]
            coverage.append({name: state[name]['hits'] for name in STRATA})
            moments = {name: {'count': collector.counts[index][name],
                              'sums': state[name]['sums'].cpu().tolist(),
                              'squares': state[name]['squares'].cpu().tolist()} for name in STRATA}
            filename = f'unit-{index:02d}-expert-{expert_index:02d}.bin'
            path = output / 'moments' / filename
            _write_moments(path, index, moments)
            _, decoded = decode_layer(path.read_bytes(), moments=True)
            norms = expert.down_proj.weight.detach().float().norm(dim=0).cpu().tolist()
            _write_scores(output / 'scores' / filename, index, contribution_scores(decoded, norms))
        routing_counts.append(coverage)
    manifest = {'scorer_id': SCORE, 'model': model_config, 'dimensions': dims, 'units': units(dims),
                'strata': list(STRATA), 'source': config['calibration']['data'],
                'source_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}
    (output / 'score-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'statistics-manifest.json').write_text(json.dumps({
        'model': model_config, 'dimensions': dims, 'sample_count': len(samples),
        'sample_ids': [s['stable_row_id'] for s in samples], 'generation': evaluation['generation'],
        'evaluation_disjoint': {'sample_ids': True, 'image_paths': True},
        'routing_counts': routing_counts, 'stratum_denominators': collector.counts,
        'moments': 'routing_weighted_raw_sums_and_sum_squares_fp32',
        'denominator': 'all_tokens_per_stratum_including_unrouted_tokens',
        'native_expert_input_order_verified': True, 'context': context,
    }, indent=2) + '\n')
    # Coverage failures preserve the raw evidence and do not qualify calibration.
    load_calibration(output, dims)
    print(json.dumps({'status': 'completed', 'calibration': str(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
