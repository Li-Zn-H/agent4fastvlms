"""Cross-environment native processor/vision reference; no GPU allocation.

Write in the original Dense interpreter, then check with the isolated vLLM
interpreter. The CPU native vision substitutes only the unavailable xFormers
attention kernel with the equivalent PyTorch equation.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def picture(size):
    import numpy as np
    from PIL import Image
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    return Image.fromarray(np.stack(((x * 7 + y) % 256, (x + y * 5) % 256,
                                    (x * 3 + y * 11) % 256), axis=-1).astype('uint8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('write', 'check'))
    parser.add_argument('path')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    path = Path(args.path)
    descriptor = json.loads((ROOT / 'configs/model-deepseek-vl2.json').read_text())
    if args.mode == 'write':
        from adapters.deepseek_vl2_dense import native_model_class, load_processor, prepare_image
        cls = native_model_class(descriptor)
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
        from test_deepseek_quantization import header
        from skills.deepseek_vl2_artifact import config_dict
        torch.manual_seed(879)
        small = header()
        # A nondivisible image/patch shape exercises native floor geometry.
        small['vision_config']['image_size'] = 58
        model = cls(DeepseekVLV2Config(**small)).float().eval()
        model.language.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        pixels = torch.randn(3, 3, 58, 58)
        def attention(q, k, v, p=0., **kwargs):
            assert p == 0.
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        with torch.inference_mode(), patch('xformers.ops.memory_efficient_attention', attention):
            features = model.vision(pixels)
        processor = load_processor(descriptor)
        rows = []
        for size in ((384, 384), (768, 384), (384, 768), (1536, 768), (17, 400)):
            question = 'Describe this image.\nKeep the native formatting.'
            inputs = prepare_image(processor, picture(size), question)
            prompt = processor.unexpanded_prompt(question)
            assert processor.prompt_token_ids(question, size) == inputs['input_ids'][0].tolist()
            rows.append({'size': size, 'prompt': prompt, 'native_inputs': dict(inputs)})
        torch.save({'vision_config': config_dict(model.config)['vision_config'],
                    'vision_weights': model.vision.state_dict(), 'pixels': pixels,
                    'features': features, 'processor_rows': rows}, path)
        report = {'status': 'passed', 'mode': 'write', 'native_vision_grid': list(features.shape),
                  'processor_cases': len(rows), 'cuda_initialized': torch.cuda.is_initialized()}
    else:
        from skills.deepseek_vl2_vllm import build_vision, processor_classes, image_keep_count
        from skills.vllm_token import configure_runtime
        from skills.deepseek_vl2_token import select_image_indices, image_layout
        from adapters.deepseek_vl2_dense import load_processor, prepare_image
        fixture = torch.load(path, map_location='cpu', weights_only=True)
        native_processor = load_processor(descriptor)
        for row in fixture['processor_rows']:
            question = 'Describe this image.\nKeep the native formatting.'
            restored = prepare_image(native_processor, picture(row['size']), question)
            for key, value in row['native_inputs'].items():
                torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
            assert native_processor.prompt_token_ids(question, row['size']) == restored['input_ids'][0].tolist()
            assert native_processor.unexpanded_prompt(question) == row['prompt']
        assert not any(name in sys.modules for name in (
            'deepseek_vl2.models.modeling_deepseek_vl_v2', 'xformers', 'attrdict'))
        model = build_vision(SimpleNamespace(**fixture['vision_config'])).eval()
        model.load_state_dict(fixture['vision_weights'], strict=True)
        with torch.inference_mode():
            actual = model.forward_features(fixture['pixels'])
        torch.testing.assert_close(actual, fixture['features'], rtol=1e-6, atol=1e-6)
        NativeProcessor, _ = processor_classes()
        processor = NativeProcessor.from_pretrained(ROOT / descriptor['path'], local_files_only=True)
        from vllm.config import ModelConfig
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.processing.inputs import ProcessorInputs
        from vllm.multimodal.processing.context import TimingContext
        model_config = ModelConfig(model=str(ROOT / descriptor['path']), trust_remote_code=True,
            dtype='bfloat16', max_model_len=4096, limit_mm_per_prompt={'image': 1},
            model_class_overrides={'DeepseekVLV2ForCausalLM':
                                  'skills.deepseek_vl2_vllm:A4FVLMDeepseekVLV2ForCausalLM'})
        pipeline = MULTIMODAL_REGISTRY.create_processor(model_config)
        selection = {'score_dtype': 'float32', 'coverage_grid': {'rows': 4, 'columns': 4}}
        for ratio in (1., .5, .1):
            runtime = {'keep_ratio': ratio, 'selection': selection}
            configure_runtime(runtime)
            for row in fixture['processor_rows']:
                native = row['native_inputs']
                batch = processor(text=row['prompt'], images=[picture(row['size'])], return_tensors='pt')
                # Original native input preparation casts the unchanged pixels
                # to BF16 immediately before vision execution.
                torch.testing.assert_close(batch['pixel_values'].to(torch.bfloat16), native['images'][0], rtol=0, atol=0)
                torch.testing.assert_close(batch['images_spatial_crop'], native['images_spatial_crop'][0], rtol=0, atol=0)
                nx, ny = [int(v) for v in batch['images_spatial_crop'][0].tolist()]
                count = image_keep_count(14, nx, ny, runtime)
                native_ids = native['input_ids'][0]
                expected = native_ids.tolist()
                locations = (native_ids == processor.image_token_id).nonzero().flatten()
                expected[int(locations[0]):int(locations[-1]) + 1] = [processor.image_token_id] * count
                assert batch['input_ids'][0].tolist() == expected, (ratio, row['size'], row['prompt'])
                prepared = pipeline.apply(ProcessorInputs(prompt=row['prompt'],
                    mm_data_items=pipeline.info.parse_mm_data({'image': [picture(row['size'])]})),
                    TimingContext(enabled=False))
                assert prepared['prompt_token_ids'] == expected
                placeholders = prepared['mm_placeholders']['image']
                assert len(placeholders) == 1 and placeholders[0].length == count
                assert placeholders[0].offset == int(locations[0])
                model_inputs = prepared['mm_kwargs'].get_data()
                torch.testing.assert_close(model_inputs['pixel_values'].to(torch.bfloat16), native['images'][0], rtol=0, atol=0)
                torch.testing.assert_close(model_inputs['images_spatial_crop'], native['images_spatial_crop'][0], rtol=0, atol=0)
                full = image_layout(14, nx, ny)[2]
                features = torch.arange(full * 4, dtype=torch.float32).reshape(full, 4)
                indices = select_image_indices(features, 14, nx, ny, ratio, selection)
                assert len(indices) == count
                assert set(image_layout(14, nx, ny)[1]).issubset(indices.tolist())
        configure_runtime(None)
        report = {'status': 'passed', 'mode': 'check', 'vision_max_abs_error': float((actual-fixture['features']).abs().max()),
                  'processor_cases': len(fixture['processor_rows']) * 3,
                  'scope': 'FP32 native vision features; exact native BF16 pixels, crop geometry, compacted prompt IDs and vLLM multimodal placeholder ranges/model kwargs',
                  'cuda_initialized': torch.cuda.is_initialized()}
    if report['cuda_initialized']:
        raise RuntimeError('CPU fixture unexpectedly initialized CUDA')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
