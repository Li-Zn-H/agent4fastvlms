"""Small real native MLA/MoE fixture for cross-interpreter LMMS IPC checks."""

import json
from pathlib import Path


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    tokens = ["<s>", "</s>", "<pad>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
    native = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=[]))
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=native, bos_token="<s>", eos_token="</s>", pad_token="<pad>")


def cpu_attention(q, k, v, p=0., **kwargs):
    import torch
    assert p == 0.
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def write_parent(path):
    import torch
    from adapters.deepseek_vl2_dense import native_model_class
    from skills.deepseek_vl2_artifact import write_parent as export
    from skills.deepseek_vl2_pruning import dimensions, units, make_recipe, apply_recipe
    descriptor = json.loads(Path("configs/model-deepseek-vl2.json").read_text())
    cls = native_model_class(descriptor)
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    language = dict(vocab_size=len(tokenizer()), hidden_size=32, intermediate_size=64,
        moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
        n_routed_experts=4, n_shared_experts=2, num_experts_per_tok=2, first_k_dense_replace=1,
        n_group=1, topk_group=1, topk_method="noaux_tc", scoring_func="sigmoid",
        max_position_embeddings=128, bos_token_id=0, eos_token_id=1)
    header = DeepseekVLV2Config(tile_tag="2D", global_view_pos="head",
        vision_config=dict(image_size=56, patch_size=14, width=32, layers=1, heads=4, mlp_ratio=2),
        projector_config=dict(input_dim=32, n_embed=32), language_config=language)
    torch.manual_seed(197)
    model = cls(header).float().eval()
    model.language.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
    dims = dimensions(language)
    scores = [[[float(i) for i in range(u["original_width"])] for _ in range(u["experts"])] for u in units(dims)]
    recipe = make_recipe(scores, [48, 16, 32], dims, {"channel_group_size": 8, "protected_layers": []})
    apply_recipe(model, recipe)
    export(model, tokenizer(), recipe, path)
    return model


def inputs(nx=2):
    import torch
    from skills.deepseek_vl2_token import image_layout
    count = image_layout(2, nx, 1)[2]
    ids = torch.tensor([[3] + [98] * count + [4, 5]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                images=torch.arange((nx + 1) * 3 * 56 * 56, dtype=torch.float32).reshape(1, nx + 1, 3, 56, 56) / 32768,
                images_seq_mask=ids == 98, images_spatial_crop=torch.tensor([[[nx, 1]]]))


def serve_fixture():
    import sys
    import torch
    from types import SimpleNamespace
    from multiprocessing.connection import Connection
    from unittest.mock import patch
    from search.deepseek_native_worker import serve
    from adapters.deepseek_vl2_dense import native_classes
    from skills.deepseek_vl2_artifact import load_physical_model
    config = json.loads(sys.argv[2])
    torch.set_num_threads(2)
    native_classes(config)
    with patch("xformers.ops.memory_efficient_attention", cpu_attention):
        serve(Connection(int(sys.argv[1])), config,
              loader=lambda cfg: (load_physical_model(cfg), SimpleNamespace(tokenizer=tokenizer())))
