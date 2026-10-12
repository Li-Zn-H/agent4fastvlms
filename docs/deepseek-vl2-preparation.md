# DeepSeek-VL2 CUDA integration

The full `deepseek-ai/deepseek-vl2` checkpoint has complete local execution
through the original workflow. Final native serving LMMS retains **97.2659%**
of the model's own Dense quality, and matched batch-1 end-to-end inference is
**4.816650x** faster. Both goal targets pass. The six Stack candidates and final
per-task measurements are listed in [status](status.md#deepseek-vl2-cuda-2026-10-12).

## Scope and environments

The checkpoint has approximately 27.48 billion stored parameter elements.
Original weights remain in `models/DeepSeek-VL2`; no expert is deleted.
The current result root is `results/model/deepseek_vl2/rtx_pro6000_gpu072a4781`.
Hardware is an RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, SM120,
driver 590.44.01. The local receipt records the physical GPU identity.
Earlier same-model quality/calibration is authenticated from the previous
`rtx_pro6000` scope; current-card timings are measured separately.

The [model descriptor](../configs/model-deepseek-vl2.json) binds the unmodified
[official implementation](https://github.com/deepseek-ai/DeepSeek-VL2) at commit
`ef9f91e2b6426536b83294c11742c27be66361b1`. Its tracked source identity is checked
before importing native code. Backend dependencies remain isolated:

| Purpose | Python | Requirements |
| --- | --- | --- |
| Original native Dense and LMMS | `.venv-deepseek/bin/python` | [Dense](../requirements-deepseek-vl2.txt), [LMMS](../requirements-lmms-deepseek-vl2.txt) |
| GPTQ export and native quantized reload | `.venv-quant-deepseek/bin/python` | [GPTQModel](../requirements-quantization-deepseek-vl2-gptqmodel.txt) |
| vLLM Runtime/KV, final LMMS and serving | `.venv-runtime-deepseek/bin/python` | [Runtime](../requirements-runtime-deepseek-vl2.txt) |

Dense uses Transformers 4.46.3, Torch 2.10.0/CUDA 13.0 and matching xFormers.
The quantization environment uses Transformers 5.14.1 and GPTQModel 7.5.0;
[the compatibility bridge](../adapters/deepseek_vl2_compat.py) preserves native
MLA/MoE forward, append-only latent cache and positions without modifying Dense.
vLLM uses version 0.26.0 with Torch 2.11.0. Native LMMS datasets remain in the
verified offline cache. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` for execution; zero inherited thread counts are invalid.

## Native geometry and MoE contracts

The decoder has 30 layers: one dense FFN followed by 29 MoE layers. Each MoE
layer retains 72 routed experts, top-6 routing and two shared experts. Original
dense, routed and combined shared intermediate widths are 12288, 1536 and 3072.
F32 routing correction biases and logits remain high precision. Physical SP
slices intermediate channels in dense, routed and shared FFNs; calibration and
hardware/packing checks account for those different topologies.

Each native 384-pixel tile follows this vision path:

```text
[T, 3, 384, 384]
  -> patch14 vision features [T, 729, 1152]
  -> 27 x 27 grid, padded to 28 x 28 for 2 x 2 merging
  -> [T, 196, 4608]
  -> projector [T, 196, 2560]
```

Native assembly adds global/local grid newlines and a view separator. For `nx`
by `ny` local tiles plus the global tile, the image run has
`14 * 15 + 1 + (ny * 14) * (nx * 14 + 1)` positions.
[Token selection](../skills/deepseek_vl2_token.py) scores each tile independently,
preserves layout tokens and original stream order, and compacts features,
placeholders and positions together. Ratio 0.85 keeps 167/196 main features per
tile; total image-run retention also includes protected layout positions.
Native dynamic resolution, images and prompts are preserved.

## Search and final configuration

Dense provides complete Search/Dev, all five native LMMS denominators and
calibration. Independent Token evaluates 11 candidates; its fastest measured
ratio is 0.90 at 1.042057x in the local timing scope. SP completes all 848 hardware
checks and recovers its local quality boundary at 52.1394% stored decoder FFN
removal. Independent GPTQ W4A16 passes native/vLLM reload and Search/Dev quality.
Its native Transformers path is slower than Dense at 0.619508x; this is separate
from the final composed serving result.

Stack freshly calibrates and quantizes each physical Parent before Token,
mechanism/reload checks, timing and complete native LMMS. Five complete quality
rejections trigger the configured paired recovery. The sixth passes at 97.4211%
and ends Stack search after six of ten configured attempts. No Search/Dev
quality screening runs at Stack or subsequent stages.

| Layer | Selected setting |
| --- | --- |
| SP | 28.4587% stored decoder FFN-parameter removal, `sp-0114` |
| Quantization | Fresh Parent GPTQ W4A16, group size 128; 6,552 INT4 Linear modules and 38 BF16 Linear exceptions |
| Token | Keep 0.85, 167/196 main features per native tile, protected layout preserved |
| Runtime | O2, effective compilation mode 3, `FULL_AND_PIECEWISE` CUDA Graphs; encoder compilation disabled |
| Attention/MoE | TRITON_MLA and Marlin, native F32 routers and physical widths observed |
| KV | Auto, batch-1 final quality/timing; independent batch-4/8 throughput retained separately |

Runtime compares four profiles, each with 128 inputs x 3 rounds. Auto KV completes
batch sizes 1/4/8 with the same protocol. E4M3 fails initialization: the Triton
MLA decode kernel requires 102,400 bytes of shared memory per block, beyond
SM120's 101,376-byte limit. This is a retained execution failure with no quality
or timing score; Auto supplies the complete usable branch.

Final serving LMMS evaluates O2/Auto first and passes all 23,349 native samples.
Original prompts, images, sample identities, generation and native scorers
match Dense. Every native score/aggregate is independently recomputed; all
actual visual selections and compacted prompts validate. The serving mean is
97.2659%, 0.1553 percentage points below native Stack. Runtime/KV fallback is
unneeded. KV dtype is observed; cache sequence length is not observed in vLLM.

## Matched end-to-end measurement and inference

The benchmark runs original Transformers Dense and final serving sequentially
on the same physical GPU. Each backend receives the same Search128 images,
questions and order, five warmups and three measured rounds. All 768 measured
requests, prompt identities and round aggregates independently validate.

| Metric | Original Dense | Final serving |
| --- | ---: | ---: |
| Mean request seconds | 0.982524 | 0.203985 |
| P50 request seconds | 0.707654 | 0.162891 |
| P95 request seconds | 2.701954 | 0.385954 |
| Requests/second | 1.017787 | 4.902323 |
| Output tokens/second | 19.409514 | 92.927116 |

Primary acceleration is Dense total request time divided by serving total
request time: **4.816650x**. Output-token throughput improves 4.787709x. Natural
answer lengths differ. The timer includes image opening, native preprocessing,
transfer, vision encoding, generation, answer decoding and synchronization.
Loading, warmup, prefix-cache reset and post-generation prompt audits are
excluded; in-engine Token traces are included. Actual selections run for all
389 serving warmup/measurement requests; no cross-request image/prefix reuse is
used. Serving answers vary across timing rounds for 20/128 inputs, while selected
visual indices stay identical. These inputs measure performance only; quality
belongs to the separate complete native five-task evaluation.

Single-image inference also passes native prompt, Token and observed Runtime/KV
checks. The example correctly describes the pictured man as talking on a cell
phone. Its first request includes shape compilation and is not warmed-up speed
measurement. The final acceptance is `goal_achieved`.

## Evidence and reuse

All paths below are relative to the current result root; generated assets are
local and excluded from the source edition:

| Evidence | Path |
| --- | --- |
| Qualified serving freeze | `stack-serving-lmms/final/serving-freeze.json`, digest `8c315f7c4ea69ac904f0` |
| Complete native serving audit | `preparation/serving-7542041f1a7e91571360-complete-serving-audit.json` |
| Matched raw timing and summary | `serving-use/benchmark/` |
| Matched timing audit | `preparation/final-serving-benchmark-completion.json` |
| Actual image inference audit | `preparation/final-serving-inference-completion.json` |
| Integration/check receipts | `preparation/final-integration-completion.json` |

The [adapter](../adapters/deepseek_vl2.py) now enables the full `deepseek_vl2`
pipeline ID. Registration stays lazy and does not qualify another model/GPU
scope. Existing JSONs and coordinators reuse all eight completed stages without
launching new GPU work:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-deepseek/bin/python -B -m search.pipeline --config configs/pipeline-deepseek-vl2.json
.venv-deepseek/bin/python -B -m search.pipeline --config configs/pipeline-deepseek-vl2.json --execute --resume
.venv-deepseek/bin/python -B -m search.serving --config configs/serving-deepseek-vl2.json --benchmark --execute
```

Execution flags reuse matching completed receipts; follow [serving](serving.md)
for new inference and [the Agent guide](agent-search.md) for a new experiment.
The saved pipeline reuses all eight stages with zero stage invocations; child
process, GPU query and Worker launch guards verify that no GPU work occurs.
All 18 focused DeepSeek registration/native input/serving tests pass, and the
refined reuse prerequisite test passes separately. The full reference CPU run
executes 365 tests (32 skipped), with the same 35 errors and one failure as the
prior 361-test run. No failure IDs are new. Missing Qwen/LLaVA weights, environments
and reference receipts prevent a green full suite; they do not qualify as
DeepSeek execution evidence. Detailed preparation, interruption and cleanup
histories remain under `preparation/`.
