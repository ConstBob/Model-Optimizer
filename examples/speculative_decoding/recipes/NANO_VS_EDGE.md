# Cosmos3 Nano vs Cosmos3 Edge — DFlash recipe diffs

This is the review surface for an Edge DFlash recipe. Do not treat a JSON diff of `train_dflash_cosmos3_nano.ipynb` as the spec: the Nano notebook hard-codes Qwen3-VL tokenizer IDs and draft shapes. Pointing `MODEL_PATH` at [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) is not enough.

The Nano runbook remains `train_dflash_cosmos3_nano.ipynb` (configure → build data → train → export). The Edge sibling should keep that layout once it exists; this file is the contract for what must change.

**Target checkpoint:** public [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) (`main` at `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`, 2026-08-26). The model card’s **vLLM** section deploys this same repo as the Cosmos3-Edge **reasoner**. Do not switch to NGC unless that revision is shown to differ. The snapshot also contains diffusion `transformer/` + `vae/` (Omni); DFlash uses the AR/VLM reasoner path, not Diffusers.

Sources for Nano: `cosmos3-nano-reasoner` `config.json` (`Qwen3VLForConditionalGeneration`) and the production overrides in `train_dflash_cosmos3_nano.ipynb`. Tokenizer/config below were loaded from the Edge snapshot (`AutoTokenizer`, `trust_remote_code=True`).

## Status

| Step | State |
| --- | --- |
| Frozen Edge reasoner on disk | **Done** (`a9d944e2`, ~8.6 GiB without card `assets/`) |
| Tokenizer / mask probe | **Done** (table below). Smoke mask `100` (`<SPECIAL_100>`) |
| Draft `dflash_architecture_config` from Edge `text_config` | **Done** (`hidden_size=2048`, heads 16/8, `head_dim=128`, 5 draft layers) |
| 1-GPU train + `export_hf_checkpoint.py` | **Done** (2 steps, text-only JSONL, seq 4096). Export is `DFlashDraftModel`, mask `100`, vocab 131072, `target_layer_ids` `[1, 7, 13, 19, 25]`, `num_target_layers=28` |
| vLLM load + generate (`method=dflash`) | **Smoke passed** on vLLM 0.27 + transformers 5.15 (engine loads and generates). Not a `MODEL_PATH` swap: see [vLLM](#vllm-dflash-on-edge) |
| vLLM aux layer ids (needed for correct AL) | **Open** — serve-time mapping bug, not train/export. See [Aux layer mapping](#aux-layer-mapping-train-vs-vllm-serve) |
| Rebuild PAI/VQA JSONL with the Edge tokenizer | **Not started.** Nano shards are not reusable |
| 8-GPU production train (global batch 16) | **Not started** |
| Sibling `train_dflash_cosmos3_edge.ipynb` | **Not started** |

Training needs a transformers build that registers `cosmos3_edge` (5.12 cannot load the class). Do not upgrade the Nano train venv in place; keep an overlay. Do not put that overlay on vLLM’s `PYTHONPATH`.

## Diffs vs Nano

These Nano notebook values must not be copied onto Edge. Swapping only `MODEL_PATH` is not a recipe.

| Item | Nano (locked in the notebook) | Edge (public checkpoint) | Action |
| --- | --- | --- | --- |
| Target class | `Qwen3VLForConditionalGeneration` / `qwen3_vl` | `Cosmos3EdgeForConditionalGeneration` / `cosmos3_edge` | Load Edge with the matching processor; expect `trust_remote_code` |
| Family | Qwen3-VL text tower | Mixture-of-Transformers (card); text tower `cosmos3_edge_text` | Use the **reasoner / AR text+vision** weights, not a policy or generator-only blob |
| Size (card) | ~16B | ~4B | Draft `hidden_size` follows the **text** tower (`2048`), not Nano’s 4096 |
| Tokenizer / vocab | Qwen3, `vocab_size` 151936 | Nemotron-style specials, `vocab_size` 131072 | Rebuild JSONL with the Edge tokenizer |
| `dflash_mask_token_id` | `151669` | Out of range for 131072 | Smoke used **`100`**. Do not reuse 151669 |
| Draft heads | `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128` | Text tower: **16 / 8 / 128**, `hidden_size=2048` | 32 heads × `head_dim=128` does not match `hidden_size=2048` |
| Draft MLP / RoPE | `intermediate_size=12288`, `rms_norm_eps=1e-06`, `rope_theta=5000000`, `max_position_embeddings=262144` | Text: `9216`, `1e-05`, `1e8`, `131072` | Override from Edge `text_config` |
| Activation | Text `silu`; DFlash draft is Qwen3 MLP | Text `relu2` | Draft stays Qwen3-shaped (`silu` in `dflash.md`). `relu2` is not a drop-in draft flag |
| Vision token IDs | `image=151655`, `video=151656`, `vision_start=151652`, `vision_end=151653` | `image=19`, `video=18`, `vision_start=20`, `vision_end=21` | Hard-coded Qwen VL ids will break data / chat templates |
| EOS | `151645` | `11` (`<\|im_end\|>`) | Completions and stop criteria must use Edge ids |
| M-RoPE | Qwen `rope_scaling.mrope_section = [24, 20, 20]` | `rope_parameters.mrope_section = [24, 20, 20]` | **Do not inherit M-RoPE onto the draft**; vLLM speculative decoding still wants standard RoPE on the draft (`doc/dflash.md`) |

## Safe to reuse from Nano (until measured otherwise)

Recipe *process* defaults, not architecture facts:

- Four-step runbook shape
- DFlash block size `8`, `num_anchors=128`, loss `decay` / `decay_factor=4`
- Draft **depth** `num_hidden_layers=5` (independent of the 28-layer Edge text tower; KV injection still needs `hidden_size` alignment)
- `training_seq_len=16384` (must stay divisible by block size). Smoke used 4096 only to fit a short job
- 8 GPU, per-device batch 1, grad accum 2 → **global batch 16** (1-GPU equivalent: accum 16)
- Export via `examples/speculative_decoding/scripts/export_hf_checkpoint.py` — never serve a raw Trainer `checkpoint-*`
- vLLM `method=dflash`, `num_speculative_tokens=7`

## Mask token (Edge)

DFlash reuses the target `embed_tokens`. The mask id must be in-vocab and should not appear in normal prompts.

| Check | Result |
| --- | --- |
| `len(tokenizer)` / `vocab_size` | 131072 |
| `tokenizer.mask_token_id` | **None** (must set in the recipe) |
| Nano `151669` | **out of range** |
| Chat specials | `10=<\|im_start\|>`, `11=<\|im_end\|>`, `12=<think>`, `13=</think>`, `18–21` vision pads |
| Template `<\|…\|>` | only those six vision/chat tags; no `<SPECIAL_*>` in `chat_template.jinja` |
| `<SPECIAL_*>` | ids **22–999** (978 tokens), none appear in the jinja file |

Smoke train+export used **`100` = `<SPECIAL_100>`**. Keep that unless a later eval shows it colliding with real text.

## Data

Nano production JSONL is tokenized / templated for Qwen3-VL. Edge still needs:

1. Edge processor + chat template for PAI / VQA / text sources
2. Completions from the **Edge** target, not Nano
3. A fresh merge (`merge_dflash_datasets.py`); media paths can follow the same absolute-path convention (`data.vlm_img_dir=/`)

## vLLM DFlash on Edge

DFlash is not EAGLE, but vLLM 0.27 serves it on the EAGLE3 plumbing: shared `embed_tokens` / `lm_head`, and **aux hidden states** from selected target layers (`SupportsEagle3`). `Qwen3VLForConditionalGeneration` already implements that interface. `Cosmos3EdgeForConditionalGeneration` does not; a serving smoke has to attach the protocol so the runner can call `set_aux_hidden_state_layers`.

Other 0.27 + transformers 5.15 mismatches that blocked engine init (not DFlash-specific):

- `get_image_size` is on `transformers.image_utils`, not `qwen3_vl` video processing
- DFlash proposer reads `image_token_index`; Edge config only has `image_token_id`
- vLLM’s Edge video processor calls `resize()` with `SizeDict(height, width)`; 5.15 `Qwen3VLVideoProcessor.resize` wants `factor` / `temporal_factor`

Do **not** fix these by putting the train-time transformers overlay on vLLM’s `PYTHONPATH` (EngineCore inherits it and breaks). Local shims around the bench venv were enough for a text generate smoke. 2-step draft output is garbage (`0000…`); that does not validate AL. Engine teardown may hang after a successful generate; that is a process-exit issue, not a load failure.

### Aux layer mapping (train vs vLLM serve)

vLLM **does** support `method=dflash`. The open issue is not “DFlash unsupported on Edge”; it is a **layer-index mismatch** between how ModelOpt trains the draft and how vLLM 0.27 picks aux hidden states at serve time.

DFlash is not EAGLE, but vLLM routes both through the same aux-hidden plumbing (`SupportsEagle3` / `set_aux_hidden_state_layers`). The draft was trained on features from specific **HuggingFace decoder blocks**. At serve time, vLLM must sample the **same physical activations** from `self.layers`. If the mapping is wrong, load and generate still succeed, but acceptance length (AL) drops because the draft sees the wrong input distribution.

#### Training side (ModelOpt + HF)

`build_target_layer_ids(28, 5)` writes HF block indices into the exported draft config, e.g. `[1, 7, 13, 19, 25]` in `dflash_config.target_layer_ids`. During training and AR validation, features are taken from `hidden_states[lid + 1]` — index `0` is the embedding, index `k` is the output after HF decoder block `k - 1` (attn + MLP fused in one block):

```python
# modelopt/torch/speculative/plugins/hf_dflash.py
hid_offset = 1
selected = [base_outputs.hidden_states[lid + hid_offset] for lid in self.target_layer_ids]
target_hidden = torch.cat(selected, dim=-1)
```

Export example (`config.json`):

```json
"dflash_config": {
  "mask_token_id": 100,
  "target_layer_ids": [1, 7, 13, 19, 25]
}
```

These integers are **HF block ids**, not vLLM `self.layers` indices. Do not change them in export to “fix” serving; remap on the vLLM side instead.

#### Serve side (vLLM)

When the engine starts, `gpu_model_runner._get_eagle3_aux_layers_from_config` reads `dflash_config.target_layer_ids` and applies a single conversion for all targets:

```python
# vllm/v1/worker/gpu_model_runner.py
if dflash_config and isinstance(dflash_config, dict):
    # Add 1 to convert DFlash's aux layer id semantics
    layer_ids = [
        i + 1 for i in (dflash_config.get("target_layer_ids") or [])
    ]
```

So `[1, 7, 13, 19, 25]` becomes aux tuple `(2, 8, 14, 20, 26)`. That `+ 1` assumes **one HF decoder layer maps to one vLLM layer** (aux `0` = embedding, aux `k` = after vLLM layer `k - 1`). This holds for **Qwen3-VL (Nano)** and is why Nano never hit this bug.

#### Why Edge breaks the assumption

Cosmos3-Edge’s HF checkpoint has **28** fused text blocks (`text_config.num_hidden_layers: 28`). vLLM’s Edge implementation loads them through Nemotron-H and **splits each HF block into two `self.layers` entries** (attention, then MLP):

```python
# vllm/model_executor/models/cosmos3_edge.py
# checkpoint block N → attention layer 2N, MLP layer 2N + 1
for physical_idx in range(28):
    attention_idx = 2 * physical_idx
    mlp_idx = attention_idx + 1
```

`Cosmos3EdgeTextConfig` sets `num_hidden_layers = 2 * num_hidden_layers` and `hybrid_override_pattern = "*-" * num_hidden_layers`, so the runtime stack has **56** layers while training still refers to **28** HF blocks.

Example for HF block `i = 1`:

| Stage | What “layer 1” means | Index used |
| --- | --- | --- |
| Train / export | Output after HF block 1 (attn + MLP) | `hidden_states[1 + 1]` → HF index `2` |
| vLLM today (`i + 1`) | After vLLM layer `1` (MLP of HF block 0) | aux `2` |
| Likely correct for Edge | After MLP of HF block `i` | aux `2 * (i + 1)` → `4` for `i = 1` |

Full remap for the smoke export ids:

| HF `target_layer_ids` | vLLM today (`i + 1`) | Proposed Edge remap (`2 * (i + 1)`) |
| --- | --- | --- |
| 1 | 2 | 4 |
| 7 | 8 | 16 |
| 13 | 14 | 28 |
| 19 | 20 | 40 |
| 25 | 26 | 52 |

The proposed formula is derived from the 2× block split; **confirm with a cosine check** (HF `hidden_states[lid+1]` vs vLLM aux candidates on the same tokens) before trusting AL numbers.

#### What to fix (and what not to)

- **Fix:** Edge-specific mapping in vLLM (`gpu_model_runner` branch for `cosmos3_edge`) or a serve shim that calls `set_aux_hidden_state_layers` with the remapped tuple. Do not multiply ids inside `config.json` and still rely on vLLM’s `+ 1` (that would double-convert).
- **Do not fix:** Retrain or change `target_layer_ids` in export unless the training hook itself is wrong; the HF semantics are consistent.
- **Nano:** No change; Qwen 1:1 layering matches vLLM’s `i + 1` rule.

## Remaining (before an 8-GPU copy of the Nano job)

1. Implement and verify the Edge aux remap in [Aux layer mapping](#aux-layer-mapping-train-vs-vllm-serve) (vLLM patch or serve shim + cosine check).
2. Rebuild JSONL with the Edge tokenizer and Edge completions.
3. 8-GPU train at global batch 16; do not mix with Nano `train_dflash.sh` (that script hard-codes Qwen heads, mask 151669, and `trust_remote_code=false`).
4. Add `train_dflash_cosmos3_edge.ipynb` from this contract. Upstream PR to `NVIDIA/Model-Optimizer` after that recipe exists.

## Out of scope here

Cluster accounts, scratch paths, Slurm partitions, and experiment numbers stay in the private ops checkout. This file is only the Nano ↔ Edge recipe contract.
