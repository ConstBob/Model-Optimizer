# Cosmos3 Nano vs Cosmos3 Edge — DFlash recipe diffs

This is the review surface for an Edge DFlash recipe. Do not treat a JSON diff of `train_dflash_cosmos3_nano.ipynb` as the spec: the Nano notebook hard-codes Qwen3-VL tokenizer IDs and draft shapes. Pointing `MODEL_PATH` at [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) is not enough.

The Nano runbook remains `train_dflash_cosmos3_nano.ipynb` (same four steps: configure → build data → train → export). The Edge sibling should keep that layout once it exists; this file is the contract for what must change.

**Target checkpoint:** public [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) (`main` at `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`, 2026-08-26). The model card’s **vLLM** section deploys this same repo as the Cosmos3-Edge **reasoner**. Do not switch to NGC unless that revision is shown to differ.

Tokenizer/config below were loaded from that snapshot (`AutoTokenizer`, `trust_remote_code=True`). A 1-GPU 2-step train+export on this checkpoint succeeded (draft `hidden_size=2048`, 16 heads, mask `100`). vLLM serving of target+draft is still blocked on the processor/transformers pin.

Sources for Nano: `cosmos3-nano-reasoner` `config.json` (`Qwen3VLForConditionalGeneration`) and the production overrides in `train_dflash_cosmos3_nano.ipynb`.

## Do not copy

| Item | Nano (locked in the notebook) | Edge (public checkpoint) | Action |
| --- | --- | --- | --- |
| Target class | `Qwen3VLForConditionalGeneration` / `qwen3_vl` | `Cosmos3EdgeForConditionalGeneration` / `cosmos3_edge` | Load Edge with the matching processor; expect `trust_remote_code` |
| Family | Qwen3-VL text tower | Mixture-of-Transformers (card); text tower `cosmos3_edge_text` | Bring-up must confirm the **reasoner / AR text+vision** weights, not a policy or generator-only blob |
| Size (card) | ~16B | ~4B | Smaller target; draft `hidden_size` must follow the **text** tower, not Nano’s 4096 |
| Tokenizer / vocab | Qwen3, `vocab_size` 151936 | Nemotron-style specials, `vocab_size` 131072 | Rebuild JSONL with the Edge tokenizer. Nano shards are not reusable |
| `dflash_mask_token_id` | `151669` | **Invalid** (out of range for 131072) | Pick an **existing reserved** embedding id after checking the chat template (see below). Do not reuse 151669 |
| Draft heads | `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128` | Text tower: **16 / 8 / 128**, `hidden_size=2048` | Copying 32 heads with `head_dim=128` does not match `hidden_size=2048`. Start from the Edge text dims |
| Draft MLP / RoPE | `intermediate_size=12288`, `rms_norm_eps=1e-06`, `rope_theta=5000000`, `max_position_embeddings=262144` | Text: `9216`, `1e-05`, `1e8`, `131072` | Override these from Edge `text_config`, not from the Nano cell |
| Activation | Text `silu`; DFlash draft is Qwen3 MLP | Text `relu2` | Draft implementation is still Qwen3-shaped (`dflash.md`). Do not assume `relu2` is a drop-in `dflash_architecture_config` flag |
| Vision token IDs | `image=151655`, `video=151656`, `vision_start=151652`, `vision_end=151653` | `image=19`, `video=18`, `vision_start=20`, `vision_end=21` (`<|image_pad|>`, `<|video_pad|>`) | Any data / chat-template code that hard-codes Qwen VL ids will break |
| EOS | `151645` | `11` (`<|im_end|>`) | Regenerated completions and stop criteria must use Edge ids |
| M-RoPE | Qwen `rope_scaling.mrope_section = [24, 20, 20]` | `rope_parameters.mrope_section = [24, 20, 20]` | Keep the existing DFlash rule: **do not inherit M-RoPE onto the draft**; vLLM speculative decoding still wants standard RoPE on the draft (`doc/dflash.md`, Qwen3.5 notes) |

## What can stay the same (until measured otherwise)

These are recipe *process* defaults from the Nano notebook, not architecture facts. Reuse them for the first Edge bring-up, then retune.

- Four-step runbook shape
- DFlash block size `8`, `num_anchors=128`, loss `decay` / `decay_factor=4`
- Draft **depth** `num_hidden_layers=5` (layer count is independent of the 28-layer Edge text tower; KV injection still needs `hidden_size` alignment)
- `training_seq_len=16384` (must stay divisible by block size)
- 8 GPU, per-device batch 1, grad accum 2
- Export via `examples/speculative_decoding/scripts/export_hf_checkpoint.py` — never serve a raw Trainer `checkpoint-*`
- vLLM `method=dflash`, `num_speculative_tokens=7`

## Mask token (Edge) — probed, not pinned

DFlash reuses the target `embed_tokens`. The mask id must be in-vocab and should not appear in normal prompts.

Probed on this snapshot:

| Check | Result |
| --- | --- |
| `len(tokenizer)` / `vocab_size` | 131072 |
| `tokenizer.mask_token_id` | **None** (must set in the recipe) |
| Nano `151669` | **out of range** |
| Chat specials | `10=<|im_start|>`, `11=<|im_end|>`, `12=<think>`, `13=</think>`, `18–21` vision pads |
| Template `<\|…\|>` | only those six vision/chat tags; no `<SPECIAL_*>` in `chat_template.jinja` |
| `<SPECIAL_*>` | ids **22–999** (978 tokens), none appear in the jinja file |

A smoke-train candidate is any of 22–999 (e.g. `100` = `<SPECIAL_100>`). **Do not pin it in the Edge notebook** until: (1) the id exists in `embed_tokens` after a real load, (2) a short train+export succeeds.

## Data

Nano production JSONL is tokenized / templated for Qwen3-VL. Edge needs:

1. Edge processor + chat template for PAI / VQA / text sources
2. Completions from the **Edge** target, not Nano
3. A fresh merge (`merge_dflash_datasets.py`); media paths can follow the same absolute-path convention (`data.vlm_img_dir=/`)

## Bring-up order (before an 8-GPU copy of the Nano job)

1. ~~Confirm `MODEL_PATH` is the frozen Edge reasoner.~~ **Done:** HF `nvidia/Cosmos3-Edge` @ `a9d944e2`. Same repo also contains diffusion `transformer/` + `vae/` (Omni). vLLM reasoner serving uses this id; DFlash should load the AR/VLM path, not Diffusers Omni.
2. ~~Tokenizer / mask probe.~~ **Done** (table above). Weights snapshot on disk excluding card `assets/` (~8.6 GiB).
3. Set `dflash_architecture_config` from Edge `text_config` (`hidden_size=2048`, heads 16/8, `head_dim=128`, `intermediate_size=9216`, `rope_theta=1e8`, `rms_norm_eps=1e-05`, `max_position_embeddings=131072`), keep 5 draft layers.
4. 1-GPU smoke: load target → short train → export. **Train+export verified** (mask `100`, draft `hidden_size=2048`, 16 heads). vLLM resolve of target+draft works; serving still blocked on the vLLM Edge processor vs the current transformers pin.
5. Only then rebuild JSONL with the Edge tokenizer and run 8-GPU training.

## Out of scope here

Cluster accounts, scratch paths, Slurm partitions, and experiment numbers stay in the private ops checkout. This file is only the Nano ↔ Edge recipe contract.
