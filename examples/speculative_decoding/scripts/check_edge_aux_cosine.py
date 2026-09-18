#!/usr/bin/env python3
"""Cosine-check Cosmos3-Edge DFlash aux layers: HF block outputs vs vLLM capture ids.

Training reads HF ``hidden_states[lid + 1]`` for ``dflash_config.target_layer_ids``.
vLLM 0.27 first does ``i + 1``, then ``edge_compat`` remaps Edge to ``2 * (i + 1)``
because each HF decoder block is split into attention then MLP.

Capture id ``k`` in Nemotron-H is taken *after* vLLM ``self.layers[k - 1]`` as
``hidden_states + residual`` (Eagle3). For HF block ``lid`` that is MLP layer
``2 * lid + 1``, capture id ``2 * (lid + 1)``.

Phases (separate interpreters: train overlay vs vLLM ``edge_compat``):

    python check_edge_aux_cosine.py --phase hf --save-dir DIR ...
    python check_edge_aux_cosine.py --phase vllm --save-dir DIR ...
    python check_edge_aux_cosine.py --phase report --save-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn


PROMPT = "Write a one-sentence summary of speculative decoding."
PASS_THRESHOLD = 0.95


def _parse_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def vllm_plus1(target_layer_ids: list[int]) -> tuple[int, ...]:
    return tuple(i + 1 for i in target_layer_ids)


def edge_remap(target_layer_ids: list[int]) -> tuple[int, ...]:
    return tuple(2 * (i + 1) for i in target_layer_ids)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    denom = float(a.norm() * b.norm())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _last_token(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        return t[0, -1]
    if t.ndim == 2:
        return t[-1]
    return t.reshape(-1, t.shape[-1])[-1]


def _seq_mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim == 3:
        a = a[0]
    if b.ndim == 3:
        b = b[0]
    n = min(a.shape[0], b.shape[0])
    vals = [_cosine(a[i], b[i]) for i in range(n)]
    return float(sum(vals) / max(len(vals), 1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=("hf", "vllm", "report"))
    p.add_argument("--model-path", default=os.environ.get("EDGE_MODEL_PATH", ""))
    p.add_argument("--draft-path", default=os.environ.get("EXPORT_PATH", ""))
    p.add_argument("--save-dir", required=True)
    p.add_argument(
        "--target-layer-ids",
        default="1,7,13,19,25",
        help="DFlash HF block ids from export dflash_config.target_layer_ids",
    )
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--load-draft", action="store_true")
    p.add_argument("--pass-threshold", type=float, default=PASS_THRESHOLD)
    return p.parse_args()


def _chat_ids(model_path: str, prompt: str) -> list[int]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    try:
        encoded = processor.apply_chat_template(
            messages, enable_thinking=True, **kwargs
        )
    except TypeError:
        encoded = processor.apply_chat_template(messages, **kwargs)
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]
    if isinstance(encoded, dict):
        ids = encoded["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if isinstance(encoded, list):
        return [int(x) for x in encoded]
    raise TypeError(f"Unexpected chat template output: {type(encoded)}")


def phase_hf(args: argparse.Namespace) -> None:
    from transformers import AutoModelForImageTextToText

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    ids = _chat_ids(args.model_path, args.prompt)
    torch.save(ids, save / "input_ids.pt")
    (save / "prompt.txt").write_text(args.prompt)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states
    packed = {
        "len": len(hs),
        "shapes": [tuple(t.shape) for t in hs],
    }
    tensors = {str(i): t.detach().float().cpu() for i, t in enumerate(hs)}
    torch.save(tensors, save / "hf_hidden_states.pt")
    (save / "hf_meta.json").write_text(json.dumps(packed, indent=2) + "\n")
    print("hf_hidden_states", packed["len"], "seq", len(ids), flush=True)


def _unwrap_text_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "language_model"):
        model = model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model = model.model
    return model


def _install_vllm_hooks(model: nn.Module) -> dict:
    text = _unwrap_text_model(model)
    layers = getattr(text, "layers", None)
    if layers is None:
        raise RuntimeError("vLLM model has no .layers after unwrap")
    store: dict[str, torch.Tensor] = {}
    names: list[str] = []
    model._edge_aux_cosine_store = store
    model._edge_aux_layer_names = names
    model._edge_aux_aux_ids = tuple(getattr(text, "aux_hidden_state_layers", ()) or ())

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states, residual = output[0], output[1]
                value = (
                    hidden_states
                    if residual is None
                    else hidden_states + residual
                )
            else:
                value = output[0] if isinstance(output, tuple) else output
            store[str(layer_idx)] = value.detach().float().cpu().contiguous()

        return hook

    for idx, layer in enumerate(layers):
        cls_name = type(layer).__name__
        names.append(f"{idx}:{cls_name}")
        if "Missing" in cls_name or "Placeholder" in cls_name:
            continue
        layer.register_forward_hook(make_hook(idx), with_kwargs=True)
    return {
        "n_layers": len(layers),
        "names": names,
        "aux_hidden_state_layers": list(model._edge_aux_aux_ids),
    }


def _export_vllm_store(model: nn.Module) -> dict:
    store = getattr(model, "_edge_aux_cosine_store", {})
    return {
        "keys": sorted(int(k) for k in store),
        "shapes": {k: list(v.shape) for k, v in store.items()},
        "aux_hidden_state_layers": list(getattr(model, "_edge_aux_aux_ids", ())),
        "names": list(getattr(model, "_edge_aux_layer_names", [])),
    }


def phase_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    save = Path(args.save_dir)
    ids = torch.load(save / "input_ids.pt", weights_only=False)
    if not isinstance(ids, list):
        ids = list(ids)

    llm_kwargs = dict(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.45 if args.load_draft else 0.35,
        disable_log_stats=True,
    )
    if args.load_draft:
        if not args.draft_path:
            raise SystemExit("--load-draft requires --draft-path / EXPORT_PATH")
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_path,
            "num_speculative_tokens": 7,
        }
        llm_kwargs["gpu_memory_utilization"] = 0.70

    llm = LLM(**llm_kwargs)
    meta = llm.apply_model(_install_vllm_hooks)[0]
    (save / "vllm_structure.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("vllm_structure", json.dumps(meta), flush=True)

    llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(max_tokens=1, temperature=0.0),
    )

    def _steal(model: nn.Module) -> dict[str, torch.Tensor]:
        return dict(getattr(model, "_edge_aux_cosine_store", {}))

    captures = llm.apply_model(_steal)[0]
    torch.save(captures, save / "vllm_layer_captures.pt")
    info = llm.apply_model(_export_vllm_store)[0]
    (save / "vllm_capture_meta.json").write_text(json.dumps(info, indent=2) + "\n")
    print("vllm_captures", sorted(int(k) for k in captures), flush=True)


def phase_report(args: argparse.Namespace) -> None:
    save = Path(args.save_dir)
    target_ids = _parse_ids(args.target_layer_ids)
    remapped = edge_remap(target_ids)
    plus1 = vllm_plus1(target_ids)
    hf = torch.load(save / "hf_hidden_states.pt", weights_only=False)
    vllm_cap = torch.load(save / "vllm_layer_captures.pt", weights_only=False)
    struct = json.loads((save / "vllm_structure.json").read_text()) if (save / "vllm_structure.json").exists() else {}

    rows = []
    all_pass = True
    for lid in target_ids:
        hf_t = hf[str(lid + 1)]
        capture_id = 2 * (lid + 1)
        layer_idx = capture_id - 1
        v_t = vllm_cap.get(str(layer_idx))
        if v_t is None:
            rows.append(
                {
                    "target_layer_id": lid,
                    "hf_index": lid + 1,
                    "remap_capture_id": capture_id,
                    "vllm_layer_idx": layer_idx,
                    "status": "MISSING_VLLM_CAPTURE",
                }
            )
            all_pass = False
            continue
        last = _cosine(_last_token(hf_t), _last_token(v_t))
        mean = _seq_mean_cosine(hf_t, v_t)
        best_idx = None
        best_cos = -2.0
        for k, vt in vllm_cap.items():
            c = _cosine(_last_token(hf_t), _last_token(vt))
            if c > best_cos:
                best_cos = c
                best_idx = int(k)
        neighbors = {}
        for delta, label in ((-1, "attn_of_same_block"), (1, "attn_of_next_block")):
            alt = vllm_cap.get(str(layer_idx + delta))
            if alt is not None:
                neighbors[label] = round(
                    _cosine(_last_token(hf_t), _last_token(alt)), 6
                )
        ok = last >= args.pass_threshold and best_idx == layer_idx
        all_pass = all_pass and ok
        rows.append(
            {
                "target_layer_id": lid,
                "hf_index": lid + 1,
                "vllm_i_plus_1": lid + 1,
                "remap_capture_id": capture_id,
                "vllm_layer_idx": layer_idx,
                "cosine_last": round(last, 6),
                "cosine_seq_mean": round(mean, 6),
                "best_vllm_layer_idx": best_idx,
                "best_cosine_last": round(best_cos, 6),
                "neighbors": neighbors,
                "pass": ok,
            }
        )

    served_aux = struct.get("aux_hidden_state_layers") or []
    remap_matches_served = (not served_aux) or (tuple(served_aux) == remapped)
    summary = {
        "target_layer_ids": target_ids,
        "vllm_i_plus_1": list(plus1),
        "edge_remap_capture_ids": list(remapped),
        "served_aux_hidden_state_layers": served_aux,
        "remap_matches_served_aux": remap_matches_served,
        "pass_threshold": args.pass_threshold,
        "rows": rows,
        "pass": all_pass and remap_matches_served,
    }
    (save / "cosine_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.phase in ("hf", "vllm") and not args.model_path:
        raise SystemExit("--model-path / EDGE_MODEL_PATH required")
    if args.phase == "hf":
        phase_hf(args)
    elif args.phase == "vllm":
        phase_vllm(args)
    else:
        phase_report(args)


if __name__ == "__main__":
    main()
