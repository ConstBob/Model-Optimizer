#!/usr/bin/env python3
"""Derive the Cosmos3-Edge recipe directly from the Nano notebook.

This is intentionally a checked transformation rather than a hand-maintained
second recipe. Every replacement must match the Nano source exactly once (or
the script fails), which keeps the Edge notebook traceable to its baseline.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
NANO_NOTEBOOK = HERE / "train_dflash_cosmos3_nano.ipynb"
EDGE_NOTEBOOK = HERE / "train_dflash_cosmos3_edge.ipynb"
EDGE_SCRIPT = HERE / "train_dflash_cosmos3_edge.py"


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else value


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def replace(cell: dict, old: str, new: str, *, count: int = 1) -> None:
    value = source(cell)
    actual = value.count(old)
    if actual != count:
        raise RuntimeError(
            f"Expected {count} occurrence(s) in cell, found {actual}: {old!r}"
        )
    set_source(cell, value.replace(old, new))


def replace_once_re(cell: dict, pattern: str, new: str) -> None:
    value = source(cell)
    updated, n = re.subn(pattern, lambda _: new, value, count=1)
    if n != 1:
        raise RuntimeError(f"Expected 1 match for {pattern!r}, found {n}")
    set_source(cell, updated)


def nano_default_model_path(notebook: dict) -> str:
    match = re.search(r'"(/(?:[^"\\]|\\.)+)"', source(notebook["cells"][2]))
    if match is None:
        raise RuntimeError("Could not find the Nano notebook default MODEL_PATH")
    return match.group(1)


def export_code_cells(notebook: dict, output: Path) -> None:
    chunks = [
        "# Generated from train_dflash_cosmos3_edge.ipynb; do not edit by hand.\n",
        "# Run with IPython. Python cells and %%bash bodies are preserved verbatim.\n\n",
    ]
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        value = source(cell).rstrip()
        chunks.append(f"# %% [notebook cell {index}]\n")
        if value.startswith("%%bash\n"):
            body = value.removeprefix("%%bash\n")
            if "'''" in body:
                raise RuntimeError(f"Cell {index} cannot be safely exported with triple quotes")
            chunks.extend(
                (
                    "get_ipython().run_cell_magic('bash', '', '''",
                    body,
                    "''')\n\n",
                )
            )
        else:
            chunks.extend((value, "\n\n"))
    output.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    nano = json.loads(NANO_NOTEBOOK.read_text(encoding="utf-8"))
    edge = copy.deepcopy(nano)

    # Public Hub id. Do not bake cluster usernames or scratch paths into the recipe.
    edge_model = "nvidia/Cosmos3-Edge"
    replace(edge["cells"][2], nano_default_model_path(nano), edge_model)
    replace(edge["cells"][4], nano_default_model_path(nano), edge_model)
    for old, new in (
        (
            'str(SPEC_ROOT / "pai_understanding_synthetic_outputs")',
            'str(SPEC_ROOT / "edge_pai_understanding_synthetic_outputs")',
        ),
        (
            'str(SPEC_ROOT / "vqa_v2_synthetic_outputs")',
            'str(SPEC_ROOT / "edge_vqa_v2_synthetic_outputs")',
        ),
        (
            'str(SPEC_ROOT / "specdec_multilingual_prompt_synthetic_outputs")',
            'str(SPEC_ROOT / "edge_specdec_multilingual_prompt_synthetic_outputs")',
        ),
    ):
        replace(edge["cells"][4], old, new)

    # Naming/path changes only.
    for cell in edge["cells"]:
        value = source(cell)
        value = value.replace("Cosmos3 Nano", "Cosmos3 Edge")
        value = value.replace("Cosmos3-Nano", "Cosmos3-Edge")
        value = value.replace("cosmos3_nano", "cosmos3_edge")
        value = value.replace("cosmos3-nano", "cosmos3-edge")
        set_source(cell, value)

    # Strip site-specific Nano defaults so the generated Edge notebook is publishable.
    replace_once_re(
        edge["cells"][18],
        r'export HF_HOME="\$\{HF_HOME:-[^}]+\}"',
        'export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"',
    )
    replace_once_re(
        edge["cells"][18],
        r"--container-image=\S+ \\",
        '--container-image="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}" \\',
    )
    replace_once_re(
        edge["cells"][18],
        r"--container-mounts=\S+ \\",
        '--container-mounts="${CONTAINER_MOUNTS:?Set CONTAINER_MOUNTS}" \\',
    )

    # Edge generation needs the proven vLLM compatibility path and must restore
    # the <think> prefix that Edge's generation template pre-fills.
    edge_generation_env = """export BACKEND=vllm
export CONTAINER_IMAGE=vllm/vllm-openai:v0.27.0
export API_MODE=openai
export CONTAINER_PYTHONPATH=/scripts/edge_compat
export ASSISTANT_PREFIX=$'<think>\\n'
"""
    for cell_index, anchor in (
        (8, "export PREPARE_SHARDS=0\n"),
        (11, "export PREPARE_SHARDS=0\n"),
    ):
        replace(edge["cells"][cell_index], anchor, anchor + edge_generation_env)
    replace(
        edge["cells"][14],
        "export BACKEND=vllm\nexport CONTAINER_IMAGE=vllm/vllm-openai:v0.24.0\n",
        "export BACKEND=vllm\n"
        "export CONTAINER_IMAGE=vllm/vllm-openai:v0.27.0\n"
        "export CONTAINER_PYTHONPATH=/scripts/edge_compat\n"
        "export ASSISTANT_PREFIX=$'<think>\\n'\n",
    )

    # Edge requires its registered Transformers class and architecture values.
    replace(
        edge["cells"][17],
        "Remote code is disabled by default; set `TRUST_REMOTE_CODE=true` only for a model checkpoint you trust and that requires custom code.",
        "Remote code is enabled because the pinned Cosmos3-Edge checkpoint requires its registered model code.",
    )
    replace(
        edge["cells"][18],
        'export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"',
        'export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"',
    )
    replace(
        edge["cells"][18],
        'model.trust_remote_code="${TRUST_REMOTE_CODE:-false}"',
        'model.trust_remote_code="${TRUST_REMOTE_CODE:-true}"',
    )
    replace(edge["cells"][18], "#SBATCH --job-name=cosmos3-dflash", "#SBATCH --job-name=cosmos3-edge-dflash")
    replace(
        edge["cells"][18],
        "python3 -m pip install -r examples/speculative_decoding/requirements.txt\n",
        "python3 -m pip install -r examples/speculative_decoding/requirements.txt\n"
        "    # Cosmos3-Edge is not registered by the Nano recipe's Transformers upper bound.\n"
        "    python3 -m pip install "
        "'transformers @ git+https://github.com/huggingface/transformers.git"
        "@df04b012229d50d2b6dfba32c61c3057c3a40ea1'\n",
    )
    for old, new in (
        ("num_attention_heads=32", "num_attention_heads=16"),
        ("intermediate_size=12288", "intermediate_size=9216"),
        ("max_position_embeddings=262144", "max_position_embeddings=131072"),
        ("rms_norm_eps=1e-06", "rms_norm_eps=1e-05"),
        ("rope_theta=5000000", "rope_theta=100000000"),
        ("dflash.dflash_mask_token_id=151669", "dflash.dflash_mask_token_id=100"),
    ):
        replace(edge["cells"][18], old, new)
    replace(
        edge["cells"][20],
        'TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"',
        'TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"',
    )

    replace(
        edge["cells"][21],
        '  --served-model-name "$SERVED_MODEL_NAME" \\\n',
        '  --served-model-name "$SERVED_MODEL_NAME" \\\n'
        "  --trust-remote-code \\\n",
    )
    replace(
        edge["cells"][21],
        "vllm serve \"$MODEL_PATH\" \\\n",
        "PYTHONPATH=\"$REPO_ROOT/examples/speculative_decoding/edge_compat\" "
        "vllm serve \"$MODEL_PATH\" \\\n",
    )
    replace(
        edge["cells"][21],
        "Add `--trust-remote-code` only when serving a model checkpoint you trust and that requires custom code.",
        "The pinned Edge checkpoint requires `--trust-remote-code`; do not use this recipe with an untrusted checkpoint.",
    )

    # Add provenance to the generated notebook without changing executable cells.
    set_source(
        edge["cells"][0],
        source(edge["cells"][0])
        + "\n\n> This notebook is generated by `derive_cosmos3_edge_recipe.py` from the Nano "
        "notebook. Do not edit it by hand.\n",
    )

    EDGE_NOTEBOOK.write_text(
        json.dumps(edge, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    export_code_cells(edge, EDGE_SCRIPT)
    print(f"Wrote {EDGE_NOTEBOOK}")
    print(f"Wrote {EDGE_SCRIPT}")


if __name__ == "__main__":
    main()
