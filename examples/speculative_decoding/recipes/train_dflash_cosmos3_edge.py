# Generated from train_dflash_cosmos3_edge.ipynb; do not edit by hand.
# Run with IPython. Python cells and %%bash bodies are preserved verbatim.

# %% [notebook cell 2]
import os
from pathlib import Path

REPO_ROOT = Path.cwd().resolve().parents[2]
SPEC_ROOT = REPO_ROOT / "examples" / "speculative_decoding"
if not (REPO_ROOT / "pyproject.toml").is_file():
    raise RuntimeError(
        f"Run this notebook from examples/speculative_decoding/recipes; got {Path.cwd()}"
    )

TRAINING_DATA = Path(
    os.environ.get("TRAINING_DATA", str(SPEC_ROOT / "CR3_data" / "cosmos3_edge_dflash_train.jsonl"))
).resolve()
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "nvidia/Cosmos3-Edge",
)
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", str(REPO_ROOT / "ckpts" / "cosmos3-edge-dflash-first-run"))
).resolve()
os.environ.update(
    REPO_ROOT=str(REPO_ROOT),
    TRAINING_DATA=str(TRAINING_DATA),
    MODEL_PATH=MODEL_PATH,
    OUTPUT_DIR=str(OUTPUT_DIR),
)
print(f"Training data will be written to: {TRAINING_DATA}")
if TRAINING_DATA.is_file() and TRAINING_DATA.stat().st_size:
    print(f"Existing file size: {TRAINING_DATA.stat().st_size / 2**30:.1f} GiB")
print(f"Training output: {OUTPUT_DIR}")

# %% [notebook cell 4]
import os
import sys
from pathlib import Path

SPEC_ROOT = Path.cwd().resolve().parent
REPO_ROOT = SPEC_ROOT.parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(SPEC_ROOT / "CR3_data"))).resolve()
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "nvidia/Cosmos3-Edge",
)
TRAINING_DATA = Path(
    os.environ.get("TRAINING_DATA", str(DATA_ROOT / "cosmos3_edge_dflash_train.jsonl"))
).resolve()
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", str(REPO_ROOT / "ckpts" / "cosmos3-edge-dflash-first-run"))
).resolve()

PAI_SHARDS = Path(
    os.environ.get("PAI_SHARDS", str(DATA_ROOT / "pai_understanding_native_shards"))
).resolve()
VQA_SHARD_PATH = Path(
    os.environ.get("VQA_SHARD_PATH", str(DATA_ROOT / "vqa_v2_train_shards"))
).resolve()
TEXT_SHARDS = Path(
    os.environ.get("TEXT_SHARDS", str(DATA_ROOT / "specdec_multilingual_prompt_full_shards"))
).resolve()
PAI_OUTPUT = Path(
    os.environ.get("PAI_OUTPUT", str(SPEC_ROOT / "edge_pai_understanding_synthetic_outputs"))
).resolve()
VQA_OUTPUT = Path(
    os.environ.get("VQA_OUTPUT", str(SPEC_ROOT / "edge_vqa_v2_synthetic_outputs"))
).resolve()
MULTILINGUAL_OUTPUT = Path(
    os.environ.get(
        "MULTILINGUAL_OUTPUT", str(SPEC_ROOT / "edge_specdec_multilingual_prompt_synthetic_outputs")
    )
).resolve()
PLAIN_TEXT_INPUT = os.environ.get("PLAIN_TEXT_INPUT", "")

PAI_REVISION = os.environ.get("PAI_REVISION", "")
TEXT_PROMPT_REVISION = os.environ.get("TEXT_PROMPT_REVISION", "")
PAI_NUM_GENERATION_SHARDS = int(os.environ.get("PAI_NUM_GENERATION_SHARDS", "5"))
PAI_LINES_PER_SHARD = int(os.environ.get("PAI_LINES_PER_SHARD", "128"))
PAI_SHUFFLE_SEED = int(os.environ.get("PAI_SHUFFLE_SEED", "42"))
VQA_NUM_SAMPLES = int(os.environ.get("VQA_NUM_SAMPLES", "20000"))
VQA_SHUFFLE_SEED = int(os.environ.get("VQA_SHUFFLE_SEED", "42"))
DEDUP_WORD_OVERLAP = float(os.environ.get("DEDUP_WORD_OVERLAP", "0.90"))
DEDUP_CACHE_CONTEXTS = int(os.environ.get("DEDUP_CACHE_CONTEXTS", "25000"))
MERGE_WORKERS = int(os.environ.get("MERGE_WORKERS", "8"))

if PAI_NUM_GENERATION_SHARDS <= 0 or PAI_LINES_PER_SHARD <= 0:
    raise ValueError("PAI_NUM_GENERATION_SHARDS and PAI_LINES_PER_SHARD must be positive.")
if not 0 < DEDUP_WORD_OVERLAP <= 1:
    raise ValueError("DEDUP_WORD_OVERLAP must be in (0, 1].")
if DEDUP_CACHE_CONTEXTS <= 0 or MERGE_WORKERS <= 0:
    raise ValueError("DEDUP_CACHE_CONTEXTS and MERGE_WORKERS must be positive.")

for name, value in {
    "REPO_ROOT": REPO_ROOT,
    "DATA_ROOT": DATA_ROOT,
    "MODEL_PATH": MODEL_PATH,
    "TRAINING_DATA": TRAINING_DATA,
    "OUTPUT_DIR": OUTPUT_DIR,
    "PAI_SHARDS": PAI_SHARDS,
    "VQA_SHARD_PATH": VQA_SHARD_PATH,
    "TEXT_SHARDS": TEXT_SHARDS,
    "PAI_OUTPUT": PAI_OUTPUT,
    "VQA_OUTPUT": VQA_OUTPUT,
    "MULTILINGUAL_OUTPUT": MULTILINGUAL_OUTPUT,
    "PLAIN_TEXT_INPUT": PLAIN_TEXT_INPUT,
    "PAI_REVISION": PAI_REVISION,
    "TEXT_PROMPT_REVISION": TEXT_PROMPT_REVISION,
    "PYTHON_BIN": sys.executable,
    "PAI_NUM_GENERATION_SHARDS": PAI_NUM_GENERATION_SHARDS,
    "PAI_LINES_PER_SHARD": PAI_LINES_PER_SHARD,
    "PAI_SHUFFLE_SEED": PAI_SHUFFLE_SEED,
    "VQA_NUM_SAMPLES": VQA_NUM_SAMPLES,
    "VQA_SHUFFLE_SEED": VQA_SHUFFLE_SEED,
    "DEDUP_WORD_OVERLAP": DEDUP_WORD_OVERLAP,
    "DEDUP_CACHE_CONTEXTS": DEDUP_CACHE_CONTEXTS,
    "MERGE_WORKERS": MERGE_WORKERS,
}.items():
    os.environ[name] = str(value)

print(f"Training data: {TRAINING_DATA}")
print(f"Training output: {OUTPUT_DIR}")

# %% [notebook cell 7]
get_ipython().run_cell_magic('bash', '', '''# CPU-only node: download PAI-Understanding and make native-video prompt shards.
set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${PYTHON_BIN:?Run the Step 2 configuration cell before this cell.}"
: "${PAI_NUM_GENERATION_SHARDS:?Run the Step 2 configuration cell before this cell.}"
PAI_ROOT=${PAI_ROOT:-$DATA_ROOT/pai_understanding}
PAI_SHARDS=${PAI_SHARDS:-$SPEC/CR3_data/pai_understanding_native_shards}
PAI_LINES_PER_SHARD=${PAI_LINES_PER_SHARD:-128}
PAI_NUM_SAMPLES=${PAI_NUM_SAMPLES:-$((PAI_NUM_GENERATION_SHARDS * PAI_LINES_PER_SHARD))}
PAI_REVISION_ARGS=()
[ -n "${PAI_REVISION:-}" ] && PAI_REVISION_ARGS+=(--revision "$PAI_REVISION")
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DATA_ROOT/.hf_datasets_cache}

# Invoke the CLI through the active kernel interpreter to avoid a stale `hf` executable on PATH.
"$PYTHON_BIN" -m huggingface_hub.cli.hf --version
# If networking on the CPU-only node stalls, retry this command with HF_HUB_DISABLE_XET=1.
"$PYTHON_BIN" -m huggingface_hub.cli.hf download shi-labs/physical-ai-bench-understanding \
  --repo-type dataset \
  --local-dir "$PAI_ROOT" \
  "${PAI_REVISION_ARGS[@]}" \
  --max-workers 8

HF_DATASETS_CACHE="$HF_DATASETS_CACHE" "$PYTHON_BIN" "$SPEC/recipes/prepare_multimodal_synthetic_shards.py" \
  --dataset pai_understanding \
  --dataset_dir "$PAI_ROOT" \
  --media_root "$PAI_ROOT" \
  --output_dir "$PAI_SHARDS" \
  --max_lines_per_shard "$PAI_LINES_PER_SHARD" \
  --num_samples "$PAI_NUM_SAMPLES" \
  --shuffle_seed "$PAI_SHUFFLE_SEED" \
  --overwrite''')

# %% [notebook cell 8]
get_ipython().run_cell_magic('bash', '', '''# Compute node: run a representative PAI shard slice in an existing Slurm GPU allocation.
set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${MODEL_PATH:?Set MODEL_PATH and run Step 2 before this compute-node cell.}"
PAI_ROOT=${PAI_ROOT:-$DATA_ROOT/pai_understanding}
export MODEL_PATH
export DATASET_DIR="$PAI_ROOT"
export MEDIA_ROOT="$PAI_ROOT"
export SHARD_PATH=${PAI_SHARDS:-$SPEC/CR3_data/pai_understanding_native_shards}
export PREPARE_SHARDS=0
export BACKEND=vllm
export CONTAINER_IMAGE=vllm/vllm-openai:v0.27.0
export API_MODE=openai
export CONTAINER_PYTHONPATH=/scripts/edge_compat
export ASSISTANT_PREFIX=$'<think>\n'
export OUTPUT_PATH="${PAI_OUTPUT:?Run the Step 2 configuration cell before this cell.}"
NODE_NAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
: "${PAI_NUM_GENERATION_SHARDS:?Run the Step 2 configuration cell before this cell.}"
PAI_START_SHARD=${PAI_START_SHARD:-0}
PAI_NUM_AVAILABLE_SHARDS=$(find "$SHARD_PATH" -maxdepth 1 -type f -name 'train-*.jsonl' | wc -l)
PAI_NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
[ "$PAI_NUM_AVAILABLE_SHARDS" -gt 0 ] || { echo "No PAI shards found in $SHARD_PATH" >&2; exit 1; }
[ "$PAI_NUM_NODES" -gt 0 ] || { echo "No allocated nodes found" >&2; exit 1; }
(( PAI_START_SHARD + PAI_NUM_GENERATION_SHARDS <= PAI_NUM_AVAILABLE_SHARDS )) || { echo "Requested PAI shard range exceeds $PAI_NUM_AVAILABLE_SHARDS available shards" >&2; exit 1; }
(( PAI_NUM_GENERATION_SHARDS % PAI_NUM_NODES == 0 )) || { echo "$PAI_NUM_GENERATION_SHARDS selected PAI shards cannot be distributed evenly over $PAI_NUM_NODES nodes" >&2; exit 1; }
PAI_SHARDS_PER_NODE=$(( PAI_NUM_GENERATION_SHARDS / PAI_NUM_NODES ))

"$SPEC/recipes/run_multimodal_synthetic_generation.sh" \
  pai_understanding "$SLURM_JOB_ID" "$PAI_START_SHARD" "$PAI_SHARDS_PER_NODE" "$NODE_NAMES"''')

# %% [notebook cell 10]
get_ipython().run_cell_magic('bash', '', '''# CPU-only node: fetch the VQA v2 train questions, annotations, and COCO images.
set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${PYTHON_BIN:?Run the Step 2 configuration cell before this cell.}"
VQA_ROOT=${VQA_ROOT:-$DATA_ROOT/vqa_v2}
IMAGE_ROOT="$VQA_ROOT/images"
: "${VQA_NUM_SAMPLES:?Run the Step 2 configuration cell before this cell.}"
VQA_SHARD_PATH=${VQA_SHARD_PATH:-$SPEC/CR3_data/vqa_v2_train_shards}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DATA_ROOT/.hf_datasets_cache}
VQA_QUESTIONS_URL=${VQA_QUESTIONS_URL:-https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip}
VQA_ANNOTATIONS_URL=${VQA_ANNOTATIONS_URL:-https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip}
COCO_TRAIN_URL=${COCO_TRAIN_URL:-https://images.cocodataset.org/zips/train2014.zip}

extract_zip() {
  local archive=$1 destination=$2 marker=$3
  if [ -f "$marker" ]; then
    echo "Already extracted: $archive"
    return
  fi
  if command -v unzip >/dev/null 2>&1; then
    unzip -n "$archive" -d "$destination"
  else
    # Minimal CPU-only node images may omit `unzip`; Python provides a compatible fallback.
    "$PYTHON_BIN" -m zipfile -e "$archive" "$destination"
  fi
  touch "$marker"
}

mkdir -p "$VQA_ROOT" "$IMAGE_ROOT"
curl --proto '=https' -L --fail --retry 5 -C - -o "$VQA_ROOT/v2_Questions_Train_mscoco.zip" \
  "$VQA_QUESTIONS_URL"
curl --proto '=https' -L --fail --retry 5 -C - -o "$VQA_ROOT/v2_Annotations_Train_mscoco.zip" \
  "$VQA_ANNOTATIONS_URL"
curl --proto '=https' -L --fail --retry 5 -C - -o "$IMAGE_ROOT/train2014.zip" \
  "$COCO_TRAIN_URL"
extract_zip "$VQA_ROOT/v2_Questions_Train_mscoco.zip" "$VQA_ROOT" "$VQA_ROOT/.questions.extracted"
extract_zip "$VQA_ROOT/v2_Annotations_Train_mscoco.zip" "$VQA_ROOT" "$VQA_ROOT/.annotations.extracted"
extract_zip "$IMAGE_ROOT/train2014.zip" "$IMAGE_ROOT" "$IMAGE_ROOT/.train2014.extracted"

HF_DATASETS_CACHE="$HF_DATASETS_CACHE" "$PYTHON_BIN" "$SPEC/recipes/prepare_multimodal_synthetic_shards.py" \
  --dataset vqa_v2 \
  --vqa_root "$VQA_ROOT" \
  --image_root "$IMAGE_ROOT" \
  --vqa_splits train \
  --num_samples "$VQA_NUM_SAMPLES" \
  --shuffle_seed "$VQA_SHUFFLE_SEED" \
  --output_dir "$VQA_SHARD_PATH" \
  --overwrite''')

# %% [notebook cell 11]
get_ipython().run_cell_magic('bash', '', '''# Compute node: distribute all remaining prepared VQA shards across the allocated nodes.
set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${MODEL_PATH:?Set MODEL_PATH and run Step 2 before this compute-node cell.}"
export MODEL_PATH
VQA_ROOT=${VQA_ROOT:-$DATA_ROOT/vqa_v2}
IMAGE_ROOT="$VQA_ROOT/images"
export VQA_ROOT IMAGE_ROOT
export SHARD_PATH=${VQA_SHARD_PATH:-$SPEC/CR3_data/vqa_v2_train_shards}
export PREPARE_SHARDS=0
export BACKEND=vllm
export CONTAINER_IMAGE=vllm/vllm-openai:v0.27.0
export API_MODE=openai
export CONTAINER_PYTHONPATH=/scripts/edge_compat
export ASSISTANT_PREFIX=$'<think>\n'
export OUTPUT_PATH="${VQA_OUTPUT:?Run the Step 2 configuration cell before this cell.}"
NODE_NAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
VQA_NUM_SHARDS=$(find "$SHARD_PATH" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
VQA_NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
VQA_START_SHARD=${VQA_START_SHARD:-0}
[ "$VQA_NUM_SHARDS" -gt 0 ] || { echo "No VQA shards found in $SHARD_PATH" >&2; exit 1; }
[ "$VQA_NUM_NODES" -gt 0 ] || { echo "No allocated nodes found" >&2; exit 1; }
[[ "$VQA_START_SHARD" =~ ^[0-9]+$ ]] || { echo "VQA_START_SHARD must be a non-negative integer: $VQA_START_SHARD" >&2; exit 1; }
(( VQA_START_SHARD >= 0 && VQA_START_SHARD < VQA_NUM_SHARDS )) || { echo "VQA_START_SHARD=$VQA_START_SHARD is outside 0 through $((VQA_NUM_SHARDS - 1))" >&2; exit 1; }
VQA_REMAINING_SHARDS=$(( VQA_NUM_SHARDS - VQA_START_SHARD ))
# The multimodal worker skips its small rounded-up tail, so all remaining shards are covered.
VQA_SHARDS_PER_NODE=$(( (VQA_REMAINING_SHARDS + VQA_NUM_NODES - 1) / VQA_NUM_NODES ))

"$SPEC/recipes/run_multimodal_synthetic_generation.sh" \
  vqa_v2 "$SLURM_JOB_ID" "$VQA_START_SHARD" "$VQA_SHARDS_PER_NODE" "$NODE_NAMES"''')

# %% [notebook cell 13]
get_ipython().run_cell_magic('bash', '', '''# CPU-only node: download the text prompts and create full prompt shards.
set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${PYTHON_BIN:?Run the Step 2 configuration cell before this cell.}"
TEXT_ROOT=${TEXT_ROOT:-$DATA_ROOT/specdec_multilingual_prompt}
TEXT_SHARDS=${TEXT_SHARDS:-$SPEC/CR3_data/specdec_multilingual_prompt_full_shards}
TEXT_REVISION_ARGS=()
[ -n "${TEXT_PROMPT_REVISION:-}" ] && TEXT_REVISION_ARGS+=(--revision "$TEXT_PROMPT_REVISION")
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DATA_ROOT/.hf_datasets_cache}

# Only default.jsonl is used below; the repository's convenience samples are intentionally not downloaded.
# If networking on the CPU-only node stalls, retry this command with HF_HUB_DISABLE_XET=1.
"$PYTHON_BIN" -m huggingface_hub.cli.hf download nvidia/Speculative-Decoding-Multilingual-Prompt-v2 default.jsonl \
  --repo-type dataset \
  --local-dir "$TEXT_ROOT" \
  "${TEXT_REVISION_ARGS[@]}" \
  --max-workers 8

HF_DATASETS_CACHE="$HF_DATASETS_CACHE" "$PYTHON_BIN" "$SPEC/recipes/prepare_multimodal_synthetic_shards.py" \
  --dataset specdec_multilingual_prompt \
  --text_data "$TEXT_ROOT" \
  --max_lines_per_shard 1024 \
  --overwrite \
  --output_dir "$TEXT_SHARDS"''')

# %% [notebook cell 14]
get_ipython().run_cell_magic('bash', '', '''# Compute node: use the allocation dynamically; no fixed node list is needed.
# START_SHARD=1059 is a resume example. Set it to 0 for a new full run.
set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${MODEL_PATH:?Set MODEL_PATH and run Step 2 before this compute-node cell.}"
export MODEL_PATH
TEXT_ROOT=${TEXT_ROOT:-$DATA_ROOT/specdec_multilingual_prompt}
export TEXT_DATA="$TEXT_ROOT"
export SHARD_PATH=${TEXT_SHARDS:-$SPEC/CR3_data/specdec_multilingual_prompt_full_shards}
export OUTPUT_PATH="${MULTILINGUAL_OUTPUT:?Run the Step 2 configuration cell before this cell.}"
export PREPARE_SHARDS=0
export BACKEND=vllm
export CONTAINER_IMAGE=vllm/vllm-openai:v0.27.0
export CONTAINER_PYTHONPATH=/scripts/edge_compat
export ASSISTANT_PREFIX=$'<think>\n'
export SGLANG_TP_SIZE=1
export NUM_TEMPERATURES=8
NODE_NAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
START_SHARD=${START_SHARD:-0}
TEXT_NUM_SHARDS=$(find "$SHARD_PATH" -maxdepth 1 -type f -name 'train-*.jsonl' | wc -l)
TEXT_NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
[ "$TEXT_NUM_SHARDS" -gt 0 ] || { echo "No text shards found in $SHARD_PATH" >&2; exit 1; }
[ "$TEXT_NUM_NODES" -gt 0 ] || { echo "No allocated nodes found" >&2; exit 1; }
[ "$START_SHARD" -lt "$TEXT_NUM_SHARDS" ] || { echo "START_SHARD=$START_SHARD is beyond $TEXT_NUM_SHARDS text shards" >&2; exit 1; }
TEXT_REMAINING_SHARDS=$(( TEXT_NUM_SHARDS - START_SHARD ))
# The text worker skips its small rounded-up tail, so all remaining shards are covered.
JOBS_PER_NODE=${JOBS_PER_NODE:-$(( (TEXT_REMAINING_SHARDS + TEXT_NUM_NODES - 1) / TEXT_NUM_NODES ))}

"$SPEC/recipes/run_multimodal_synthetic_generation.sh" \
  specdec_multilingual_prompt "$SLURM_JOB_ID" "$START_SHARD" "$JOBS_PER_NODE" "$NODE_NAMES"''')

# %% [notebook cell 16]
get_ipython().run_cell_magic('bash', '', '''set -euo pipefail
SPEC="$(cd .. && pwd)"
: "${DATA_ROOT:?Run the Step 2 configuration cell before this cell.}"
: "${PYTHON_BIN:?Run the Step 2 configuration cell before this cell.}"
PAI_ROOT=${PAI_ROOT:-$DATA_ROOT/pai_understanding}
VQA_ROOT=${VQA_ROOT:-$DATA_ROOT/vqa_v2}
: "${PAI_OUTPUT:?Run the Step 2 configuration cell before this cell.}"
: "${VQA_OUTPUT:?Run the Step 2 configuration cell before this cell.}"
: "${PLAIN_TEXT_INPUT:?Set PLAIN_TEXT_INPUT in the Step 2 configuration cell before this cell.}"
: "${MULTILINGUAL_OUTPUT:?Run the Step 2 configuration cell before this cell.}"
: "${TRAINING_DATA:?Run the Step 2 configuration cell before this cell.}"
: "${DEDUP_WORD_OVERLAP:?Run the Step 2 configuration cell before this cell.}"
: "${DEDUP_CACHE_CONTEXTS:?Run the Step 2 configuration cell before this cell.}"
: "${MERGE_WORKERS:?Run the Step 2 configuration cell before this cell.}"

"$PYTHON_BIN" "$SPEC/recipes/merge_dflash_datasets.py" \
  --source "pai_understanding=$PAI_OUTPUT" \
  --source "vqa_v2=$VQA_OUTPUT" \
  --source "curated_text=$PLAIN_TEXT_INPUT" \
  --source "specdec_multilingual_prompt=$MULTILINGUAL_OUTPUT" \
  --media-root "pai_understanding=$PAI_ROOT" \
  --media-root "vqa_v2=$VQA_ROOT/images" \
  --output "$TRAINING_DATA" \
  --jobs "$MERGE_WORKERS" \
  --word-overlap "$DEDUP_WORD_OVERLAP" \
  --cache-contexts "$DEDUP_CACHE_CONTEXTS"

wc -l "$TRAINING_DATA"''')

# %% [notebook cell 18]
get_ipython().run_cell_magic('bash', '', '''set -euo pipefail
REPO_ROOT="$(cd ../../.. && pwd)"

: "${MODEL_PATH:?Run Step 1 before this cell.}"
: "${TRAINING_DATA:?Run Steps 1 and 2 before this cell.}"
: "${OUTPUT_DIR:?Run Step 1 before this cell.}"
[ -d "$MODEL_PATH" ] || { echo "Missing MODEL_PATH: $MODEL_PATH" >&2; exit 1; }
[ -s "$TRAINING_DATA" ] || { echo "Missing training JSONL: $TRAINING_DATA" >&2; exit 1; }

export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
export REPO_ROOT MODEL_PATH TRAINING_DATA OUTPUT_DIR TRUST_REMOTE_CODE
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Submit this heredoc directly; no batch-script file is created.
sbatch --export=ALL <<'SBATCH'
#!/usr/bin/env bash
#SBATCH -p interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --account=coreai_tritoninference_triton3
#SBATCH --job-name=cosmos3-edge-dflash
#SBATCH --signal=TERM@120

set -euo pipefail
trap "kill -- -$$" TERM INT HUP

export NUM_GPUS=8
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${SLURM_JOB_ID}}"
export VLM_MIN_PIXELS="${VLM_MIN_PIXELS:-50176}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-802816}"
export VLM_MAX_ASSISTANT_TOKENS="${VLM_MAX_ASSISTANT_TOKENS:-2048}"
export VLM_MAX_PROMPT_TOKENS="${VLM_MAX_PROMPT_TOKENS:-8192}"
export VLM_VIDEO_MIN_PIXELS="${VLM_VIDEO_MIN_PIXELS:-100352}"
export VLM_VIDEO_MAX_PIXELS="${VLM_VIDEO_MAX_PIXELS:-2097152}"

echo "Model: ${MODEL_PATH}"
echo "Training data: ${TRAINING_DATA}"
echo "Output: ${OUTPUT_DIR}"

srun \
  --ntasks=1 \
  --gpus-per-task="${NUM_GPUS}" \
  --kill-on-bad-exit=1 \
  --container-image="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}" \
  --container-workdir=/ \
  --container-mounts="${CONTAINER_MOUNTS:?Set CONTAINER_MOUNTS}" \
  bash -lc '
    set -euo pipefail
    export WANDB_MODE=disabled
    export TOKENIZERS_PARALLELISM=false
    unset NCCL_ASYNC_ERROR_HANDLING
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    mkdir -p "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}"

    cd "${REPO_ROOT}"
    python3 -m pip install -r examples/speculative_decoding/requirements.txt
    # Cosmos3-Edge is not registered by the Nano recipe's Transformers upper bound.
    python3 -m pip install 'transformers @ git+https://github.com/huggingface/transformers.git@df04b012229d50d2b6dfba32c61c3057c3a40ea1'
    python3 -m pip install torchcodec
    python3 -m pip install -e ".[hf]"

    python3 -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NUM_GPUS}" \
      examples/speculative_decoding/main.py \
      --config modelopt_recipes/general/speculative_decoding/dflash.yaml \
      model.model_name_or_path="${MODEL_PATH}" \
      model.trust_remote_code="${TRUST_REMOTE_CODE:-true}" \
      data.data_path="${TRAINING_DATA}" \
      data.vlm_processor="${MODEL_PATH}" \
      data.vlm_img_dir=/ \
      training.output_dir="${OUTPUT_DIR}" \
      training.num_train_epochs=25 \
      training.per_device_train_batch_size=1 \
      training.gradient_accumulation_steps=2 \
      training.training_seq_len=16384 \
      training.answer_only_loss=true \
      training.save_steps=1000 \
      training.save_total_limit=10 \
      training.logging_steps=10 \
      training.dataloader_num_workers=2 \
      training.dataloader_prefetch_factor=2 \
      training.ddp_find_unused_parameters=false \
      training.report_to=none \
      dflash.dflash_block_size=8 \
      dflash.dflash_num_anchors=128 \
      dflash.dflash_loss_objective=decay \
      dflash.dflash_loss_decay_factor=4 \
      dflash.dflash_architecture_config.num_hidden_layers=5 \
      dflash.dflash_architecture_config.num_attention_heads=16 \
      dflash.dflash_architecture_config.num_key_value_heads=8 \
      dflash.dflash_architecture_config.head_dim=128 \
      dflash.dflash_architecture_config.intermediate_size=9216 \
      dflash.dflash_architecture_config.max_position_embeddings=131072 \
      dflash.dflash_architecture_config.rms_norm_eps=1e-05 \
      dflash.dflash_architecture_config.rope_theta=100000000 \
      dflash.dflash_mask_token_id=100
  '
SBATCH''')

# %% [notebook cell 20]
get_ipython().run_cell_magic('bash', '', '''set -euo pipefail
REPO_ROOT="$(cd ../../.. && pwd)"
RUN_DIR=${OUTPUT_DIR:-$REPO_ROOT/ckpts/cosmos3-edge-dflash-first-run}
CKPT_DIR=${CKPT_DIR:?Set CKPT_DIR to a saved checkpoint directory, for example $RUN_DIR/checkpoint-180000}
EXPORT_PATH=${EXPORT_PATH:-$RUN_DIR/export-$(basename "$CKPT_DIR" | sed 's/^checkpoint-//')}

[ -f "$CKPT_DIR/modelopt_state.pth" ] || { echo "Missing DFlash checkpoint: $CKPT_DIR" >&2; exit 1; }
[ ! -e "$EXPORT_PATH" ] || { echo "Export path already exists: $EXPORT_PATH" >&2; exit 1; }
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
EXPORT_ARGS=()
if [[ "${TRUST_REMOTE_CODE,,}" == "true" || "$TRUST_REMOTE_CODE" == "1" ]]; then
  EXPORT_ARGS+=(--trust_remote_code)
fi
python3 "$REPO_ROOT/examples/speculative_decoding/scripts/export_hf_checkpoint.py" \
  --model_path "$CKPT_DIR" \
  --export_path "$EXPORT_PATH" \
  "${EXPORT_ARGS[@]}"

[ -s "$EXPORT_PATH/config.json" ] && [ -s "$EXPORT_PATH/model.safetensors" ] \
  || { echo "Export is incomplete: $EXPORT_PATH" >&2; exit 1; }
echo "DFlash deployment checkpoint: $EXPORT_PATH"''')

