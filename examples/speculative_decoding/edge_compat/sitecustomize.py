"""Load the narrowly scoped Cosmos3-Edge/vLLM compatibility shim."""

from __future__ import annotations

import traceback

try:
    import vllm_cosmos3_edge
except Exception:
    traceback.print_exc()
