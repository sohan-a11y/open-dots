"""OPEN DOTS Phase 1: HDC sparse memory retrofit + Dots ledger."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS (exact removal)
