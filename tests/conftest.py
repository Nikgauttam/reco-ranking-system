import os

# PyTorch and faiss-cpu each bundle their own OpenMP on macOS, causing a duplicate
# lib abort. Setting this before any imports prevents the crash. On Linux (CI/prod)
# this is not needed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
