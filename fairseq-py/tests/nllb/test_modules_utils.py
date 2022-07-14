import numpy as np

test_length = 70000
test_dim = 1024
test_dtype = np.float32
test_lang = "bn"
test_idx_type = "OPQ64,IVF65536,PQ64"


def generate_embedding(
    file=None, emb_length: int = test_length, dim: int = test_dim, dtype=test_dtype
):
    data = np.random.randn(emb_length, dim).astype(dtype)
    if file is not None:
        data.tofile(file)
    return data
