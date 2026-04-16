import numpy as np

# adjustable safety limit (MB per matrix)
MAX_MEMORY_MB = 512


def estimate_max_n(dtype=np.float32):
    bytes_per_element = np.dtype(dtype).itemsize
    max_bytes = MAX_MEMORY_MB * 1024 * 1024
    return int((max_bytes / bytes_per_element) ** 0.5)


MAX_N = estimate_max_n()


def generate_matrix(n, dtype=np.float32):
    if n > MAX_N:
        raise ValueError(
            f"Matrix too large: {n} > {MAX_N} (limit ~{MAX_MEMORY_MB}MB)"
        )

    return np.random.rand(n, n).astype(dtype)

if __name__ == "__main__":
    generate_matrix(10000)