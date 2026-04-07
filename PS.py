# PS.py
import numpy as np

def sum(matrices):
    if not matrices:
        raise ValueError("No matrices provided")

    # check shape consistency
    first_shape = matrices[0].shape
    for m in matrices:
        if m.shape != first_shape:
            raise ValueError("All matrices must have the same shape for sum")

    result = np.zeros_like(matrices[0])

    for m in matrices:
        result = result + m

    return result

def multiply(matrices):
    if not matrices:
        raise ValueError("No matrices provided")

    result = matrices[0]

    for i in range(1, len(matrices)):
        m = matrices[i]

        # shape check: (n x k) @ (k x m)
        if result.shape[1] != m.shape[0]:
            raise ValueError(
                f"Incompatible shapes: {result.shape} and {m.shape}"
            )

        result = result @ m

    return result
