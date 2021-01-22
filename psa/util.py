import lab as B

__all__ = ["cos_sim"]


def cos_sim(x, y):
    """Cosine similarity between two tensors.

    Args:
        x (tensor): First tensor.
        y (tensor): Second tensor.

    Returns:
        scalar: Cosine similarity.
    """
    norm_x = B.sqrt(B.sum(x ** 2))
    norm_y = B.sqrt(B.sum(y ** 2))
    return B.sum(x * y) / norm_x / norm_y
