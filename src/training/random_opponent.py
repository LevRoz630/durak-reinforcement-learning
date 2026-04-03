import numpy as np


def random_action(mask: np.ndarray) -> int:
    """Return a uniformly random legal action index."""
    legal = np.where(mask)[0]
    return int(np.random.choice(legal))
