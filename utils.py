import numpy as np


def shuffle_batch(X, y, batch_size):
    """
    Create batches.
    :param X: Features
    :param y: Labels
    :param batch_size: Batch size
    :return: features and labels
    """
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, np.reshape(y_batch, (y_batch.shape[0],))
