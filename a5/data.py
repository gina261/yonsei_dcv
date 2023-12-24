import os
import gzip
import numpy as np


class DataSampler:
    """
    A helper class to iterate through data and labels in minibatches.

    Example usage:

    X = torch.randn(N, D)
    y = torch.randn(N)
    sampler = DataSampler(X, y, batch_size=64)
    for X_batch, y_batch in sampler:
        print(X_batch.shape)  # (64, D)
        print(y_batch.shape)  # (64,)

    The loop will run for exactly one epoch over X and y -- that is, each entry
    will appear in exactly one minibatch. If the batch size does not evenly
    divide the number of elements in X and y then the last batch will be have
    fewer than batch_size elements.

    You can use a DataSampler object to iterate through the data as many times
    as you want. Each epoch will iterate through the data in a random order.
    """
    def __init__(self, X, y, batch_size):
        """
        Create a new DataSampler.

        Inputs:
        - X: Numpy array of shape (N, D)
        - y: Numpy array of shape (N,)
        - batch_size: Integer giving the number of elements for each minibatch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        """
        Iterate through the data. This returns a generator which yields tuples:
        - X_batch: Numpy array of shape (batch_size, D)
        - y_batch: Numpy array of shape (batch_size,)

        Note that when the batch size does not divide the number of elements N,
        then the last minibatch will have fewer than batch_size elements.
        """
        N = self.X.shape[0]
        perm = np.random.permutation(N)
        start, stop = 0, self.batch_size
        while start < N:
            idx = perm[start:stop]
            X_batch = self.X[idx]
            y_batch = self.y[idx]
            start += self.batch_size
            stop += self.batch_size
            yield X_batch, y_batch

    def __len__(self):
        """ Get the number of minibatches in an epoch. """
        return self.X.shape[0] // self.batch_size

_DATA_DIR = 'data'


def load_fashion_mnist(data_dir=_DATA_DIR, num_train=50000, num_val=10000,
                       num_test=10000, seed=0):
    """
    Load and preprocess Fashion MNIST data. More specifically:

    (1) Load the raw data from disk
    (2) Shuffle the train and test sets
    (3) Subsample the training set to create train and val sets
    (4) Subsample the test set
    (5) Preprocess all images to be float32 in the range [0, 1]

    Inputs:
    - data_dir: Path to the data directory
    - num_train: Size of training set after subsampling
    - num_val: Size of validation set after subsampling
    - num_test: Size of test set after subsampling
    - seed: Random seed to use when shuffling data

    Returns a dictionary with keys and values:
    - X_train: float32 array of shape (num_train, 1, 28, 28)
    - X_val: float32 array of shape (num_val, 1, 28, 28)
    - X_test: float32 array of shape (num_test, 1, 28, 28)
    - y_train: int64 array of shape (num_train,) in the range [0, 10)
    - y_val: int64 array of shape (num_val,) in the range [0, 10)
    - y_test: int64 array of shape (num_test,) in the range [0, 10)
    """

    # Load training & test data
    filenames = [
        ['X_train', 'train-images-idx3-ubyte.gz'],
        ['y_train', 'train-labels-idx1-ubyte.gz'],
        ['X_test', 't10k-images-idx3-ubyte.gz'],
        ['y_test', 't10k-labels-idx1-ubyte.gz']
    ]
    data = {}
    
    # Preprocess (training/test) images: Convert to float in the range [0, 1]
    for name in filenames[::2]:
        with gzip.open(os.path.join(data_dir, name[1]), 'rb') as f:
            data[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16) \
                            .reshape(-1,1*28*28).astype(np.float32) / 255.
    # Training/test labels
    for name in filenames[1::2]:
        with gzip.open(os.path.join(data_dir, name[1]), 'rb') as f:
            data[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    
    X_train = data['X_train']; y_train = data['y_train']; X_test = data['X_test']; y_test = data['y_test']
    
    # Shuffle the training and test sets
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]
    idx = rng.permutation(X_test.shape[0])
    X_test = X_test[idx]
    y_test = y_test[idx]
    
    X_train_orig = X_train
    y_train_orig = y_train
    X_train = X_train_orig[:num_train]
    y_train = y_train_orig[:num_train]
    X_val = X_train_orig[num_train:(num_train + num_val)]
    y_val = y_train_orig[num_train:(num_train + num_val)]
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }
    return data


if __name__ == '__main__':
    data = load_fashion_mnist()
    for k, v in data.items():
        print(k, v.shape, v.dtype)
