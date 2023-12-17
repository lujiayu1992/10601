import numpy as np
import argparse
import math
import feature


def sigmoid(x: np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(-x)
    return 1 / (1 + e)


def train(
    theda: np.ndarray,  # shape (D,) where D is feature dim
    X: np.ndarray,     # shape (N, D) where N is num of examples
    y: np.ndarray,     # shape (N,)
    num_epoch: int,
    learning_rate: float
) -> None:
    # TODO: Implement `train` using vectorization
    D = theda.size
    N = X.shape[0]
    for _ in range(num_epoch):
        for i in range(N):
            sig = sigmoid(X[i] @ theda)
            # print("sig", sig, "y", y[i],"predict", np.sign(X[i] @ theda))
            derivative = -(y[i]-sig)*X[i]
            theda -= learning_rate*derivative
    return theda


def predict(
    theda: np.ndarray,  # shape (D,)
    X: np.ndarray      # shape (N, D)
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    predict = np.sign(X @ theda) == 1
    return predict.astype(int)


def compute_error(
    y_pred: np.ndarray,
    y: np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    return np.sum(y_pred != y)/y.size


def read_file(file_name):
    dataset = np.loadtxt(file_name, delimiter='\t',
                         comments=None, dtype='f')
    Y = dataset[:, 0]
    X = dataset[:, 1:]
    return X, Y


class Model:
    def __init__(self, file_name):
        self.file_name = file_name

    def train(self, num_epoch, learning_rate):
        self.X, self.Y = read_file(self.file_name)
        N = self.X.shape[1]
        self.theda = np.zeros(N+1)
        self.X = np.insert(self.X, 0, 1, axis=1)
        self.theda = train(self.theda, self.X, self.Y,
                           num_epoch, learning_rate)

    def error(self, input_file):
        X, Y = read_file(input_file)
        X = np.insert(X, 0, 1, axis=1)
        return compute_error(predict(self.theda, X), Y)

    def metrics(self):
        return compute_error(predict(self.theda, self.X), self.Y)


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str,
                        help='path to formatted training data')
    # parser.add_argument("validation_input", type=str,
    #                     help='path to formatted validation data')
    parser.add_argument("test_input", type=str,
                        help='path to formatted test data')
    # parser.add_argument("train_out", type=str,
    #                     help='file to write train predictions to')
    # parser.add_argument("test_out", type=str,
    #                     help='file to write test predictions to')
    # parser.add_argument("metrics_out", type=str,
    #                     help='file to write metrics to')
    parser.add_argument("num_epoch", type=int,
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()
    model = Model(args.train_input)
    model.train(args.num_epoch, args.learning_rate)
    print(model.metrics())
    print(model.error(args.test_input))
