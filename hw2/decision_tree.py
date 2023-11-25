import math
import numpy as np
import sys


class Input:
    def __init__(self, file_path):
        self.data = None
        self.file_path = file_path
        self.labels = None

    def read(self) -> np.ndarray:
        lines = open(self.file_path, "r").read().splitlines()
        buffer = []
        for line in lines[1:]:
            features = [int(s) for s in line.split("\t")]
            buffer.append(features)
        self.data = np.array(buffer)
        self.labels = np.array([s for s in lines[0].split("\t")])
        return self.data


class DecisionTree:
    def __init__(self, data, labels, level, max_depth):
        self.data = data
        self.labels = labels
        self.size = self.data.shape[0]
        self.level = level
        self.max_depth = max_depth
        self.left_node = None
        self.right_node = None
        self.split_index = None
        self.cl = None

    def entropy(self):
        ys = self.data[:, -1]
        positive_ratio = np.sum(ys == 1) / self.size
        negative_ratio = np.sum(ys == 0) / self.size
        if positive_ratio * negative_ratio == 0:
            return 0
        return -positive_ratio * math.log2(positive_ratio) - negative_ratio * math.log2(
            negative_ratio
        )

    def mutual_info(self, i):
        if i >= self.data.shape[1]-1 or self.data.shape[1] == 1:
            return 0, None, None
        entropy = self.entropy()

        remaining_index = np.delete(np.arange(self.data.shape[1]), i, 0)
        left_node = DecisionTree(
            self.data[self.data[:, i] == 0][:, remaining_index],
            self.labels[remaining_index],
            self.level + 1,
            self.max_depth,
        )
        right_node = DecisionTree(
            self.data[self.data[:, i] == 1][:, remaining_index],
            self.labels[remaining_index],
            self.level + 1,
            self.max_depth,
        )
        info = (
            entropy
            - left_node.size / self.size * left_node.entropy()
            - right_node.size / self.size * right_node.entropy()
        )
        return info, left_node, right_node

    def fit(self):
        ys = self.data[:, -1]
        if self.level >= self.max_depth:
            self.cl = int(np.sum(ys == 1) >= np.sum(ys == 0))
            return
        if (ys==0).all():
            self.cl = ys[0]
            return
        max_info = 0
        for i in range(self.data.shape[1]):
            current_info, left_node, right_node = self.mutual_info(i)
            if current_info > max_info:
                max_info = current_info
                self.left_node = left_node
                self.right_node = right_node
                self.split_index = i
        if self.left_node:
            self.left_node.fit()
        if self.right_node:
            self.right_node.fit()

    def predict_row(self, row):
        if self.cl is not None:
            return self.cl
        ys = self.data[:, -1]
        split = row[self.split_index]
        row = np.delete(row, self.split_index, 0)
        if split:
            # right
            return self.right_node.predict_row(row)
        else:
            # left
            return self.left_node.predict_row(row)

    def predict_data(self, data):
        ys = np.apply_along_axis(self.predict_row, 1, data)
        return ys

    def error(self, data):
        return np.sum(data[:, -1] != self.predict_data(data))/data.shape[0]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_train = sys.argv[1]
        input_test = sys.argv[2]
    else:
        raise ValueError(
            "Please pass in five command line arguments: "
            "python inspection.py <train_input> "
            f"<train_out>; you have {len(sys.argv)} args"
        )
    train = Input(input_train)
    test = Input(input_test)
    train.read()
    test.read()
    model = DecisionTree(train.data, train.labels, 0, 2)
    model.fit()
    print(model.error(train.data))
    print(model.error(test.data))
