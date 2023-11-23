import math
import numpy as np
import sys

class Input:
    def __init__(self, file_path):
        self.data = None
        self.file_path = file_path

    def read(self) ->np.ndarray:
        lines = open(self.file_path, 'r').read().splitlines()[1:]
        buffer = []
        for line in lines:
            features = [int(s) for s in line.split("\t")]
            buffer.append(features)
        self.data = np.array(buffer)
        return self.data
        

class MajorityVoteClassifier:
    def __init__(self):
        self.cl = None

    def fit(self, data):
        ys = data[:,-1]
        self.cl = int (np.sum(ys == 1) >= np.sum(ys == 0))
        return self.cl

    def error(self, data):
        ys = data[:,-1]
        return np.sum(ys != self.cl)/ys.size

    def entropy(self, data):
        ys = data[:,-1]
        positive_ratio = np.sum(ys == 1)/ ys.size
        negative_ratio = np.sum(ys == 0)/ ys.size
        return - positive_ratio*math.log2(positive_ratio) - negative_ratio*math.log2(negative_ratio)

def write_result(file_path, model, data):
    with open(file_path, "w+") as f:
        f.write(f"entropy: {model.entropy(data)}\n"
                f"error: {model.error(data)}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input  = sys.argv[1]
        output = sys.argv[2]
    else:
        raise ValueError(
            "Please pass in five command line arguments: "
            "python inspection.py <train_input> "
            f"<train_out>; you have {len(sys.argv)} args"
        )
    train = Input(input)
    model = MajorityVoteClassifier()
    data = train.read()
    model.fit(data)
    write_result(output, model, data)
