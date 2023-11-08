import sys
import numpy as np
import csv


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
        self.cls = None

    def fit(self, data):
        unq, cnt = np.unique(data, axis=0, return_counts=True)
        features = []
        xs = []
        ys = []
        for line in unq:
            features = line[:-1]
            same_fatures = np.where((unq[:,:-1] == features).all(axis=1))
            xs.append(features)
            if same_fatures[0].size > 1:
                # tie breaker
                print(same_fatures[0])
                
                break
            # else:
            #     ys.append (unq[-1])
                
        
if __name__ == "__main__":
    if len(sys.argv) == 2:  # 6
        train_input = sys.argv[1]
        # test_input     = sys.argv[2]
        # train_out      = sys.argv[3]
        # test_out       = sys.argv[4]
        # met_out        = sys.argv[5]
    else:
        raise ValueError(
            "Please pass in five command line arguments: "
            "python majority_vote.py <train_input> <test_input> "
            "<train_out> <test_out> <metrics_out>"
        )
train = Input(train_input)
model = MajorityVoteClassifier()
model.fit(train.read())

