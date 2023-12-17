import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


class Encoder:
    def __init__(self, file_name):
        self.glove_map = load_feature_dictionary(file_name)

    def encode(self, file_name):
        dataset = load_tsv_dataset(file_name)
        label = np.array([row[0] for row in dataset])
        reviews = [row[1] for row in dataset]
        review_features = np.array([self.feature(review)
                                   for review in reviews])
        return np.concatenate((label.reshape(label.size, 1), review_features), axis=1)

    def feature(self, review):
        words = np.array(review.split(), dtype=object)
        embed = np.array([self.glove_map[word]
                         for word in words if word in self.glove_map])
        return np.average(embed, axis=0)

    def format(self, input_file, output_file):
        features = self.encode(input_file)
        np.savetxt(output_file, features, delimiter='\t', fmt='%.6f')


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str,
                        help='path to training input .tsv file')
    # parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    # parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str,
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str,
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    # parser.add_argument("validation_out", type=str,
    #                     help='path to output .tsv file to which the feature extractions on the validation data should be written')
    # parser.add_argument("test_out", type=str,
    #                     help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    encoder = Encoder(args.feature_dictionary_in)
    encoder.format(args.train_input, args.train_out)