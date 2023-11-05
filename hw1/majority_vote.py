import sys
import numpy as np
import csv

if __name__      == '__main__':
	if len(sys.argv) == 6:
		train_input    = sys.argv[1]
		test_input     = sys.argv[2]
		train_out      = sys.argv[3]
		test_out       = sys.argv[4]
		met_out        = sys.argv[5]
	else: 
		raise ValueError("Please pass in five command line arguments: "
				   "python majority_vote.py <train_input> <test_input> "
				   "<train_out> <test_out> <metrics_out>")
sda = 4
