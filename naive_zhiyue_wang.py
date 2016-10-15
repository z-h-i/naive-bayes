import sys
import pandas as pd
from naive.NaiveBayes import NaiveBayes

'''
main(): 
	the program takes two arguments: the training and
	test datasets
	arguments must either:
		1) be file names within the same directory
		2) be full directory paths
'''
def main(argv):
	if len(argv) is not 2:
		print("Please enter two file names")
		sys.exit()
	train_file = argv[0]
	test_file = argv[1]
	try:
		train = pd.read_table(train_file)
		test = pd.read_table(test_file)
	except:
		print("Not valid file names or files aren't in the current directory or not '.dat' files")
		sys.exit()
	
	model = NaiveBayes(train)
	
	print('\n')
	model.print_probabilities()
	
	model.predict(train)
	accuracy = model.get_accuracy()
	print("\n")
	print("Accuracy on training set (" + str(len(train)) + 
		" instances):  " + str(round(accuracy * 100, 1)) + "%")
		
	model.predict(test)
	accuracy = model.get_accuracy()
	print("\n")
	print("Accuracy on test set (" + str(len(test)) + 
	" instances):  " + str(round(accuracy * 100, 1)) + "%")
	print('\n')

if __name__ == "__main__":
	main(sys.argv[1:])
