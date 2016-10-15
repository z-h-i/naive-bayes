import pandas as pd
from functools import reduce
from itertools import chain as ch

class NaiveBayes:
	
	def __init__(self, train_data):
		self.train_data = train_data
		self.test_data = None
		self.probabilities = self.set_probabilities()
		self.predictions = None
	
	# grabs probabilities necessary for naive Bayes' computation
	def set_probabilities(self):
		dataframe = self.train_data
		__, cols = dataframe.shape
		
		counts = ([[[sum(dataframe[dataframe.iloc[:, cols - 1] == class_val][attribute] == attr_val) 				# grab counts for each attribute's value
			 for attr_val in [0, 1]] for attribute in dataframe.columns[0:cols - 1]] for class_val in [0, 1]])	# given each class, then
		
		probabilities = ([[[counts[class_val][attr][attr_val] / sum(counts[class_val][attr]) 		 		   # compute the likelihoods of each
			for attr_val in [0, 1]] for attr in range(cols - 1)] for class_val in [0, 1]])		 		   # attribute with respect to class
		return probabilities
	
	# classifies a single instance via Bayes': (likelihoods) * (class probability)
	def classify(self, instance):
		likelihoods = ([reduce(lambda x, y: x * y, map(lambda prob, val: prob[val], attr_probs, instance)) 					# calculate product of all likelihoods
				  for attr_probs in self.probabilities])
		class_1_prob = float(sum(self.train_data.iloc[:, len(self.train_data.columns) - 1])) / len(self.train_data)			# calculate class value == 1 probability
		class_est_probs = [likelihood * class_prob for likelihood, class_prob in zip(likelihoods, [1 - class_1_prob, class_1_prob])]		# compute class probability estimates
		likely_class = [prediction for prediction, probability in enumerate(class_est_probs) if probability is max(class_est_probs)]
		return likely_class
	
	def predict(self, test_data):
		dataframe = test_data
		rows, cols = dataframe.shape
		predictions = [self.classify(dataframe.iloc[row, 0:(cols - 1)]) for row in range(rows)]
		predictions = list(ch.from_iterable(predictions))
		self.test_data = test_data
		self.predictions = predictions
	
	def get_accuracy(self):
		test_data = self.test_data
		accuracy = sum(map(lambda x, y: x == y, self.predictions, test_data.iloc[:, len(test_data.columns) - 1].tolist()))
		accuracy = float(accuracy) / len(self.predictions)
		return accuracy
	
	def print_probabilities(self, probabilities = None, attr_names = None, _class = None):
		if probabilities is None:
			probabilities = self.probabilities
			attr_names = self.train_data.columns
		class_1_prob = float(sum(self.train_data.iloc[:, len(self.train_data.columns) - 1])) / len(self.train_data)			# calculate class value == 1 probability
	
		if len(probabilities) is 2:
			print('P(' + attr_names[len(attr_names) - 1] + '=' + str(0) + ')=' + str(format((1 - class_1_prob), '.2f')) + ': ', end = '')
			self.print_probabilities(probabilities[0], attr_names, '0')
			print('\n')
			print('P(' + attr_names[len(attr_names) - 1] + '=' + str(1) + ')=' + str(format(class_1_prob, '.2f')) + ': ', end = '')
			self.print_probabilities(probabilities[1], attr_names, '1')
			print('\n')
		else:
			for i in range(len(probabilities)):
				print('P(' + attr_names[i] + '=' + str(0) + '|' + _class + ')=' + str(format(probabilities[i][0], '.2f')) + ' ', end = '')
				print('P(' + attr_names[i] + '=' + str(1) + '|' + _class + ')=' + str(format(probabilities[i][1], '.2f')) + ' ', end = '')
		
		
		
