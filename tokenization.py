from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.punkt import PunktParameters


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for sentence in text:
			tokenizedText.append(sentence.split())

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		punkt_params = PunktParameters()
		punkt_params.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
		tokenizer=TreebankWordTokenizer(punkt_params)

		for sentence in text:
			tokenizedText.append(tokenizer.tokenize(sentence))

		return tokenizedText