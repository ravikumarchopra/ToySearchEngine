from util import *

# Add your import statements here
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import re


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText= re.split('\.\s|\?\s|\!\s', text)

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		punkt_params = PunktParameters()
		punkt_params.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
		segmentedText= PunktSentenceTokenizer(punkt_params).tokenize(text)
		

		return segmentedText