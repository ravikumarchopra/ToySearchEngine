from util import *

# Add your import statements here
# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		# Fill in code here
        lemmatizer=WordNetLemmatizer()
		
        for sentence in text:
            lemmatized_sentences=[]
            for word in sentence:
                lemmatized_sentences.append(lemmatizer.lemmatize(word))
            reducedText.append(lemmatized_sentences)
            
		return reducedText


