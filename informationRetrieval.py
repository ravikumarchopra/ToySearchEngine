from util import *

# Add your import statements here
from collections import Counter



class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = {}
		for doc, docID in zip(docs, docIDs):
			for sentence in doc:
				for word in sentence:
					if word in index:
						if docID not in index[word]:
							index[word].append(docID)
					else:
						index[word]=[docID]

		self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered_all = []

		for query in queries:
			docList= []
			for sentence in query:
				for word in sentence:
					if word in self.index:
						docList.extend(self.index[word])

			doc_IDs_ordered_all.append([key for key, value in Counter(docList).most_common()])
						

		return doc_IDs_ordered_all




