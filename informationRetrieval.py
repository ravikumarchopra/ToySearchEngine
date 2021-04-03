from util import *

# Add your import statements here
from collections import Counter
import math
import numpy as np


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.IDF = None
        self.docVectors = None

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
                    if word not in ['.' , ',', '?', '!']:
                        if word in index:
                            if docID not in index[word]:
                                index[word].append(docID)
                        else:
                            index[word] = [docID]

        self.index = index

        terms = [*self.index]

        D = len(docs)
        tfs, IDF = [], []
        for doc in docs:
            tf = []
            for term in terms:
                tf.append(sum([sentence.count(term) for sentence in doc]))
            tfs.append(tf)

        for term in terms:
            idf = math.log(D/len(self.index[term]))
            IDF.append(idf)

        tfs = np.asarray(tfs)

        self.IDF = np.asarray(IDF)
        self.docVectors = np.multiply(tfs, self.IDF)

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

        terms = [*self.index]

        for query in queries:
            sim_docs = {}
            tf = []
            for term in terms:
                tf.append(sum([sentence.count(term) for sentence in query]))
            tfVector = np.asarray(tf)
            queryVector = np.multiply(tfVector, self.IDF)

            for docID, docVector in zip(range(1, len(self.docVectors)+1), self.docVectors):
                try:
                    dot = np.dot(queryVector, docVector)
                    if dot == 0:
                        sim_docs[docID] = 0.0
                    else:
                        normD = np.linalg.norm(docVector)
                        normQ = np.linalg.norm(queryVector)
                        cosine = dot/normD/normQ
                        sim_docs[docID] = cosine
                except:
                    pass

            doc_IDs_ordered_all.append(sorted(sim_docs, key=sim_docs.get, reverse=True))

        
        # for query in queries:
        # 	docList = []
        # 	for sentence in query:
        # 		for word in sentence:
        # 			if word in self.index:
        # 				docList.extend(self.index[word])

        # 	doc_IDs_ordered_all.append(
        # 	    [key for key, value in Counter(docList).most_common()])

        return doc_IDs_ordered_all
