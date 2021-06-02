from util import *

# Add your import statements here
from tqdm import tqdm
import math
import numpy as np


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.IDF = None
        self.docVectors = None
        print('Initializing Vector Space based IR system ...')

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
        print('Building doc index :')
        # Building doc index
        for doc, docID in tqdm(zip(docs, docIDs), total=len(docIDs), unit=' Documents', desc='Documents Processed : '):
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
        print('Calculating tf values for documents :')
        # Calculating tf values for documents
        for doc in tqdm(docs, total=D, unit=' Documents', desc='Documents Processed : '):
            tf = []
            for term in terms:
                tf.append(sum([sentence.count(term) for sentence in doc]))
            tfs.append(tf)

        print('Calculating IDF values for terms :')
        # Calculating IDF values for terms
        for term in tqdm(terms, total=len(terms), unit=' Terms', desc='Terms Processed : '):
            idf = math.log(D/len(self.index[term]))
            IDF.append(idf)

        tfs = np.asarray(tfs)

        self.IDF = np.asarray(IDF)
        self.docVectors = np.multiply(tfs, self.IDF)
        print('[ Document vectors created. ]')

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

        print('Creating query vectors :')
        # Creating query vectors
        for query in tqdm(queries, total=len(queries), unit=' Queries', desc='Queries Processed : '):
            sim_docs = {}
            tf = []
            for term in terms:
                tf.append(sum([sentence.count(term) for sentence in query]))
            tfVector = np.asarray(tf)
            queryVector = np.multiply(tfVector, self.IDF)
            
            # Finding Similar documents for queries
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

        return doc_IDs_ordered_all
