from typing import Sequence
from numpy.lib.function_base import append
from numpy.ma.core import dot
from util import *

# Add your import statements here
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import math
from scipy.linalg import svd
import numpy as np


class LSA():

    def __init__(self):
        self.index = None
        self.IDF = None
        self.term_document_matrix = None
        self.term_document_matrix_reduced = None
        self.term_space = None
        self.doc_space = None
        self.tf_idf_vectorizer = None
        print('Initializing LSA based IR system :')

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

        doc_list = []
        print('Creating tf-IDF vectors for documents :')
        # Building doc index
        for doc, docID in tqdm(zip(docs, docIDs), total=len(docIDs), unit=' Documents', desc='Documents Processed : '):
            doc_words = []
            for sentence in doc:
                doc_words.extend(sentence)
            doc_list.append(' '.join(doc_words))

        vectorizer = TfidfVectorizer()
        self.term_document_matrix = vectorizer.fit_transform(doc_list).T
        self.tf_idf_vectorizer = vectorizer

        print('Performing SVD ...')
        # Performing SVD
        k = 901  # np.linalg.matrix_rank(self.term_document_matrix)
        U, S, V = np.linalg.svd(self.term_document_matrix.toarray(), full_matrices=True)
        U_k, S_k, V_k = U[:, :k], np.diag(S[:k]), V[:k]
        self.term_document_matrix_reduced = np.dot(U_k, np.dot(S_k, V_k))
        self.term_space = np.dot(U_k, S_k)
        self.doc_space = np.dot(S_k, V_k)
        print('[ SVD Performed. ]')

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
        queryVectors = []
        terms=self.tf_idf_vectorizer.get_feature_names()
        print('Creating query vectors :')
        # Creating query vectors
        for query in tqdm(queries, total=len(queries), unit=' Queries', desc='Queries Processed : '):
            query_words = []
            for sentence in query:
                for word in sentence:
                    if word in terms:
                        query_words.append(word)
            term_index_vector=[self.tf_idf_vectorizer.vocabulary_[x] for x in query_words]
            queryVector = np.mean(self.term_space[term_index_vector], axis=0)
            queryVectors.append(queryVector)

        docVectors = self.doc_space.T
        print('Finding Similar documents for queries : ')
        # Finding Similar documents for queries
        for queryVector in tqdm(queryVectors, total=len(queryVectors), unit=' Queries', desc='Queries Processed : '):
            sim_docs = {}
            for docID, docVector in zip(range(1, len(docVectors)+1), docVectors):
                # docVector=docVectors[:,docID-1]
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

            doc_IDs_ordered_all.append(
                sorted(sim_docs, key=sim_docs.get, reverse=True))

        return doc_IDs_ordered_all
