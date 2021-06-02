from util import *
# Add your import statements here
import numpy as np
from tqdm import tqdm

class ESA():

    def __init__(self):
        self.index = None
        self.IDF = None
        self.docVectors = None
        self.article_terms = None
        self.articleVectors = None
        self.doc_art_matrix = None
        print('Initializing ESA based IR system ...')

    def buildIndex(self, docs, docIDs, articles, articleIDs):
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
        arg3 : list
                A list of lists of lists where each sub-list is
                an article and each sub-sub-list is a sentence of the article
        arg4 : list
                A list of integers denoting IDs of the articles
        Returns
        -------
        None
        """

        index = {}
        article_index = {}

        print('Building doc index :')
        # building inverted index for documents
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

        print('Calculating tf value for documents :')
        # calculating tf values for each document
        for doc in tqdm(docs, total=D, unit=' Documents', desc='Documents Processed : '):
            tf = []
            for term in terms:
                tf.append(sum([sentence.count(term) for sentence in doc]))
            tfs.append(tf)

        print('Calculating IDF value for terms :')
        # calculating IDF values for each term
        for term in tqdm(terms, total=len(terms), unit=' Terms', desc='Terms Processed : '):
            idf = np.log(D/len(self.index[term]))
            IDF.append(idf)

        tfs = np.asarray(tfs)
        self.IDF = np.asarray(IDF)
        # calculating tf-IDF matrix for documents
        self.docVectors = np.multiply(tfs, self.IDF)
        print('[ Document vectors created. ]')

        print('Building article index :')
        # building inverted index for articles
        for article, articleID in tqdm(zip(articles, articleIDs), total=len(articleIDs), unit=' Articles', desc='Articles Processed : '):
            for sentence in article:
                for word in sentence:
                    if word in article_index:
                        if articleID not in article_index[word]:
                            article_index[word].append(articleID)
                    else:
                        article_index[word] = [docID]

        self.article_terms = [*article_index]

        a_tfs, a_IDF = [], []
        print('Calculating tf value for articles :')
        # calculating tf values for each article
        for article in tqdm(articles, total=len(articles), unit=' Articles', desc='Articles Processed : '):
            a_tf = []
            for term in self.article_terms:
                a_tf.append(sum([sentence.count(term) for sentence in article]))
            a_tfs.append(a_tf)

        print('Calculating IDF value for articles :')
        # calculating IDF values for each articles
        for term in tqdm(self.article_terms, total=len(self.article_terms), unit=' Terms', desc='Terms Processed : '):
            a_idf = np.log(len(articles)/len(article_index[term]))
            a_IDF.append(a_idf)

        a_tfs = np.asarray(a_tfs)
        # calculating tf-IDF matrix for articles
        self.articleVectors = np.multiply(a_tfs, a_IDF)
        print('[ Article vectors created. ]')

        print('Creating Article-Document Matrix :')
        # Computing article document matrix
        doc_art_matrix = []
        for docID in tqdm(docIDs, total=D, unit=' Documents', desc='Documents Processed : '):
            doc_art_vec = []
            for i, term in enumerate(terms):
                try:
                    pos = self.article_terms.index(term)
                    wt_vec = self.articleVectors[:, pos]
                except:
                    wt_vec = np.zeros(self.articleVectors.shape[0]).T

                doc_art_vec.append(self.docVectors[docID-1][i]*wt_vec)
            doc_art_matrix.append(sum(doc_art_vec))
        self.doc_art_matrix = doc_art_matrix
        print('[ Article-Document Matrix created. ]')



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
        queryVectors = []

        print('Creating query vectors :')
        # Creating query vectors
        for query in tqdm(queries, total=len(queries), unit=' Queries', desc='Queries Processed : '):
            sim_docs = {}
            tf = []
            for term in terms:
                tf.append(sum([sentence.count(term) for sentence in query]))
            tfVector = np.asarray(tf)
            queryVector = np.multiply(tfVector, self.IDF)
            queryVectors.append(queryVector)
        
        print('Creating Query-Article Matrix :')
        # Creating Query-Article Matrix
        num_queries = len(queries)
        query_art_matrix = []
        for qID in tqdm(range(num_queries), unit=' Queries', desc='Queries Processed : '):
            query_art_vec = []
            for i, term in enumerate(terms):
                try:
                    pos = self.article_terms.index(term)
                    wt_vec = self.articleVectors[:, pos]
                except:
                    wt_vec = np.zeros(self.articleVectors.shape[0]).T

                query_art_vec.append(queryVectors[qID][i]*wt_vec)

            query_art_matrix.append(sum(query_art_vec))
        query_art_matrix = np.asarray(query_art_matrix)

        print('Finding relevent documents for queries :')
        # Finding relevent documents for queries
        for queryVector in tqdm(query_art_matrix, total=len(query_art_matrix), unit=' Queries', desc=' Queries Processed : '):
            for docID, docVector in zip(range(1, len(self.doc_art_matrix)+1), self.doc_art_matrix):
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
