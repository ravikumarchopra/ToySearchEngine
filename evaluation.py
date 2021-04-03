from util import *

# Add your import statements here
import math


class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The precision value as a number between 0 and 1
        """

        precision = -1

        retrieved = query_doc_IDs_ordered[:k]
        relevant = true_doc_IDs

        rel_and_ret = list(set(relevant) & set(retrieved))
        precision = len(rel_and_ret)/len(retrieved)

        # if precision==0:
        #     print('query_id: ', query_id ,', Retrieved Docs: ', retrieved,', Relevant Docs: ', relevant)

        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgement - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean precision value as a number between 0 and 1
        """

        meanPrecision = -1

        precisions = []
        for ret_doc_IDs, query_id in zip(doc_IDs_ordered, query_ids):
            relevant = [int(q['id']) for q in qrels if int(q['query_num']) == query_id]
            precisions.append(self.queryPrecision(
                ret_doc_IDs, query_id, relevant, k))

        meanPrecision = sum(precisions)/len(precisions)

        return meanPrecision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The recall value as a number between 0 and 1
        """

        recall = -1

        try:
            retrieved = query_doc_IDs_ordered[:k]
            relevant = true_doc_IDs

            rel_and_ret = list(set(relevant) & set(retrieved))
            recall = len(rel_and_ret)/len(relevant)
        except:
            pass

        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean recall value as a number between 0 and 1
        """

        meanRecall = -1

        recalls = []
        for ret_doc_IDs, query_id in zip(doc_IDs_ordered, query_ids):
            relevant = [int(q['id']) for q in qrels if int(q['query_num']) == query_id]
            recalls.append(self.queryRecall(
                ret_doc_IDs, query_id, relevant, k))

        meanRecall = sum(recalls)/len(recalls)

        return meanRecall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The fscore value as a number between 0 and 1
        """

        fscore = -1

        retrieved = query_doc_IDs_ordered
        relevant = true_doc_IDs

        try:
            P = self.queryPrecision(retrieved, query_id, relevant, k)
            R = self.queryRecall(retrieved, query_id, relevant, k)

            fscore = 2*P*R/(P+R)
        except:
            pass


        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1

        Fscores = []
        for ret_doc_IDs, query_id in zip(doc_IDs_ordered, query_ids):
            relevant = [int(q['id']) for q in qrels if int(q['query_num']) == query_id]
            Fscores.append(self.queryFscore(
                ret_doc_IDs, query_id, relevant, k))

        meanFscore = sum(Fscores)/len(Fscores)

        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, rel_score, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The nDCG value as a number between 0 and 1
        """

        nDCG = -1

        retrieved = query_doc_IDs_ordered[:k]
        relevant = true_doc_IDs
        rel, ideal_ordering = [], []
        DCG, IDCG = 0, 0
        try:
            for i in range(k):
                if retrieved[i] in relevant:
                    rel.append(rel_score[relevant.index(retrieved[i])])
                else:
                    rel.append(0)
                DCG += rel[i]/math.log(i+2, 2)

            ideal_ordering = rel.copy()
            ideal_ordering.sort(reverse=True)
            for i in range(k):
                IDCG += (ideal_ordering[i]/math.log(i+2, 2))

            nDCG = DCG/IDCG
        except:
            pass

        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = -1
        NDCGs = []
        for ret_doc_IDs, query_id in zip(doc_IDs_ordered, query_ids):
            relevant, rel_score = [], []
            for q in qrels:
                if int(q['query_num']) == query_id:
                    rel_score.append(int(5-q['position']))
                    relevant.append(int(q['id']))
            NDCGs.append(self.queryNDCG(
                ret_doc_IDs, query_id, relevant, rel_score, k))

        meanNDCG=sum(NDCGs)/len(NDCGs)

        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The average precision value as a number between 0 and 1
        """

        avgPrecision=-1

        retrieved=query_doc_IDs_ordered[:k]
        relevant=true_doc_IDs
        precisions=[]

        try:
            rel=0
            for i in range(k):
                if retrieved[i] in relevant:
                    rel += 1
                    precisions.append(rel/(i+1))

            avgPrecision=sum(precisions)/len(precisions)
        except:
            pass

        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The MAP value as a number between 0 and 1
        """

        meanAveragePrecision=-1

        averagePrecisions=[]
        for ret_doc_IDs, query_id in zip(doc_IDs_ordered, query_ids):
            relevant=[int(q['id']) for q in q_rels if int(q['query_num']) == query_id]
            averagePrecisions.append(self.queryAveragePrecision(
                ret_doc_IDs, query_id, relevant, k))

        meanAveragePrecision=sum(averagePrecisions)/len(averagePrecisions)

        return meanAveragePrecision
