from util import *

# Add your import statements here
from nltk.corpus import stopwords
import re


class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence with stopwords removed
        """

        stopwordRemovedText = []

        regex = re.compile(r'[a-z]{3,45}')
        stop_words = stopwords.words("english")

        for sentence in text:
            pSentence = self.processSentence(sentence)
            words = []
            for word in pSentence:
                word = word.strip()
                if word not in stop_words and regex.match(word):
                    words.append(word)
            stopwordRemovedText.append(words)

        return stopwordRemovedText

    def processSentence(self, sentence):
        sentence = ' '.join(sentence)
        sentence = sentence.lower().encode('ascii', 'ignore').decode()
        sentence = sentence.replace("\\", "")
        sentence = sentence.replace("/", " ")
        sentence = sentence.replace("-", " ")
        sentence = sentence.replace(".", " ")
        sentence = sentence.replace("'", " ")
        return sentence.strip().split(' ')
