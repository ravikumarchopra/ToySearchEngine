# Add your import statements here
import wikipedia
import json
import numpy as np


def cosine_sim(queryVector, docVector):
    cosine = 0
    try:
        dot = np.dot(queryVector, docVector)
        if dot == 0:
            cosine = 0.0
        else:
            normD = np.linalg.norm(docVector)
            normQ = np.linalg.norm(queryVector)
            cosine = dot/normD/normQ
    except:
        pass
    return cosine

# Add any utility functions here
def fetchWikiArticles(self, titles):
    wikipedia.set_lang("en")
    articles = {}
    j = 0
    unique_words = set()
    for title in zip(titles):
        for t in title:
            for sentence in t:
                result = []
                for word in sentence:
                    unique_words.add(word)

    wikipedia.set_lang("en")
    articles = []
    j = 0
    for word in titles:
        result = wikipedia.search(word, results=4)
        for x in result:
            try:
                articles.append({
                    'id': j,
                    'title': x,
                    'body': wikipedia.summary(x)
                })
                j += 1
            except:
                pass

    with open('articles.json', 'w') as outfile:
        json.dump(articles, outfile)
