import numpy as np
import collections
import pandas as pd
import random
import nltk
import re
import json
import kmeans

import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(["'", 'b', 'e', 'f', 'g', 'h', 'j', 'l', 'n',
                  'p', 'r', 'u', 'v', 'w', 'how', 'i', 'use', 'get', 'work', 'create', 'k'])


def tokenization(text):
    next_text = re.sub(r"[^\\a-z+#\\n]+", " ",
                       re.sub("<[^>]*>", "", text.lower()))
    tokens = [word for sent in nltk.sent_tokenize(
        next_text) for word in nltk.word_tokenize(sent)]

    stems = [stemmer.stem(t) for t in tokens]
    filtered_tokens = []

    for token in stems:
        if not token in stopwords:
            filtered_tokens.append(token)

    return filtered_tokens


def data_prepare(path):
    df = pd.read_json(path)[["title", "body"]].dropna()

    df["title"] = df["title"].apply(tokenization)
    df["body"] = df["body"].apply(tokenization)
    print(df.head())
    df.to_json('datasets/pretty_stackoverflow.json', orient='split')


def get_data_from_file(file):
    with open(file) as json_file:
        data = json.load(json_file)
        category = data['columns']
        dataset = data['data']
    return dataset


def dummy_fun(doc):
    return doc


def clusters_to_markdown_table(file, clusters):
    lenc = len(clusters[0])
    with open(file, 'w+') as f:
        f.write('cluster_id ' + '| * ' * (lenc-1) + '\n')
        for i, e in enumerate(clusters):
            f.write(str(i) + ' | ')
            f.write(' | '.join([t[0] for t in e]))
            f.write('\n')


if __name__ == '__main__':
    # dataset preprocessing
    # data_prepare('datasets/stackoverflow.json')
    data = get_data_from_file('datasets/pretty_stackoverflow.json')
    titles = [x[0] for x in data]

    tfidf = TfidfVectorizer(
        max_df=0.8,
        max_features=10000,
        min_df=0.01,
        stop_words=stopwords,
        use_idf=True,
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)

    tfidf_matrix = tfidf.fit_transform(titles)

    num = 20
    indexing_matrix = [(i, e) for i, e in enumerate(tfidf_matrix.toarray())]
    m = kmeans.kmeans(num, titles)
    m.fit(indexing_matrix)
    from pprint import pprint
    clusters = m.get_clusters()

    update_cluster = []
    for c in clusters:
        words = []
        for block in c:
            words.extend(block)
        cnt = collections.Counter(words)
        update_cluster.append(cnt.most_common(16))

    my_file = 'my_cluster.md'
    clusters_to_markdown_table(my_file, update_cluster)

