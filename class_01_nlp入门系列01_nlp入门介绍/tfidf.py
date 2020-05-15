from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 语料库
corpus = [
    "John likes to watch movies, Mary likes movies too",
    "John also likes to watch football games",
]

# bag of words
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(bow.toarray())

# tf-idf
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(bow.toarray())
print(tfidf.toarray())

