import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
import string
import nltk
#nltk.download('punkt_tab')
documents=["I love programming in Python.Pyhton is great for data analysis.",
           "I enjoy learning machine learning techniques.",
           "Data Science is a mix of statistics,programming ,and machine learnning.",
           "Machine learning is part of the broader field of artificial intelligence.",
           "Statistics and dataScience are closely related fields."]
Stopwords=set(['is','a','the','for','and','of','in','to'])
def preprocess(text):
    tokens=word_tokenize(text.lower())
    tokens=[t for t in tokens if t not in Stopwords and t not in string.punctuation]
    return tokens
processed_docs=[preprocess(doc) for doc in documents]
dictionary=corpora.Dictionary(processed_docs)
corpus=[dictionary.doc2bow(doc) for doc in processed_docs]
lda_model=gensim.models.LdaMulticore(corpus,id2word=dictionary,passes=10)
topics=lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)
print(topics)
