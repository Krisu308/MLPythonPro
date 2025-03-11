import string 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
# Download necessary NLTK data 
nltk.download('punkt') 
nltk.download('stopwords') 
# Sample text 
text = "I love programming in Python, especially for data science!"
print(text)
# Tokenization 
tokens = word_tokenize(text)
print("Tokenization:",tokens)
print()
# Lowercasing 
tokens = [token.lower() for token in tokens]
print("Lower case:",tokens)
print()
# Removing punctuation 
tokens = [token for token in tokens if token not in string.punctuation]
print("Punctuation:",tokens)
print()
# Removing stopwords 
stop_words = set(stopwords.words('english')) 
tokens = [token for token in tokens if token not in stop_words]
print("Stop Words:",tokens)
print()
# Stemming 
stemmer = PorterStemmer() 
tokens = [stemmer.stem(token) for token in tokens] 
# Resulting pre-processed text 
print("Porter Stemming:",tokens)
print()
