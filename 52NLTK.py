import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
text="Natural Language Processing with Python is amazing!"
tokens=word_tokenize(text)
print(tokens)
