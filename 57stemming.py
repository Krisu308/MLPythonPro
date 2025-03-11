import nltk 
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer 
# Download necessary NLTK data 
nltk.download('punkt') 
# Sample words 
words = ["running", "runner", "happily", "better", "fishing", "jumps"] 
# Initialize different stemmers 
porter_stemmer = PorterStemmer() 
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer("english") 
# Apply stemming using different stemmers 
print("Porter Stemmer:") 
for word in words: 
    print(f"{word} -> {porter_stemmer.stem(word)}") 
    print("\nLancaster Stemmer:") 
for word in words: 
    print(f"{word} -> {lancaster_stemmer.stem(word)}") 
    print("\nSnowball Stemmer:") 
for word in words: 
    print(f"{word} -> {snowball_stemmer.stem(word)}") 
