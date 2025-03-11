import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
# Download necessary NLTK data 
nltk.download('punkt') 
nltk.download('wordnet') 
nltk.download('omw-1.4') 
# Sample text 
text = "The Lt. M. J. Kundaliya Arts & Commerce Mahila College Running the Computer Science Department." 
print("original text is :",text)
print()
# Tokenize the text 
tokens = word_tokenize(text)
print("Tokens :",tokens)
print()
# Initialize the Lemmatizer 
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens 
lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

# Print the lemmatized tokens 
print("Lemmatization :",lemmatized_tokens)
