from textblob import TextBlob
text="I love programming."
blob=TextBlob(text)
print(blob.sentiment)
