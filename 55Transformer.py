from transformers import pipeline
summarizer=pipeline("summarization")
text="Natural Language Processing(NLP) is a fascinating field.."
print(summarizer(text,max_length=17,min_length=10))
