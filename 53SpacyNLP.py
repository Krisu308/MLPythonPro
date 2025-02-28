import spacy
nlp=spacy.load("en_core_web_sm")
doc=nlp("John is learnning Naturasl Language Processing in New York.")
for entity in doc.ents:
    print(entity.text,entity.label_)
