import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

def ner_processing(file):
    document = nlp(file)
    return displacy.render(document, style='ent') 