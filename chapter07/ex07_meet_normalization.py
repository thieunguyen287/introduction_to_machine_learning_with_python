import spacy
import nltk


en_nlp = spacy.load('en_core_web_sm')
stemmer = nltk.stem.PorterStemmer()


def compare_normalization(doc):
    doc_spacy = en_nlp(doc)
    print "Lemmatization:", [token.lemma_ for token in doc_spacy]
    print "Stemming:", [stemmer.stem(token.norm_.lower()) for token in doc_spacy]
    for token in doc_spacy:
        print token.text, token.pos_, token.dep_, token.lemma_

compare_normalization(u"Our meeting today was worse than yesterday,"
                      u"I'm scared of meeting the clients tomorrow.")
