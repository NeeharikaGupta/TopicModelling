doc1 = "The link between smoking and cancer is well known."
doc2 = "As well as smoking is related to other linked diseases like emphysema and bronchitis."
doc3 = "Smokers have a greater risk of heart diseases"
doc4 = "This can be seen as an evidence in the recent court cases in the USA where smokers have been awarded damages from the tobacco companies"
doc5 = "Further, there is substantial research that even passive smoking can have long term effects on health"

#making corpus using the above documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

#some necessary imports
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import IPython
from IPython.core.getipython import get_ipython
stopwords=stopwords.words('english')
exclude=string.punctuation
lemma=WordNetLemmatizer()

#preprocessing
def preprocess(doc):
    # remove stopwords
    remove_stopwords=" ".join([i for i in doc.lower().split() if i not in stopwords])
    # remove punctuations
    remove_punc="".join([ch for ch in remove_stopwords if ch not in exclude])
    # lemmatization
    normalized=" ".join([lemma.lemmatize(word) for word in remove_punc.split()])
    return normalized

#final preprocessed document
doc_clean=[preprocess(doc).split() for doc in doc_complete]

#some more imports
import gensim
from gensim import corpora

#Dictionary encapsulates the mapping between normalized words and their integer ids
dictionary = corpora.Dictionary(doc_clean)
print(dictionary)

#word id and word freq is made by corpora.gensim.doc2bow(it brings a bag of words model
doc_term_matrix=[dictionary.doc2bow(doc) for doc in doc_clean]

#using gensim.lda model(a predefined model)
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=5)
print(ldamodel.print_topics(num_topics=2, num_words=2))

#for visulaizing the lda generated topics
import pyLDAvis
from pyLDAvis import gensim
vis = gensim.prepare(ldamodel,doc_term_matrix, dictionary)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
