import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import umap

pd.set_option('display.max_colwidth',200)

from sklearn.datasets import fetch_20newsgroups
dataset=fetch_20newsgroups(shuffle=True,random_state=1,remove=('headers','footers','quotes'))
doc=dataset.data
names=dataset.target_names
print(names)

#preprocessing
news=pd.DataFrame({'document':doc})
#remove everthing except alphabets
news['preprocess']=news['document'].str.replace("[^a-zA-Z#]"," ")
#removing short words which are less than 3
news['preprocess']=news['preprocess'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))
#applying lower case to all the remaining words
news['preprocess']=news['preprocess'].apply(lambda x:x.lower())
stopwords=stopwords.words('english')
#tokenization
token=news['preprocess'].apply(lambda x:x.split())
#removal of stop words
token=token.apply(lambda x:[item for item in x if item not in stopwords])

#rejoining the tokens
detoken=[]
for i in range(len(news)):
    t=' '.join(token[i])
    detoken.append(t)
news['preprocess']=detoken

#TF-IDF vectorizer
#creates a matrix where the rows are your docs topics and the columns will be top 1000 frequency words
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True)
X = vectorizer.fit_transform(news['preprocess'])

#SVD decomposition
#returns the svd having 20 as dimentionality of data-topics and 1000 words(top)
from sklearn.decomposition import TruncatedSVD
svd_model=TruncatedSVD(n_components=20,algorithm='randomized',n_iter=25,random_state=12)
svd_model.fit(X)

terms=vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    #zipping the top 1000 terms results with a particular topic
    terms_comp = zip(terms, comp)
    #sorting all those top words for a particular topic
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:5]
    print("Topic "+str(i+1)+": ")
    for t in sorted_terms:
        print(t[0],end=" ")
        print("")

X_topics = svd_model.fit_transform(X)

#mapping this dimensionality space into a graphical format using umap
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)
plt.scatter(embedding[:, 0], embedding[:, 1],c= dataset.target,s=5,edgecolor='none')
plt.show()