import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from scipy import stats
import re
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
import warnings
from sklearn.utils import resample
import statistics
import pyodbc
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')
%matplotlib inline 


cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password+'')

df_tss = pd.read_sql(sql,cnxn)
cnxn.close()

df_tss['agency_txt'] = df_tss['agency_txt'].str.upper()

sentences = df_tss[df_tss['TSSComments'].isnull()==False]['TSSComments']

sid = SentimentIntensityAnalyzer()

df_tss['COMPOUND_SENT'] = df_tss['TSSComments'].apply(lambda x: sid.polarity_scores(x)['compound'] if pd.isnull(x)==False else None)

stop = set(stopwords.words('english'))
useless_words = ['would','could','should','le','non','federal','government','agency','way']
exclude = set(string.punctuation) 
for word in useless_words:
    stop.add(word)
    
def clean(doc):
    
    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    def strip_urls(text):
        #url regex
        url_re = re.compile(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")
        stripped_text = url_re.sub('',text)
        return stripped_text

    def strip_emails(text):
        #email address regex
        email_re = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
        stripped_text = email_re.sub('',text)
        return stripped_text

    def strip_nonsense(text):
        # leave words that are at least three characters long, do not contain a number, and are no more 
        # than 17 chars long
        no_nonsense = re.findall(r'\b[a-z][a-z][a-z]+\b',text)
        stripped_text = ' '.join(w for w in no_nonsense if w != 'nan' and len(w) <= 17)
        return stripped_text
    
    doc = doc.lower()
    tag_free = strip_html_tags(doc)
    url_free = strip_urls(tag_free)
    email_free = strip_emails(url_free)
    normalized_1 = strip_nonsense(email_free)
    
    stop_free = " ".join([i for i in normalized_1.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in punc_free.split())
    
    return normalized
    
def topic_model(x):
    
    n_samples = 2000
    n_features = 1000
    n_components = 7
    n_top_words = 10
    
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()


    print("Loading dataset...")
    t0 = time()
    data_samples = x
    print("done in %0.3fs." % (time() - t0))

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english',
                                    ngram_range  = (1,2))
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english',
                                   ngram_range = (1,2))
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print()

    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features, n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    
def tokenize_text(x):
    raw_text = x.tolist()

    text_data = []
    for text in raw_text:
        tokens = clean(text)
        text_data.append(tokens)
    
    return text_data

def lda_to_list (x):
    n_samples = 2000
    n_features = 1000
    n_components = 6
    n_top_words = 10
    #max_df=0.95, min_df=2,
    tf_vectorizer = CountVectorizer(
                                max_features=n_features,
                                stop_words='english',
                                   ngram_range = (1,2))

    tf = tf_vectorizer.fit_transform(x)
    
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda.fit(tf)
    temp_list =[]
    for topic_idx, topic in enumerate(lda.components_):
        #message = "Topic #%d: " % topic_idx
        message = ''
        message += ", ".join([tf_feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]])
        temp_list.append(message)
    return temp_list

unique_agency = df_tss['agency_txt'].unique()

lda_list = []

for agency in unique_agency:
    temp_list = []
    df_temp = df_tss[df_tss['agency_txt']==agency]
    sentences_temp = df_temp[df_temp['TSSComments'].isnull()==False]['TSSComments']
    return_list = lda_to_list(tokenize_text(sentences_temp))
    return_list.append(agency)  
    return_list.append(len(df_temp))
    if return_list is None:
        print(agency)
        break
    lda_list.append(return_list)
    
    
rand_list = lda_to_list(tokenize_text(sentences))    

rand_list.append('ALL GOV')
rand_list.append(0)

lda_list.append(rand_list)

cols_temp = ['topic_1','topic_2','topic_3','topic_4','topic_5','topic_6','agency','comm_len']
df_lda = pd.DataFrame(lda_list,columns =cols_temp )

df_lda['comm_len'] = pd.to_numeric(df_lda['comm_len'])

df_lda_long = df_lda[df_lda['comm_len']>25]

df_lda_melt = pd.melt(df_lda_long,id_vars='agency',value_vars = [x for x in list(df_lda_long.columns) if x !='agency'])
