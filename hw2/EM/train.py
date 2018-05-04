
# coding: utf-8

# In[ ]:


from sklearn.clustering import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from em import EMEstimator
from utils import preprocess, visual_frequent_words


# In[ ]:


newsgroups_train = fetch_20newsgroups(data_home='./', subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(data_home='./', subset='test', remove=('headers', 'footers', 'quotes'))


# In[ ]:


# Data Summary
print('-' * 80)
for idx, cate in enumerate(newsgroups_train.target_names):
    print('Category {}: {}'.format(idx, cate))
print('-' * 80) 
print('# of training documents: {}'.format(len(newsgroups_train.data)))
print('# of test documents: {}'.format(len(newsgroups_test.data)))


# In[ ]:


newsgroups_train.data = preprocess(newsgroups_train.data)
newsgroups_test.data = preprocess(newsgroups_test.data)


# In[ ]:


# Word Occurence Matrix 
vectorizer = CountVectorizer(strip_accents='ascii', 
                             stop_words='english',
                             min_df=10)


# In[ ]:


vectorizer.fit(newsgroups_train.data)


# In[ ]:


word_occur_train = vectorizer.transform(newsgroups_train.data)
word_occur_test = vectorizer.transform(newsgroups_test.data)


# In[ ]:


word_occur_train.shape


# In[ ]:


word_occur_train = word_occur_train.toarray()


# In[ ]:


word_occur_test = word_occur_test.toarray()


# In[ ]:


for k in [5, 10, 20, 30]:
    print('*' * 80)
    print('Number of topics {}'.format(k))
    estimator = EMEstimator()
    estimator.fit(word_occur_train)
    visual_frequent_words(estimator.log_mu, vectorizer.get_feature_names(), topN=15)

