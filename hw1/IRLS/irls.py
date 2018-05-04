
# coding: utf-8

# In[41]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold


# In[2]:


TRAIN_DATA = './a9a'
TEST_DATA = './a9a.t'


# In[3]:


def parser(path):
    """Parser for the dataset
    args:
        path: path for train/test dataset
    return:
        pandas dataframe
    """
    data = []
    with open(path, 'r') as f:
        line = f.readline()
        while line != None and line != '':
            sample = dict()
            line = line.split()
            sample['label'] = int(line[0])
            for feat in line[1:]:
                feat_id, feat_value = feat.split(':')
                sample[feat_id] = float(feat_value)  
            data.append(sample)
            line = f.readline()
    data = pd.DataFrame(data).fillna(0)
    return data


# In[4]:


train = parser(TRAIN_DATA)
test = parser(TEST_DATA)


# In[5]:


train_cols = train.columns.values.tolist()
test_cols = test.columns.values.tolist()


# In[6]:


print('Difference between train and test: {}'.format(set(train_cols) - set(test_cols)))


# In[7]:


test['123'] = 0 # impute missing data 
train['intercept'] = 1 # add a dim for intercept 
test['intercept'] = 1 # add a dim for intercept 


# In[8]:


test = test[train.columns] # enusure the corresponding columns 


# In[9]:


train.loc[train.label == -1, 'label'] = 0
test.loc[test.label == -1, 'label'] = 0


# In[10]:


feat_cols = train.columns.tolist()
feat_cols.remove('label')
print('Features: ', feat_cols)


# In[31]:


for idx, feat in enumerate(feat_cols):
    if feat == '123':
        print(idx)


# In[11]:


def sigmoid(x):
    """numerically stable sigmoid"""
    return np.exp(-np.logaddexp(0, -x))


# In[16]:


class IRLS4LR(object):
    def __init__(self, max_iter=1e3, tol=1e-3, l2_penalty=False):
        """the iteration will stop when ``max{|delta_i | i = 1, ..., n} <= tol``
        where ``delta_i`` is the i-th component of the delta w."""
        self.weight = None
        self.l2_norms = []
        self.train_accs= []
        self.test_accs = []
        self.num_iteration = 0
        self.max_iter = max_iter
        self.tol = tol
        self.l2_penalty = l2_penalty
        
    def fit(self, X, y, X_test, y_test):
        """Fit lr model on (X, y) and evaluate on (X_test, y_test)"""
        num_samples, num_features = X.shape
        converged = False
        self.l2_norms = []
        self.train_accs = []
        self.test_accs = []
        self.num_iteration = 0
        self.weight = np.zeros(num_features)
        while not converged:
            self.num_iteration += 1
            mu = self.predict(X)[0]
            R = np.diag(np.multiply(mu, (1 - mu)))
            H = np.matmul(np.matmul(X.transpose(), R), X)
            g = np.matmul(X.transpose(), mu - y)
            if self.l2_penalty:
                H = H + self.l2_penalty * np.identity(H.shape[0])
                g = g + self.l2_penalty * self.weight
            delta = - np.matmul(np.linalg.pinv(H), g) # psuedo inverse for singular matrices
            self.weight += delta
            
            # evaluate
            y_pred = self.predict(X)[1]
            y_test_pred = self.predict(X_test)[1]
            self.l2_norms.append(np.linalg.norm(self.weight))
            self.train_accs.append(accuracy_score(y, y_pred))
            self.test_accs.append(accuracy_score(y_test, y_test_pred))
            print('Iteration {}: Train Accuracy {}, Test Accuracy {}'.format(self.num_iteration, 
                                                                             self.train_accs[-1],
                                                                             self.test_accs[-1]))
            if np.linalg.norm(delta) < self.tol or self.num_iteration > self.max_iter:
                converged = True
                
    def predict(self, X):
        mu = sigmoid(np.matmul(X, self.weight))
        y_pred = np.zeros_like(mu)
        y_pred[mu > 0.5] = 1
        return mu, y_pred


# In[17]:


solver = IRLS4LR()


# In[18]:


X_train = train.loc[:, feat_cols].values
y_train = train['label'].values
X_test = test.loc[:, feat_cols].values
y_test = test['label'].values
print('Train {}, Test {}'.format(X_train.shape, X_test.shape))


# In[40]:


16281 * 0.0001


# In[19]:


solver.fit(X_train, y_train, X_test, y_test)


# In[35]:


solver.l2_norms[-1]


# In[20]:


def cv(X, y, n_fold=5, l2_penalty=0.01):
    """Run K fold cross validation of LR
    return:
        cv_train_acc, cv_test_acc
    """
    kf = KFold(n_splits=n_fold)
    cv_train_accs, cv_test_accs = [], []
    for idx, (cv_train, cv_test) in enumerate(kf.split(X)):
        print('CV Fold {}'.format(idx))
        X_train, y_train = X[cv_train, :], y[cv_train]
        X_test, y_test = X[cv_test, :], y[cv_test]
        solver = IRLS4LR(l2_penalty = l2_penalty)
        solver.fit(X_train, y_train, X_test, y_test)
        cv_train_accs.append(solver.train_accs[-1])
        cv_test_accs.append(solver.test_accs[-1])
    return np.mean(cv_train_accs), np.mean(cv_test_accs)


# In[21]:


for l2_penalty in [0.01, 0.1, 1, 10]:
    cv_train_acc, cv_test_acc = cv(X_train, y_train, l2_penalty=l2_penalty)
    print('l2_penalty={}: cv train acc {}, cv test acc {}'.format(l2_penalty, cv_train_acc, cv_test_acc))


# In[22]:


l2_solver = IRLS4LR(l2_penalty=0.1)
l2_solver.fit(X_train, y_train, X_test, y_test)


# In[34]:


solver.weight[27]


# In[36]:


l2_solver.l2_norms[-1]


# In[ ]:


l2_penalty=100: cv train acc 0.8451061134501477, cv test acc 0.844415148157663
l2_penalty=10: cv train acc 0.8486916991300557, cv test acc 0.8473327961351913
l2_penalty=1: cv train acc 0.8495285852319711, cv test acc 0.847824149470856
l2_penalty=0.8: cv train acc 0.8496283985987286, cv test acc 0.8478241447552826
l2_penalty=0.6: cv train acc 0.8496591105399265, cv test acc 0.8479162823474201
l2_penalty=0.4: cv train acc 0.8497819568309319, cv test acc 0.8478855698167076
l2_penalty=0.2: cv train acc 0.8497742775192245, cv test acc 0.8479777121244186
l2_penalty=0.1: cv train acc 0.8497282101969421, cv test acc 0.8479777121244186
l2_penalty=0.06: cv train acc 0.849720532064264, cv test acc 0.8479469995937061
l2_penalty=0.04: cv train acc 0.8496974985505015, cv test acc 0.8478855792478546   
l2_penalty=0.02: cv train acc 0.8496207198765358, cv test acc 0.8478548667171422
l2_penalty=0.01: cv train acc 0.8496207198765358, cv test acc 0.8478548667171422
l2_penalty=0.001: cv train acc 0.8496283977144566, cv test acc 0.8478241589020034
 


# In[75]:


plt.plot(solver.test_accs, 'b^', label='LR')
plt.plot(l2_solver.test_accs, 'r^', label='L2 regularized LR')
plt.xlabel('num of iterations')
plt.ylabel('test accuracy')


# In[76]:


plt.legend()
plt.show()


# In[72]:


plt.plot(solver.l2_norms, 'bs', label='LR')
plt.plot(l2_solver.l2_norms, 'rs', label='L2 regularized LR')
plt.xlabel('num of iterations')
plt.ylabel('l2 norm of weights')


# In[73]:


plt.legend()
plt.show()

