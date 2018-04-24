import numpy as np
from scipy.special import logsumexp


def log_normalize(log_prob, axis):
    """Normalize log_prob over axis"""
    log_sum = logsumexp(log_prob, axis=axis)
    
    if not isinstance(log_sum, np.ndarray):
        log_sum = np.array([log_sum])
    if log_prob.shape[0] == log_sum.shape[0]:
        # column normalize 
        return (log_prob.transpose() - log_sum).transpose()
    else:
        # row normalize
        return log_prob - log_sum


class EMEstimator(object):
    """EM estimator for mixture of multinomial model"""
    def __init__(self, num_mixtures=20, tol=0):
        """
        args:
            word_occur: word occurrence matrix for the corpus
        """
        self.num_mixtures = num_mixtures # K
        self.tol = tol
        self.log_pi, self.log_mu = None, None
    
    def ll_obs(self):
        word_occur, log_pi, log_mu = self.word_occur, self.log_pi, self.log_mu
        log_gamma_hat = np.matmul(word_occur, log_mu) + log_pi # (D, K)
        ll_obs = logsumexp(log_gamma_hat, axis=1).sum() # (D,)
        return ll_obs
        
    def init_params(self, method, kmeans_pi=None, kmeans_mu=None):
        word_occur = self.word_occur
        num_mixtures = self.num_mixtures
        num_doc, num_word = word_occur.shape # D, W
        if method == 'uniform':
            log_pi = - np.log(num_mixtures) * np.ones(num_mixtures)
            per_word_occur = word_occur.sum(axis=0) # T_w
            with np.errstate(divide='raise'):
                try:
                    log_mu_k = np.log(per_word_occur) - np.log(per_word_occur.sum()) # \mu_k
                except FloatingPointError:
                    print('Divide by zero, check if you have zero value in the log !')
            log_mu = np.repeat(log_mu_k.reshape(-1, 1), num_mixtures, axis=1) # \mu
        elif method == 'random':
            pi = np.random.randint(1, 9, size=num_mixtures)
            log_pi = np.log(pi) - np.log(pi.sum())
            mu = np.random.randint(1, 9, size=(num_word, num_mixtures))
            log_mu = np.log(mu) - np.log(mu.sum(axis=0))
        elif method == 'k-means':
            log_pi, log_mu = np.log(kmeans_pi), np.log(kmeans_mu) 
            
        assert log_pi.shape == (num_mixtures, )
        assert log_mu.shape == (num_word, num_mixtures)
        self.log_pi, self.log_mu = log_pi, log_mu
    
    def E_step(self):
        word_occur, log_pi, log_mu = self.word_occur, self.log_pi, self.log_mu
        num_doc, num_word = word_occur.shape
        num_mixtures = self.num_mixtures
        
        log_gamma_hat = np.matmul(word_occur, log_mu) + log_pi # (D, K)
        log_gamma = log_normalize(log_gamma_hat, axis=1) # (D, K)
        
        assert log_gamma.shape == (num_doc, num_mixtures)
        self.log_gamma = log_gamma
    
    def M_step(self):
        word_occur, log_gamma = self.word_occur, self.log_gamma
        num_mixtures = self.num_mixtures
        num_doc, num_word = word_occur.shape
        # Note the tricky reshape
        log_mu_hat = logsumexp(log_gamma.reshape(num_doc, num_mixtures, 1),
                               b=word_occur.reshape(num_doc, 1, num_word), 
                               axis=0).transpose() # (W, K)
        log_mu = log_normalize(log_mu_hat, axis=0) # (W, K)
        
        log_pi_hat = logsumexp(log_gamma, axis=0) # (K, )
        log_pi = log_normalize(log_pi_hat, axis=0)
        assert log_pi.shape == (num_mixtures,)
        assert log_mu.shape == (num_word, num_mixtures)
        self.log_pi, self.log_mu = log_pi, log_mu

    def fit(self, word_occur, init_method='random'):
        self.word_occur = word_occur
        self.init_params(method=init_method)
        
        prev_ll_obs =  self.ll_obs() # log likelihood of the observed data
        delta_ll_obs = np.infty # increment of log likelihood
        
        num_iter = 0
        while delta_ll_obs >= self.tol:
            self.E_step()
            self.M_step()
            
            curr_ll_obs = self.ll_obs()
            delta_ll_obs = curr_ll_obs - prev_ll_obs
            prev_ll_obs = curr_ll_obs
            
            num_iter += 1
            print(delta_ll_obs)
            print('Iteration {}: log likelihood of the observed data {}'.format(num_iter, curr_ll_obs))

    def predict(word_occur):
        assert self.log_mu != None and self.log_pi != None, "Fit the model before predicting !"
        target = None
        return target