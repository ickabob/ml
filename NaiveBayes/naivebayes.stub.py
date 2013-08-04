import random, sys
import utility
import numpy as np

class BiNomNiaveBayes(utility.Classifier):
    def __init__(self, D, num_features, alpha0=2, beta0=2, alpha1=2, beta1=2):
        '''
        Build a Binary Naive Bayes classifier from a dataset.
        
        :param D: the dataset consisting of (x, y) pairs
        :param num_features: the dimension of our datapoints
        :param alpha0: 1st hyperparameter of the Beta class prior for theta_pi
        :param beta0: 2nd hyperparameter of the Beta class prior for theta_pi
        :param alpha1: 1st hyperparameter of the Beta prior for theta_ki
        :param beta1: 2nd hyperparameter of the Beta prior for theta_ki

        '''
        super(D, num_features)
        
        num_republicans = 0
        for x,y in D:
            if x == 'republican':
                num_republicans += 1
        self.num_republicans = num_republicans
        self.num_democrats = self.size - self.num_republicans

class MultiNomNaiveBayes(utility.Classifier):
    def __init__(self, D, num_topics, num_words, beta0=2, beta1=2, alpha=2):
        '''Build a MultiCatigorical Naive Bayes classifier from a dataset.
        
        :param D: the dataset consisting of (x, y) pairs
        :param num_topics: the number of topics
        :param num_words: the number of words
        :param beta0: 1st hyperparameter of the Beta prior for theta_ki
        :param beta1: 2nd hyperparameter of the Beta prior for theta_ki
        :param alpha: hyperparameters of the Dirichlet prior for pi_k
        '''
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_docs = len(D)
        # calculate N_ki = word_counts[k,i]
        # calculate N_k = topic_counts[k]
        self.word_counts = np.zeros((self.num_topics, self.num_words), dtype=float)
        self.topic_counts = np.zeros(self.num_topics, dtype=float)
        for x, y in D:
            self.topic_counts[y] += 1
            self.word_counts[y,x] += 1
        # ----------------------------------------------------------------------- #
        # Need to fill in the computation of:
        # log_pi,         pi = (N_k + alpha - 1) / (N + K * alpha - K)
        # log_theta,      theta_ki = (N_ki + beta0 - 1) / (N_k + beta0 + beta1 - 2)
        # log_thata_not,  theta_not_ki = 1 - theta_ki
        # ----------------------------------------------------------------------- #
        self.log_pi = np.zeros(self.num_topics, dtype=float)
        self.log_theta = np.zeros((self.num_topics, self.num_words), dtype=float)
        self.log_theta_not = np.zeros((self.num_topics, self.num_words), dtype=float)
        
        for k in xrange(num_topics):
            # log_pi,         pi = (N_k + alpha - 1) / (N + K * alpha - K)
            pi_k_numer = (self.topic_counts[k] + alpha - 1)
            pi_k_denom = (self.num_docs + self.num_topics * alpha - self.num_topics)
            self.log_pi[k] = np.log(pi_k_numer) - np.log(pi_k_denom) # log(N_k/N) = log(N_k) - log(N)
  
            for i in xrange(num_words):
                # log_theta,      theta_ki = (N_ki + beta0 - 1) / (N_k + beta0 + beta1 - 2)
                theta_ki_numer = (self.word_counts[k,i] + beta0 - 1)
                theta_ki_denom = (self.topic_counts[k] + beta0 + beta1 - 2)
                theta_ki = theta_ki_numer / float(theta_ki_denom)
                
                self.log_theta[k,i] = np.log(theta_ki_numer) - np.log(theta_ki_denom)
                self.log_theta_not[k,i] = np.log(1 - theta_ki)


    def predict(self, x):
        # -------------------------------------------------------------------------------------------#
        # Need to fill in the computation of:
        # log_p_y,         p(y=k), i=k,...,num_topics
        # log_p_x_given_y, p(x|y=k)=p(x_1|y=k)*p(x_2|y=k)*...*p(x_num_words|y=k), k=1,...,num_topics
        # -------------------------------------------------------------------------------------------#
        # should be a array of size num_topics
        log_p_y = self.log_pi

        # should be an array of size num_topics, see hint below
        log_p_x_given_y = np.zeros(self.num_topics, dtype=float)
        xmask = np.zeros((self.num_words), dtype=int)
        for i in x:
            xmask[i] = 1

        log_p_x_given_y = np.choose(xmask, (self.log_theta_not, self.log_theta)).sum(axis=1)
        #pdb.set_trace()
        log_p_y_given_x = log_p_y + log_p_x_given_y
        # predict k which maximizes p(y=k|x) \propto p(y=k)p(x|y=k)
        return log_p_y_given_x.argmax()
        
'''
HINT FOR COMPUTING p(x|y)
to compute log_p_x_given_y we need to do the following

for k in range(self.num_topics):
    log_p_x_given_y_equal_k = 0.
    for i in range(self.num_words):
        if i is in x:
            log_p_x_given_y_equal_k += log_theta[k,i]
        else:
            log_p_x_given_y_equal_k += log_theta_not[k,i]

This for loop will be REALLY SLOW if num_words is large, which it is.
Lucky for us we can implement something that looks like the above in
Numpy using the np.choose function to perform all the selection, and
then sum the selections, again using Numpy because we're smart.

Here is an example of how the choose function works.

A = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3],
              [4, 4, 4]])
B = np.array([[-1, -1, -1],
              [-2, -2, -2],
              [-3, -3, -3],
              [-4, -4, -4]])
x = np.array([0, 1, 0])
C = np.choose(x, (A, B))

The result is a new matrix C, with the same shape as A and B, 
where each element C_ij is equal to A_ij if x_j = 0 and B_ij if x_j = 1
it looks like
C = [[1, -1, 1],
     [2, -2, 2],
     [3, -3, 3],
     [4, -4, 4]]

Pretty fancy eh?
'''

def train_and_test(percent_train=0.80, percent_validate=0.10, alpha_one=1, beta_one=1, beta_two=1, num_docs=None):
    getdata = utility.getdata20News
    D, vocab_index_dict, topic_index_dict = getdata()
    num_topics = len(topic_index_dict)
    num_words = len(vocab_index_dict)
    orig_num_docs = len(D)
    
    if num_docs is None: 
        num_docs = orig_num_docs
    else:
        D = D[:num_docs]

    num_train = int(percent_train * num_docs)
    #num_validate = int(percent_validate * num_docs)
    num_test = num_docs - num_train - num_validate
    
    Dtrain = D[:num_train]
    #Dvalid = D[num_train:num_valid]
    Dtest = D[num_train:]
    
    # print info about this run
    print 'using {0} of {1} documents'.format(num_docs, orig_num_docs)
    print 'number of train documents:', num_train
    print 'number of test documents:', num_test
    print 'number of topics:', num_topics
    print 'vocabulary size:', num_words
    
    nb = NaiveBayes(Dtrain, num_topics, num_words)
    errors = 0.
    Dtest = Dtest[:50]
    for i, (x, y) in enumerate(Dtest):
        if not i % 500:
            # print status messages to stderr so we don't get buffered
            print >> sys.stderr, '{0} of {1}'.format(i, num_test)
        yhat = nb.predict(x)
        if y != yhat:
            errors += 1
    print '% error:', errors / num_test
    
if __name__ == '__main__':
    # You chould easily augment this stuff below to allow you to pass 
    # the hyperparameters (beta0, beta1, alpha) from the command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--percent-train', type=float, default=0.85,
                        help='percentage of data to use for training')
    parser.add_argument('-n', '--num-docs', type=int, default=None,
                        help='number of documents to use')
    parser.add_argument('-a1', '--alpha-one', type=int, default=None,
                        help='alpha parameter used dirchlet distribution for hte class priors')
    parser.add_argument('-b1', '--beta-one', type=int, default=None,
                        help='beta parameter used dirchlet distribution for hte class priors')
    parser.add_argument('-b2', '--beta-two', type=int, default=None,
                        help='beta parameter used dirchlet distribution for hte class priors')
    args = parser.parse_args()
    train_and_test(args.percent_train, args.num_docs, args.alpha_one, args.beta_one, args.beta_two )
