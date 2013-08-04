import random
import numpy as np

class Classifier(object):
    """base classifier class"""
    
    def __init__(self, D):
        pass
        
    def fit():
        pass
        
    def predict():
        pass

class LogisticRegression(Classifier):
    '''
    Logistic regression multinomial classifier.
    Logistic Regression learns functions of the form f :X -> Y, or
    P(Y|X) in the case where Y is discrete-valued, and X = <X1 ...Xn> is any vector
    containing discrete or continuous variables.
    '''
    def __init__(self, num_classes, num_features):
        '''
        num_classes:   K
        num_features:  M
        learning rate: epsilon

        Regularization not currently implemented.
        (Plans to implement L1 regularization in the future.)
        '''
        self.num_classes = num_classes
        self.num_features = num_features
        self.W = np.zeros((num_topics, num_words), dytpe=float)
        
    def prob_y_given_x(self, x):
        '''
        Compute the logistic function for p(y=k | x) for each class k
        and return an array of probabilities. [p1, p2, p3, ..., pk].
        '''
        exp_inner_of_theta_and_x = np.exp(np.dot(theta, x))
        Z = sum(exp_inner_of_theta_and_x)
        return exp_inner_of_theta_and_x / Z

    def predict(self, x):
        '''
        What do we believe given some evidence?
        '''
        return argmax(p_y_given_x(x))
    
    def update(self, X=None, label=None, learning_rate=0.01):
        '''
        iterate gradient decent

        PARAMS:
        X:
        --
        A feature vector used when computing the linear predictor function.

        label:
        ------
        The label associated with x.  The loss function makes use of this 
        to 
        
        learning_rate:
        --------------
        Size used in the gradient descent algorithm. Effectivly how far
        the  MLE point estimate will be moved in the direction of the gradient 
        each iteratrion.
        '''

        if X or label is None:
            raise ValueError
        
        #shorthand for learning rate
        eta = learning_rate 
            
        #weight states before and after update
        theta_0 = self.W
        theta_1 = None
        
        theta_1 = theta_0 + eta * self.gradient((X,label)) - r * theta_0
        self.W = theta_1
        return self

    def gradient(self, p):
        """
        Compute the gradient at point p
        """
        x_i = p[0]
        y_i = p[1]
        log_p_y = np.log(self.p_y_given_x(x_i))
        err = -prob_y + np.eye(self.num_classes)[y_i]
        return np.outer(err, x_i)

    def fit(self, D, stop_condition=None, epsilon=0.5, batch=False):
        '''
        Fit the model's parameters to training data using gradient descent
        to maximize model parameters W.

        PARAMS:
        
        D:
        --
        A collection containing the data to train on.
        
        stop_cond:
        ----------
        A function to determine when the model has been fit to the data.
        
        epsilon:
        --------
        Size used in the gradient descent algorithm. Effectivly how far
        the MLE point estimate will be moved in the direction of the gradient 
        each iteratrion.
        
        batch:
        ------
        Boolean indicating whether to run stochiastic or batch gradient descent.
        (maybe add minibatch as an option in the future.)
        '''
        while not stop_condition():
            if batch:
                #batch gradient descent
                print """Get a better computer"""
                break
            else:
                #stochiastic gradient descent
                for data in D:
                    self.update(data)
        return self

    
