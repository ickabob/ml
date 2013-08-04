
import numpy as np

from .classifier import LogisticRegression
from ..datasets import load_20NewsGroup

def test():
    
    useNetPosts = myutils.fetch_20newsgroups(data_home="../datasets/20news",
                                             subset='all',
                                             shuffle=False)
    #set data proportions
    percent_train = .80
    percent_validate = .10
    percent_test = .10
    
    
    
    
if __name__ == "__main__" and __package__ is None:
    __package__ = "cs5950"
    test()
