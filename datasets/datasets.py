# -*- coding: utf-8 -*-
# Author:  Ian Kane <ian.c.kane@wmich.edu>
#
# Thanks:  Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          And any other sklearn contributors.
#
# License: BSD Style.

import os
import logging
from os import environ
from os.path import join
from sklearn.datasets.twenty_newsgroups import download_20newsgroups
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups


import numpy as np


logger = logging.getLogger(__name__)

VOCAB_NAME = "vocabulary.txt"

'''
Took a note from sklearn and how they handle their
datasets and this is just handy
'''
def get_data_home(data_home=None):
    if data_home is None:
        data_home = environ.get('CS5950_LEARN_DATA',
                                join('~','datasets'))
    data_home = expanduser(data_home)
    if not os.path.exists(data_home):
        makedirs(data_home)
    return data_home

    
def fetch_20newsgroups_bows(subset='all',
                            data_home="/home/ian/School/CS5950/datasets/"):
    """
    Load the 20 newsgroups dataset and transform in into bag-of-words.
    This is a wrapper around sklearn.fetch_20newsgroups

    Parameters:
    -----------
    
    subset: ['train'|'test'|'all'], optional
        Select the dataset to load

    data_home: optional, default: None
        Specify an download and cache folder for the datasets.  

    Returns
    -------
    
    bunch : Bunch object
        bunch.data: sparse matrix, shape [n_samples, n_features]
        bunch.target: array, shape [n_samples]
        bunch.target_names: list, length [n_classes]
    """
    data_home = get_data_home(data_home=data_home)
    target_file = os.path.join(data_home, "20newsgroup_bow.pk")

    data_train = fetch_20newsgroups(data_home=data_home,
                                    subset='train',
                                    catagories=None,
                                    shuffle=True,
                                    random_state=12)
    data_test = fetch_20newsgroups(data_home=data_home,
                                    subset='test',
                                    catagories=None,
                                    shuffle=True,
                                    random_state=12)

    if os.path.exists(target_file):
        X_train, X_test = joblib.load(target_file)
    else:
        vocabulary = dict((t,i) for i, t in enumerate(open(vocab_path)))
        
    
        
    
def fetch_CongressionalVotes(data_home="/home/ian/CS5950/datasets/"):
    with open(data_home, 'r') as f:
        D = []
        while True:
            line = f.readline().strip()
            if not line:
                break # EOF
            votes = line.split(',')
            label = votes[0]
            votes = votes[1:]
            D.append((votes, label))
    return D
