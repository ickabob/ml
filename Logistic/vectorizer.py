import numpy as np
import scipy.sparsne as sp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import DEFAULT_ANALYZER, WordNGramAnalyzer

class InclusionVectorizer(BaseEstimator):
    """
    Convert a collection of raw documents to a matrix of bits where a_i_j is high
    when vocabulary word i is present in document j.
    
    This implementation produces a sparse representation of the document using
    scipy.sparse.coo_matrix


    Parameters
    ----------
    analyzer: WordNGramAnalyzer or CharNGramAnalyzer, optional

    vocabulary: dict or iterable, optional
        Either a dictionary where keys are tokens and values are indices in
        the matrix, or an iterable over terms (in which case the indices are
        determined by the iteration order as per enumerate).

        This is useful in order to fix the vocabulary in advance.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.
    
    """
    def __init__(self, analyzer=DEFAULT_ANALYZER, vocabulary=None,
                 max_features=None):
        self.analyzer = analyzer
        self.fit_vocabulary = vocabulary is None
        if vocabulary is not None and not isinstance(vocabulary, dict):
            vocabulary = dict((t,i) for i, t in enumerate(vocabulary))
        self.vocabulary = vocabulary
        self.max_features = max_features

    def fit(self, raw_documents):
        """Learn a vocabulary dictionary of all tokens in the raw documents

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents):
        """Learn the vocabulary dictionary and return the count vectors

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: array, [n_samples, n_features]
        """
        if not self.fit_vocabulary:
            self.transform(raw_documents)

        terms = set()
        term_sets_per_doc = list()
        
        max_features = self.max_features
        
        for doc in raw_documents:
            terms_current = set(self.analyzer.analyzer(doc))
            terms.update(terms_current)

            term_sets_per_doc.append(terms_current)
        n_docs = len(terms_per_doc)

        self.vocabulary = dict((t,i) for i,t in enumerate(terms))
        return self._term_sets_to_matrix(term_sets_per_doc)
        

    def transform(self, raw_documents):
        """Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided in the constructor.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: sparse matrix, [n_samples, n_features]
        """
    
        if not self.vocabulary:
            raise ValueError("Vocabulary wasn't fitted or is empty!")
        
        terms_per_doc = [set(analyzer.analyze(doc))  
                               for doc in raw_documents]
        return self._term_sets_to_matrix(terms_per_doc)

    def _term_sets_to_matrix(self, term_sets):
        i_indices = list()
        j_indices = list()
        values = list()
        vocabulary = self.vocabulary

        for i, term_set in enumerate(term_sets):
            for term in term_set:
                j = vocabulary.get(term)
                if j is not None:
                    i_indices.append(i)
                    j_indices.append(j)
                    values.append(1)
            #free memory some memory we dont need
            term_set.clear()
        shape = (len(term_sets), max(vocabulary.itervalues()) + 1)
        return sp.coo_matrix((values, (i_indices, j_indices)),
                             shape=shape, dtype=int))
        
