"""
Machine Learning CS5950
=======================

cs5950 is a Python module written for a course at WMU in the 
summer of 2013.  CS5950 - Machine Learning is an over view
of classic ML models and algorithms.

This module depends heavily on Scipy/Numpy/matplotlib

-Ian K.
 Sat Aug  3 19:55:32 EDT 2013

"""

try:
    from numpy.testing import nosetester

        class NoseTester(nosetester.NoseTester):
        """ Subclass numpy's NoseTester to add doctests by default
        """

        def test(self, label='fast', verbose=1, extra_argv=['--exe'],
                        doctests=True, coverage=False):
            """Run the full test suite

            Examples
            --------
            This will run the test suite and stop at the first failing
            example
            >>> from cs5950 {import test
            >>> test(extra_argv=['--exe', '-sx']) #doctest: +SKIP
            """
            return super(NoseTester, self).test(label=label, verbose=verbose,
                                    extra_argv=extra_argv,
                                    doctests=doctests, coverage=coverage)

    test = NoseTester().test
    del nosetester
except:
    pass

__all__ = ['NaiveBayes', 'Logistic', 'Knn', 'datasets']
__version__ = '0.01'
