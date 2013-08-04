#We can caluculate the number of n step walks from a vertex i to vertex
#  j by inspecting the ij_th entry of A^n = A*A*A*A*...(n times) *A*A.
#  Calculate the number of 7 step walked from vertex 2 to vertex 7.

import numpy as np

print '''================================================================================
CS4950 HW1 PART1\nIan Kane\n28-Jun-13
================================================================================
'''

_string ='''0 1 0 0 1 0 0
1 0 1 0 0 0 1
0 1 0 1 0 1 1
0 0 1 0 1 1 1
1 0 0 1 0 1 0
0 0 1 1 1 0 1
0 1 1 1 0 1 0'''

def tallywalk(M, n, i, j):
    return np.linalg.matrix_power(M,n)[i,j]

M = np.matrix (_string.replace('\n', ';' ))
sevenstep27walk = tallywalk(M, 7, 1, 6)
eigset = (spectrumM, normed_eigvecsM) =  np.linalg.eig(M)

print '''M:\n {0}\n
Number of 7 step walks from vertex 2 to vertex 7:\n {1}\n
The set of eigenvalues (spectrum):\n {2}'''.format(M, sevenstep27walk, spectrumM)




    
