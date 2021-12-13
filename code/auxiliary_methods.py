import itertools

import numpy as np



def pow2db(x):
    return 10 * np.log10(x)


def db2pow(xdb):
    return 10.**(xdb/10.)


def JainsFairness(C):
# JainsFairness returns the Jain's fairness measure given the inputted array
#   OUTPUT: 
#       * fairness - Jain's fairness measure from 0 to 1
#   INPUT:
#       * C - array of capacities that each WLAN experiences
    C = np.array(C).reshape((1, -1))
    numRows = len(C)
    fairness = np.zeros(numRows)

    for i in range(numRows):
        fairness[i] = np.sum(C[i])**2 / (len(C[i])*np.sum(C[i]**2))
    return fairness


def allcomb(*args):
# ALLCOMB - All combinations
#    B = ALLCOMB(A1,A2,A3,...,AN) returns all combinations of the elements
#    in A1, A2, ..., and AN. B is P-by-N matrix is which P is the product
#    of the number of elements of the N inputs.
#    Empty inputs yields an empty matrix B of size 0-by-N. Note that
#    previous versions (1.x) simply ignored empty inputs.
#
#    Example:
#       allcomb([1 3 5],[-3 8],[0 1]) 
#         1  -3   0
#         1  -3   1
#         1   8   0
#         ...
#         5  -3   1
#         5   8   1
#
#    This functionality is also known as the cartesian product.
#
    return np.array(list(itertools.product(*args)))


def indexes2val(i,j,k,a,b): 
# indexes2val provides the index ix from i, j, k (value for each variable) 
#   OUTPUT:  
#       * ix - index of the (i,j,k) combination 
#   INPUT:  
#       * i - index of the first element 
#       * j - index of the second element 
#       * k - index of the third element 
#       * a - size of elements containing "i" 
#       * b - size of elements containing "j" 
#           (size of elements containing "k" is not necessary) 
    ix = i + j*a + k*a*b; 
    return ix 

 
def val2indexes(ix, a, b, c): 
# val2indexes provides the indexes i,j,k from index ix 
#   OUTPUT:  
#       * i - index of the first element 
#       * j - index of the second element 
#       * k - index of the third element 
#   INPUT:  
#       * ix - index of the (i,j,k) combination 
#       * a - size of elements containing "i" 
#       * b - size of elements containing "j" 
#       * c - size of elements containing "k" 
    i = ix % a; 
    y = ix % (a * b) 
    j = y//a 
    k = ix // (a * b) 

    assert ix == i+a*j+a*b*k 

    return i, j, k 
