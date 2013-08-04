from collections import defaultdict
from functools import partial
from myutil import take, grouper
from operator import itemgetter

import pdb

def getdata():
    try:
        D = []
        datafilepath = "./20-news-parse.txt"
        with open(datafilepath, 'r') as f:
            for subjline, articleline in grouper(f,2):
                x,y = (subjline.strip(), articleline.strip().split())
                D.append((x,set(y)))
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    finally: 
        return D
        
def knn_predict(k, d, Ds):
    """
    Compare d pairwise to the whole collection and coallate the distances.
    """
    def distance(x, xp):
        return len(x | xp) - len(x & xp)
        distances = [(xp, distance(d, yp)) for xp,yp in Ds]
        if k == 1:
            #the closest p
            return min(distances, key=itemgetter(1))[0]
            #k > 1
            distances.sort(key=itemgetter(1))
            nn = distances[:k]
            pdb.set_trace()
            counts = defaultdict(int)
            for xp, dist in nn:
                counts[xp] += 1
                return max(count.items(), key=itemgetter(1))[0]
                
def classify(k=1, percent_train=0.70, N=None):
    D = getdata()
    N_original = len(D)
    if N is None:
        N = len(D)
    else:
        D = D[:N]
        
    num_train = int(percent_train * N)
    num_test = N - num_train
    
    print '''using {0} of {1} datapoints
    number train: {2}
    number test: {3}
    k: {4}'''.format(N, N_original, num_train, num_test, k)
    
    D_train = D[:num_train] # {(x_1, y_1),...(x_num_train-1, y_num_train-1)}
    D_test = D[num_train:] # {(x_num_train, y1_num_train),...(x_N, y_N)}
    
    errors = 0
    for x, y in D_test:
        xhat = knn_predict(k, y, D_train)
        errors += int(xhat != x)
    print errors / float(num_test)
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--k-nearest-neighbors', type=int, default=1,
                        help='number of nearest neighbors to use')
    parser.add_argument('-p', '--percent-train', type=float, default=0.70,
                        help='percentage of data to use for training')
    parser.add_argument('-N', '--number', type=int, default=None,
                        help='amount of data to use')
    args = parser.parse_args()
    classify(args.k_nearest_neighbors, args.percent_train, args.number)
