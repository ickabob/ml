
import random
from collections import defaultdict
from operator import itemgetter

def get20Newsdata():
    with open('../20-news-parse.txt', 'r') as f:
        D = []
        while True:
            label = f.readline().strip()
            if not label:
                break # EOF
            words = set(f.readline().strip().split(' '))
            D.append((words, label))
    return D

def getCongressionalData():
    with open('../house-votes-84.data', 'r') as f:
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
    
def distance20(x1, x2):
    return len(x1 | x2) - len(x1 & x2)

def distanceVotes(x1, x2):
    differences = 0
    for p1,p2 in zip(x1,x2):
        if p1 or p2 == '?':
            break
        if p1 != p2:
            differences += 1
    return differences        

def knn_predict(D, x, k):
    distances = [(yp, distanceVotes(x, xp)) for xp, yp in D]
    if k == 1:
        # k = 1 is treated as a special case
        return min(distances, key=itemgetter(1))[0]
    # k > 1
    distances.sort(key=itemgetter(1))
    nn = distances[:k]
    counts = defaultdict(int)
    for yp, d in nn:
        counts[yp] += 1
    return max(counts.items(), key=itemgetter(1))[0]

def test_knn(k=1, percent_train=0.85, N=None):
    D = getCongressionalData()
    N_original = len(D)
    if N is None:
        N = len(D)
    else:
        # only use N examples from the data set
        D = D[:N]

    num_train = int(percent_train * N)
    num_test = N - num_train
    
    # print infor about this run
    print 'using {0} of {1} examples'.format(N, N_original)
    print 'number train:', num_train
    print 'number test:', num_test
    print 'k:', k

    D_train = D[:num_train] # {(x_1, y_1),...(x_num_train-1, y_num_train-1)}
    D_test = D[num_train:] # {(x_num_train, y1_num_train),...(x_N, y_N)}

    errors = 0
    for x, y in D_test:
        yhat = knn_predict(D_train, x, k)
        errors += int(y != yhat)
    print errors / float(num_test)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k-nearest-neighbors', type=int, default=1,
                        help='number of nearest neighbors to use')
    parser.add_argument('-p', '--percent-train', type=float, default=0.85,
                        help='percentage of data to use for training')
    parser.add_argument('-N', '--number', type=int, default=None,
                        help='amount of data to use')
    parser.add_argument('-d', '--dataset', type=int, default=None,
                        help='dataset to use')
    args = parser.parse_args()

    for i in xrange(100):
        test_knn(i+1, args.percent_train, args.number)
    
