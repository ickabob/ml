import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from myutil import grouper

D = [] #List of tuples, [(x1,y1), (x2,y2), ... (xn, yn)]
Xs = []
Ys = []
datafilepath = ("./20-news-parse.txt");

try:
    with open(datafilepath, 'r') as f:
        for line1, line2 in grouper(f,2):
            point = (line1.strip(),line2.strip())
            D.append(point)
            Xs.append(point[0])
            Ys.append(point[1])
except:
    pass
finally:
    ntopics = len(set(Xs))
    words = list()
    for line in Ys:
        for word in line.split():
            words.append(word)

    tpcfrequencies = Counter ([x for x in Xs]).most_common(ntopics)
    firsthundred = Counter (words).most_common(100)
    secondhundred = Counter (words).most_common(200)[100:200]

    fig1, (ax1) = plt.subplots(1)
    ind = np.arange(ntopics)
    
    fig1.suptitle("Usenet article subjects")
    ax1.bar(ind, [x[1] for x in tpcfrequencies], .20)
    ax1.set_xlabel("topic")
    ax1.set_ylabel("occurences")
    ax1.set_xticks(ind)
    ax1.set_xticklabels([x[0] for x in tpcfrequencies], rotation=80)
    plt.tight_layout()
    
    fig2, (ax2) = plt.subplots(1)
    ind = np.arange(len(set(firsthundred)))

    fig2.suptitle("100 most occuring words")
    ax2.bar(ind, list(x[1] for x in firsthundred), .10)
    ax2.set_xlabel("word")
    ax2.set_ylabel("occurences")
    ax2.set_xticks(ind)
    ax2.set_xticklabels([x[0] for x in firsthundred], rotation=80)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    
    fig3, (ax3) = plt.subplots(1)
    ind = np.arange(len(set(secondhundred)))
        
    fig3.suptitle("100-200 most occuring words")
    ax3.bar(ind, list(x[1] for x in secondhundred), .10)
    ax3.set_xlabel("word")
    ax3.set_ylabel("occurences")
    ax3.set_xticks(ind)
    ax3.set_xticklabels([x[0] for x in secondhundred], rotation=80)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
