

import sys
import numpy as np
import utils

import INCREMENT

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
from scipy.spatial import distance as Distance


class Instance:

    def __init__(self, data, label=None):
        self.data = data
        self.label = label
        
    @staticmethod
    def distance(x,y):
        return Distance.euclidean(x.data,y.data)


def contingency(clustering):
    labels = [-1]*len(clustering)

    cont = {}

    for i,cluster in enumerate(clustering):
        trueVals = map(lambda x: x.label, cluster)
        label = utils.mode(trueVals)
        
        for l in trueVals:
            if l not in cont:
                cont[l] = [0]*len(labels)
            
            cont[l][i] += 1

        
        
        

    final = []
    for key,value in cont.items():
        final.append(value)

    #print "Final Error : %d of %d" % (error, total)
    return final


def homogeneity(cont,eps=10e-10):
    if(len(cont[0]) == 1):
        return 1.0
    
    x = 0.0
    label_counts = map(sum,cont)
    N = float(sum(label_counts))
    cluster_sizes = [0.0]*len(cont[0])
    num_labels = len(cont)
    
    #H(C|K)
    for label, row in enumerate(cont):
        for cluster, val in enumerate(row):
                cluster_sizes[cluster] += val
    
    for label, row in enumerate(cont):
        for cluster, val in enumerate(row):
            one = val/float(N)
            two = val/ cluster_sizes[cluster]
            if (two + eps != 0.0):
                x += one * np.log(two+eps)
    
    x *= -1
    
    y = 0.0
    
    #H(C)
    for count in label_counts:
        num = eps + count / N
        if(num != 0.0):
            y += (num) * np.log( num ) #different from paper
    
    y *= -1
    
    return 1.0 - (x / y)

def completeness(cont, eps=10e-10):
    if(len(cont[0]) == 1):
        return 1.0
    
    x = 0.0
    label_counts = map(sum,cont)
    N = sum(label_counts)
    cluster_sizes = [0.0]*len(cont[0])
    num_labels = len(cont)
    
    
    for label, row in enumerate(cont):
        for cluster, val in enumerate(row):
            #H(K|C)
            one = val/float(N)
            two = val/float(label_counts[label])
            if (two+eps != 0.0):
                x += one * np.log(two+eps)
            
            #H(K)
            cluster_sizes[cluster] += val
    
    x *= -1
    
    y = 0.0
    
    for size in cluster_sizes:
        num = eps + size / N
        if(num != 0.0):
            y += (num) * np.log( num ) #Different from paper
    
    y *= -1
    
    return 1.0 - (x / y)

def V_measure(cont, B=1):
    h = homogeneity(cont)
    c = completeness(cont)
    
    v = ((1 + B) * h * c) / (( B * h) + c)
    
    return v

def All_measures(cont, B=1):
    h = homogeneity(cont)
    c = completeness(cont)
    
    v = ((1 + B) * h * c) / (( B * h) + c)
    
    return (h,c,v)

def checkAccuracy(cont):

    total = 0.0
    correct =  0.0
    
    m = [0] * len(cont[0])
    idx = [0] * len(cont[0])
    
    for i,r in enumerate(cont):
        for j,v in enumerate(r):
            if i == j:
                total += v
            else:
                total += v
            
            if(v > m[j]):
                idx[j] = i
                m[j] = v
    
    for i,r in enumerate(cont):
        for j,v in enumerate(r):
            if i == idx[j]:
                correct += v

    error = total-correct
    #print "Final Error : %d of %d" % (error, total)
    return (error, total, error/total)

def cluster_kmeans(X, Y, K=3):
    clusters = []
    
    for i in range(K):
        clusters.append([])

    kmeans = KMeans(n_clusters=K)

    kmeans.fit(X)
    labels = kmeans.labels_

    #measures = metrics.homogeneity_completeness_v_measure(Y, labels)    

    for x,y,t in zip(X,labels,Y):
        #print "Predicted: %d Actual: %d: Instance: %s" %(y,t,str(x))
        clusters[y].append(Instance(x,t))
        
    #print "KMeans: H: %f, C: %f, V: %f" %(measures)
    return clusters

def printMetrics(cluster):
    cont = contingency(cluster)
    
    print "Error: %d of %d: %f" % (checkAccuracy(cont))
    print "H: %f C: %f V: %f" % (All_measures(cont))
    
    utils.print_cont(cont)

    print

def getIris():
    iris = datasets.load_iris()
    
    print "Iris: %d" %(len(iris.data))
    return iris.data, iris.target

def getDigit():
    digits = datasets.load_digits()
    
    print "Digits: %d" % (len(digits['data']))
    
    return digits['data'], digits['target']

def getData():
    return getDigit()

def main(args):

    X,Y = getData()

    
    
    clusters = cluster_kmeans(X,Y,K=5)

    print "Initial:"
    printMetrics(clusters)

    increment = INCREMENT.OpticsINCREMENT(clusters, distance=Instance.distance)

    increment.run(minPts=10)
    
    
    print "Final:"
    printMetrics(increment.final)
    
    

if __name__ == "__main__":
    main(sys.argv)






























