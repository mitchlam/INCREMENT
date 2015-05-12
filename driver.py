#!/usr/bin/python

import sys
import numpy as np
import utils
import validation

import argparse

import INCREMENT

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from sklearn import datasets
from sklearn import metrics
from scipy.spatial import distance as Distance

from sklearn.cross_validation import train_test_split


class Instance:

    def __init__(self, data, label=None):
        self.data = data
        self.label = label
        
    @staticmethod
    def distance(x,y):
        return Distance.euclidean(x.data,y.data)


def classify_data(X,Y, args, holdout = 0.8):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=holdout, random_state=np.random.RandomState())
    
    model = None
    if (args.initial == "perceptron"):
        model = Perceptron()
    elif(args.initial == "svm"):
        model = SVC(kernel="poly")
    else:
        raise("Model Not Supported.")
    
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    labels = set(Y_pred)
    
    k = len(labels)
    
    labels = list(labels)
    
    clusters = []
    
    for i in range(k):
        clusters.append([])
    
    for x,y,t in zip(X_test, Y_pred, Y_test):
        clusters[labels.index(y)].append(Instance(x,t))
        
    return clusters
    

def cluster_data(X,Y, args):
    
    alg = None
    
    if (args.initial == "dbscan"):
        alg = DBSCAN()
    elif (args.initial == "spectral"):
        alg = SpectralClustering(n_clusters=args.K, affinity="nearest_neighbors")
    elif (args.initial == "kmeans"):
        alg = KMeans(n_clusters=args.K)
    else:
        raise("Model Not Supported.")
        
    alg.fit(X)
    
    labels = alg.labels_
    
    k = len(set(labels))
    
    clusters = []
    
    for i in range(k):
        clusters.append([])
        
    for x,y,t in zip(X,labels,Y):
        clusters[int(y)].append(Instance(x,t))
    
    return clusters
    

def getIris():
    iris = datasets.load_iris()
    return iris.data, iris.target

def getDigit():
    digits = datasets.load_digits()
    
    return digits['data'], digits['target']

def loadCSV(filename):
    
    f = open(filename, "r")

    labels = []
    data = []
    targets = []
    
    header = f.readline()
    
    for line in f:
        l = [i.strip() for i in line.split(",")]
        data.append(map(lambda x: float(x), l[:-1]))
        lbl = l[-1]
        
        if lbl not in labels:
            labels.append(lbl)
            
        targets.append(lbl)

    f.close()
    
    
    
    return np.array(data),np.array(targets)

def getData(f):
    if( f == "iris"):
        return getIris()
    
    if (f == "digit"):
        return getDigit()
    
    
    return loadCSV(f)

def main(args):
    X,Y = getData(args.dataset)
    
    print "Using: %s (%d)" % (args.dataset, len(X))
    
    clusters = []
    
    print "Initial Clustering:", args.initial
    

        
    if (args.supervised):
        clusters = classify_data(X,Y, args)
    else:
        clusters = cluster_data(X,Y, args)
        

    print "Initial:"
    validation.printMetrics(clusters)

    increment = INCREMENT.ClosestINCREMENT(clusters, distance=Instance.distance)
    increment.run(minPts=args.minPts, query_size=args.query_size, times_presented=args.times_presented ,labeler = lambda p: p.label)
    

    
    print "INCREMENT: (%d)" % (increment.num_queries)
    validation.printMetrics(increment.final)
    
    
    #write data and cluster to file
    #try:
    if (args.output != None):    
        filename = args.output
        
        f = open(filename,"w")
        
        print "Writing Clustering to", filename
        for c,cluster in enumerate(increment.final):
            for instance in cluster:
                s = ""
                for i in instance.data:
                    s += "%s, " % (str(i))
                
                f.write("%s%s, %s\n" % (s, str(c), str(instance.label)))
        f.close()
    #except:
    pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Driver to run INCREMENT.")
    parser.add_argument("dataset", help="The dataset to use for input to clustering.")
    parser.add_argument( "-o", "--out", metavar="Output", help="The file in which to store INCREMENT's final clustering.", dest="output")
    parser.add_argument( "-m", "--minPts", help="The minPts parameter to pass to OPTICS.", type=int, default=5)
    parser.add_argument( "-q", "--query-size", help="The number of points to present to the user per query.", type=int, default=9)
    parser.add_argument( "-t", "--times-presented", help="The minimum number of times a point is presented to the user.", type=int, default=1)
    parser.add_argument( "-k", metavar="Clusters" , help="The number of clusters to use with the initial clustering algorithm (where applicable).", type=int, default=20, dest="K")
    parser.add_argument("-i", "--initial", help="Initial clustering algorithm", type=str, default = "kmeans")
    parser.add_argument("-s", "--supervised", help="Specify whether or not to use a supervised model to form initial clustering.", action="store_true")
    
    
    args = parser.parse_args()
    main(args)






























