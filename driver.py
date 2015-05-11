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


def cluster_data(X,Y, method="kmeans", K=5, **kwargs):
    
    alg = None
    
    if (method == "dbscan"):
        alg = DBSCAN(**kwargs)
    elif (method == "spectral"):
        alg = SpectralClustering(n_clusters=K, **kwargs)
    else:
        alg = KMeans(n_clusters=K, **kwargs)
        
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
    
    #clusters = cluster_data(X,Y, method = "kmeans", K=args.K)
    #clusters = cluster_data(X,Y, method = "dbscan", eps=0.5, min_samples=5)
    clusters = cluster_data(X,Y, method = "spectral", K=args.K, affinity="nearest_neighbors")
    #clusters = cluster_kmeans(X,Y,K=1)
    #clusters = cluster_dbscan(X,Y, e=0.5, minPts=5 )
    
    instances = []
    
    for x,y in zip(X,Y):
        instances.append(Instance(x, y))

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
    
    
    args = parser.parse_args()
    main(args)






























