

import sys
import numpy as np
import utils
import validation

import INCREMENT

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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

def cluster_dbscan(X, Y, e=0.3, minPts=5):
    dbscan = DBSCAN(eps=e, min_samples = minPts)

    dbscan.fit(X)
    labels = dbscan.labels_

    K = len(set(labels))

    clusters = []
    
    for i in range(K):
        clusters.append([])

    #measures = metrics.homogeneity_completeness_v_measure(Y, labels)    

    for x,y,t in zip(X,labels,Y):
        #print "Predicted: %d Actual: %d: Instance: %s" %(y,t,str(x))
        clusters[int(y)].append(Instance(x,t))
        
    #print "KMeans: H: %f, C: %f, V: %f" %(measures)
    return clusters

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
    
    #clusters = cluster_kmeans(X,Y,K=10)
    clusters = cluster_dbscan(X,Y, e=0.5, minPts=5 )
    
    print "Initial:"
    validation.printMetrics(clusters)

    increment = INCREMENT.INCREMENT(clusters, distance=Instance.distance)

    increment.run(minPts=2, query_size=9, labeler = lambda p: p.label)
    
    
    print "Final:"
    validation.printMetrics(increment.final)
    

if __name__ == "__main__":
    main(sys.argv)






























