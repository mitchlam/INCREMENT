

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
            
        targets.append(labels.index(lbl))

    f.close()
    
    
    
    return np.array(data),np.array(targets)

def getData():
    #return getIris()
    return loadCSV("COIL.csv")

def main(args):

    X,Y = getData()
    #print X[:5]
    clusters = cluster_kmeans(X,Y,K=20)
    #clusters = cluster_dbscan(X,Y, e=0.5, minPts=5 )
    
    print "Initial:"
    validation.printMetrics(clusters)

    increment = INCREMENT.INCREMENT(clusters, distance=Instance.distance)

    increment.run(minPts=5, query_size=9, times_presented=2 ,labeler = lambda p: p.label)
    
    print "Final:"
    validation.printMetrics(increment.final)
    

if __name__ == "__main__":
    main(sys.argv)






























