

import sys
import numpy as np
import utils

import INCREMENT

from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.spatial import distance as Distance


class Instance:

    def __init__(self, data, label=None):
        self.data = data
        self.label = label
        
    @staticmethod
    def distance(x,y):
        return Distance.euclidean(x.data,y.data)


def confusion(clustering):
    labels = [-1]*len(clustering)

    conf = {}

    for i,cluster in enumerate(clustering):
        trueVals = map(lambda x: x.label, cluster)
        label = utils.mode(trueVals)
        
        for l in trueVals:
            if l not in conf:
                conf[l] = [0]*len(labels)
            
            conf[l][i] += 1

        
        
        

    final = []
    for key,value in conf.items():
        final.append(value)

    #print "Final Error : %d of %d" % (error, total)
    return final

def checkAccuracy(clustering):
    labels = [-1]*len(clustering)

    conf = {}
    error = 0.0
    total = 0
    for cluster in clustering:
        trueVals = map(lambda x: x.label, cluster)
        label = utils.mode(trueVals)
        
        #print "Label: %d" % (label)
        
        e = 0.0
        for l in trueVals:
            if l not in conf:
                conf[l] = [0]*len(labels)
            
            conf[l][label] += 1
            if l != label:
                e += 1
        
        #print "Error: %d of %d: %f" % (e, len(cluster), e/len(cluster)) 
        error += e
        total += len(cluster)
        
    if total == 0:
        return (0,0,0)
    
    #print "Final Error : %d of %d" % (error, total)
    return (error, total, error/total)

def main(args):

    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target

    print "Iris: %d" %(len(X))

    K = 3
    
    clusters = []
    
    for i in range(K):
        clusters.append([])

    kmeans = KMeans(n_clusters=K)

    kmeans.fit(X)
    labels = kmeans.labels_
    

    for x,y,t in zip(X,labels,Y):
        #print "Predicted: %d Actual: %d: Instance: %s" %(y,t,str(x))
        clusters[y].append(Instance(x,t))

    print "Initial Error: %d of %d: %f" % (checkAccuracy(clusters))
    conf = confusion(clusters)
    
    for i,line in enumerate(conf):
        print "%d:\t%s" % (i, str(line))
    
    increment = INCREMENT.OpticsINCREMENT(clusters, Instance.distance)

    increment.subcluster()
    increment.selectRepresenatives()
    increment.queryUser()
    increment.mergeSubclusters()
    
    
    print "Final Error: %d of %d: %f" % (checkAccuracy(increment.final))
    conf = confusion(increment.final)
    
    for i,line in enumerate(conf):
        print "%d:\t%s" % (i, str(line))

if __name__ == "__main__":
    main(sys.argv)






























