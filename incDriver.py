#!/usr/bin/python

import sys
import numpy as np
import incUtils as utils
import incValidation as validation
import time
import random

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
    
    def __repr__(self):
        return str(self.data)
        
    @staticmethod
    def distance(x,y):
        #return np.linalg.norm(x.data-y.data, ord = 1)
        #return Distance.euclidean(x.data,y.data)
        return Distance.cityblock(x.data,y.data)
    
    @staticmethod
    def as_array(x):
        return x.data
    
    @staticmethod
    def aggregate(instances):
            
        data = map(lambda i: i.data, instances)
        labels = map(lambda i : i.label, instances)
        
        data = np.array(data)
    
        
        #print "Labels:", labels
        label = utils.mode(labels)
        
        I = Instance(np.average(data, axis=0), label)
        
        
        return I


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
        alg = KMeans(n_clusters=args.K, precompute_distances=True, n_jobs=-1)
    elif (args.initial == "active"):
        clusters = []

        for x,y in zip(X,Y):
            clusters.append([Instance(x,y)])

        return clusters

    elif (args.initial == "none"):
        clusters = [[]]

        for x,y in zip(X,Y):
            clusters[0].append(Instance(x,y))

        return clusters
    
    elif(args.initial == "random"):
        clusters = []
        K = args.K
        for k in range(K):
            clusters.append([])

        for x,y in zip(X,Y):
            k = random.randint(0,K-1)
            clusters[k].append(Instance(x,y))
            
            
        return clusters

    elif(args.initial == "pre"):
        clusters = {}
        
        for x,y in zip(X,Y):
            if x[-1] not in clusters:
                clusters[x[-1]] = []
            
            clusters[x[-1]].append(Instance(x[:-1],y)) #Remove the id column and assignment columns
        
        '''
        print clusters.keys()
        print
        print
        '''
        #print clusters.values()
        
        return clusters.values()
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
    num = 1

    for line in f:
        l = [i.strip() for i in line.split(",")]

        try:
            data.append(map(lambda x: float(x), l[:-1]))
        except ValueError as e:
            print e
            print l
            print "Line: %d" % (num)
            sys.exit()
        except:
            pass

        lbl = l[-1]
        
        if lbl not in labels:
            labels.append(lbl)
            
        targets.append(lbl)
        
        num += 1

    f.close()
    
    
    
    return np.array(data),np.array(targets)

def getData(f):
    if( f == "iris"):
        return getIris()
    
    if (f == "digit"):
        return getDigit()
    
    
    return loadCSV(f)

def formatTime(t):

    def separate(t,s):
        period = 0
        if (int(t) / s > 0):
            period = int(t) / s
            while (t-s >= 0):
                t -= s
        return (t, period)

    t, days = separate(t, 3600*24)
    t, hours = separate(t,3600)
    seconds, minutes = separate(t, 60)

    result = ""
    
    if days > 0:
        result += "%d d " % (days)
    if hours > 0:
        result += "%d h " % (hours)
    if minutes > 0:
        result += "%d m " % (minutes)

    result += "%f s" %(seconds)

    return result
   
def runIncrement(args, increment, alg="INCREMENT"): 
    print "Running %s:" % (alg)
    start = time.clock()
    #increment.run(minPts=args.minPts, query_size=args.query_size, times_presented=args.times_presented ,labeler = lambda p: p.label, num_queries=args.num_queries)
    increment.run(labeler = lambda p: p.label, **args)
    end = time.clock()
    #print "%s: (%d)  --  (%s)" % (alg,increment.num_queries,formatTime(start-end))
    #validation.printMetrics(increment.final)
    
def main(args):

    starttime = time.clock()
    lasttime = starttime
    
    X,Y = getData(args.dataset)
    
    print "Using: %s (%d)  --  (%s)" % (args.dataset, len(X), formatTime(time.clock() - lasttime))
    lasttime = time.clock()
    
    clusters = []
    
    print "Initial Clustering:", args.initial
    

        
    if (args.supervised):
        clusters = classify_data(X,Y, args)
    else:
        clusters = cluster_data(X,Y, args)
        

    print "Initial:  --  (%s)" % (formatTime(time.clock() - lasttime))
    lasttime  = time.clock()

    validation.printMetrics(clusters)

    increment = INCREMENT.MergeINCREMENT(clusters, distance=Instance.distance, aggregator=Instance.aggregate, as_array=Instance.as_array, verbose=args.verbose)
    runIncrement(vars(args),increment)

    '''
    other = INCREMENT.OtherINCREMENT(clusters,distance=Instance.distance, aggregator=Instance.aggregate, verbose=args.verbose)
    runIncrement(vars(args), other, "Other")
    '''
    
    '''
    oracle = INCREMENT.AssignmentINCREMENT(clusters, distance=Instance.distance, aggregator=Instance.aggregate, verbose=False)
    runIncrement(vars(args), oracle, "Oracle")
    '''
    
    print "INCREMENT: (%d)" % (increment.num_queries)
    #print "SubClusters"
    #validation.printMetrics(increment.subclusters)
    
    print
    print "Final"
    validation.printMetrics(increment.final)
    
    
    '''
    print "Other: (%d)" % (other.num_queries)
    
    print "Subclusters:"
    validation.printMetrics(other.subclusters)
    print
    validation.printMetrics(other.final)
    '''
    
    '''    
    print "Oracle: (%d)" %(oracle.num_queries)
    validation.printMetrics(oracle.final)
    '''
    
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

    print "Total Time: %s" %(formatTime(time.clock() - starttime))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Driver to run INCREMENT.")
    parser.add_argument("dataset", help="The dataset to use for input to clustering.")
    parser.add_argument( "-o", "--out", metavar="Output", help="The file in which to store INCREMENT's final clustering.", dest="output")
    parser.add_argument( "-m", "--minPts", help="The minPts parameter to pass to OPTICS.", type=int, default=5)
    parser.add_argument( "-q", "--query-size", help="The number of points to present to the user per query.", type=int, default=9)
    parser.add_argument( "-t", "--times-presented", help="The minimum number of times a point is presented to the user.", type=int)
    parser.add_argument( "-n", "--num-queries", help="The number of queries to answer.", type=int)
    parser.add_argument( "-k", metavar="Clusters" , help="The number of clusters to use with the initial clustering algorithm (where applicable).", type=int, default=20, dest="K")
    parser.add_argument("-i", "--initial", help="Initial clustering algorithm", type=str, default = "kmeans")
    parser.add_argument("-s", "--supervised", help="Specify whether or not to use a supervised model to form initial clustering.", action="store_true")
    parser.add_argument("-v", "--verbose", help="Set INCREMENT to print verbosely.", action="store_true")
    parser.add_argument("-N", "--normalize", help="Normalize Data", action="store_true")
    
    
    args = parser.parse_args()
    main(args)
    
    #print "Verbose", args.verbose






























