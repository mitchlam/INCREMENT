#!/usr/bin/python

class Instance:

    def __init__(self, data, label=None):
        self.data = data
        self.label = label
    
    def __repr__(self):
        return str(self.data)
        
    @staticmethod
    def distance(x,y):
        return np.linalg.norm(x.data-y.data)
        #return Distance.euclidean(x.data,y.data)
        #return Distance.cityblock(x.data,y.data)
    
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


def classify_data(X,Y, args, holdout = 0.5):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=holdout, random_state=np.random.RandomState())
    
    train_data = np.array(map(lambda x: x.flatten(), X_train))
    test_data = np.array(map(lambda x: x.flatten(), X_test))
    
    
    model = None
    if (args.initial == "perceptron"):
        model = Perceptron()
    elif(args.initial == "svm"):
        model = SVC(kernel="poly")
    elif (args.initial == "GMM"):
        model = GMM(n_components=args.K)
    else:
        raise("Model Not Supported.")
    
    model.fit(train_data, Y_train)
    
    Y_pred = model.predict(test_data)
    
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
    data = np.array(map(lambda x: x.flatten(), X))
     
    if (args.initial == "dbscan"):
        alg = DBSCAN()
    elif (args.initial == "spectral"):
        alg = SpectralClustering(n_clusters=args.K, affinity="nearest_neighbors")
    elif (args.initial == "kmeans"):
        alg = KMeans(n_clusters=args.K, precompute_distances=True, n_jobs=-1)
    elif (args.initial == "mean-shift"):
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples = 500)
        alg = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    elif (args.initial == "complete" or args.initial == "average" or args.initial == "ward"):
        alg = AgglomerativeClustering(n_clusters = args.K, linkage=args.initial)
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
    
   
    alg.fit(data)
    
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

def loadCSV(filename, image=False):
    
    f = open(filename, "r")

    labels = []
    data = []
    targets = []
    
    header = f.readline()
    num = 1

    for line in f:
        l = [i.strip() for i in line.split(",")]

        try:
            if image:
                data.append(loadImage(l[0]))
            else:
                data.append(map(lambda x: float(x), l[:-1]))
                
        except ValueError as e:
            print e
            print l
            print "Line: %d" % (num)
            sys.exit()
        except Exception as e:
            print e
            continue

        lbl = l[-1]
        
        if lbl not in labels:
            labels.append(lbl)
            
        targets.append(lbl)
        
        num += 1

    f.close()
    
    
    
    return np.array(data),np.array(targets)

def loadImage(filename):
    #print "Loading images:", filename
    transformer = caffe.io.Transformer({'data':(1,1, 64, 64)})
    transformer.set_transpose('data', (2,0,1))
    im =  transformer.preprocess("data", caffe.io.load_image(filename, color=False))
    return im


def getData(f, image=False):
    if( f == "iris"):
        return getIris()
    
    if (f == "digit"):
        return getDigit()
    
    return loadCSV(f, image)

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
    #print "Running %s:" % (alg)
    start = time.time()
    #increment.run(minPts=args.minPts, query_size=args.query_size, times_presented=args.times_presented ,labeler = lambda p: p.label, num_queries=args.num_queries)
    increment.run(labeler = lambda p: p.label, **args)
    end = time.time()
    #print "%s: (%d)  --  (%s)" % (alg,increment.num_queries,formatTime(start-end))
    #validation.printMetrics(increment.final)
    
def testIncrement(args, increment, alg="INCREMENT"): 

    print "Testing INCREMENT"
    
    
    start = time.time()
    #increment.run(labeler = lambda p: p.label, **args)
    
    increment.subcluster(labeler = lambda p: p.label, **args)
    increment.selectRepresentatives(labeler = lambda p: p.label, **args)
    
    for i in range(1,len(increment.subclusters),10):
        args["num_queries"] = i
        increment.generateFeedback(labeler = lambda p: p.label, **args)
        increment.mergeSubclusters(labeler = lambda p: p.label, **args)
        
        increment.verbose = 3
        
        print "~!", i, "!~"
        validation.printMetrics(increment.final, printMat=False)
        print
        
        
    
    end = time.time()

    
def main(args):

    starttime = time.time()
    lasttime = starttime
    
    X,Y = getData(args.dataset, args.image)
    
    print "Using: %s (%d)  --  (%s)" % (args.dataset, len(X), formatTime(time.time() - lasttime))
    lasttime = time.time()
    
    clusters = []
    
    print "Initial Clustering:", args.initial
    

        
    if (args.supervised):
        clusters = classify_data(X,Y, args)
    else:
        clusters = cluster_data(X,Y, args)
        

    print "Initial:  --  (%s)" % (formatTime(time.time() - lasttime))
    lasttime  = time.time()

    validation.printMetrics(clusters)

    increment = INCREMENT.MergeINCREMENT(clusters, distance=Instance.distance, aggregator=Instance.aggregate, as_array=Instance.as_array, verbose=args.verbose, convolution=args.convolution)
    if not args.test:
        runIncrement(vars(args),increment)
    else:
        testIncrement(vars(args), increment)

    '''
    other = INCREMENT.MergeINCREMENT(increment.final,distance=Instance.distance, aggregator=Instance.aggregate, as_array=Instance.as_array, verbose=args.verbose)
    runIncrement(vars(args), other, "Other")
    '''
    
    '''
    oracle = INCREMENT.AssignmentINCREMENT(clusters, distance=Instance.distance, aggregator=Instance.aggregate, verbose=False)
    runIncrement(vars(args), oracle, "Oracle")
    '''
    
    print "INCREMENT: (%d)" % (increment.num_queries)
    print "SubClusters:", len(increment.subclusters)
    validation.printMetrics(increment.subclusters, printMat=False)
    
    print
    print "Final"
    validation.printMetrics(increment.final)
    
    
    '''
    print "Other: (%d)" % (other.num_queries)
    
    #print "Subclusters:"
    #validation.printMetrics(other.subclusters)
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

    print "Total Time: %s" %(formatTime(time.time() - starttime))

import os
import sys
import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GMM

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

from sklearn import datasets
from sklearn import metrics
from scipy.spatial import distance as Distance

from sklearn.cross_validation import train_test_split

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Driver to run INCREMENT.")
    parser.add_argument("dataset", help="The dataset to use for input to clustering.")
    parser.add_argument( "-o", "--out", metavar="Output", help="The file in which to store INCREMENT's final clustering.", dest="output")
    parser.add_argument( "-m", "--minPts", help="The minPts parameter to pass to OPTICS.", type=int, default=5)
    parser.add_argument( "-q", "--query-size", help="The number of points to present to the user per query.", type=int, default=1)
    parser.add_argument( "-t", "--times-presented", help="The minimum number of times a point is presented to the user.", type=int)
    parser.add_argument( "-n", "--num-queries", help="The number of queries to answer.", type=int)
    parser.add_argument( "-k", metavar="Clusters" , help="The number of clusters to use with the initial clustering algorithm (where applicable).", type=int, default=20, dest="K")
    parser.add_argument("-i", "--initial", help="Initial clustering algorithm", type=str, default = "kmeans")
    parser.add_argument("-s", "--supervised", help="Specify whether or not to use a supervised model to form initial clustering.", action="store_true")
    parser.add_argument("-v", "--verbose", help="Set INCREMENT to print verbosely.", type=int, default=2)
    parser.add_argument("-N", "--normalize", help="Normalize Data", action="store_true")
    parser.add_argument("-I", "--image", help="Specifies that training data is a list of image location paths", action="store_true")
    parser.add_argument("-C", "--convolution", help="Specifies that training data is a list of image location paths", action="store_true")
    parser.add_argument("-T", "--test", help="Specifies to run tests using the defined settings. SLOW.", action="store_true")
    
    
    args = parser.parse_args()
    
    os.environ['GLOG_minloglevel'] = str(args.verbose)
       
    import incUtils as utils
    import incValidation as validation
    import INCREMENT
    import caffe
    
    main(args)
    
    #print "Verbose", args.verbose


































