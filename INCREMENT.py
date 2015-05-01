
import utils
import optics


#class Cluster:
    
#    def __init__(self,instances, representative = None): #, label=None):
#        self.instances = instances
#        self.representative = representative
        #self.label = label

class BaseINCREMENT(object):
    #Uses naive implementations of everything

    def __init__(self, clustering, distance=utils.EuclideanDistance, **kwargs):
        self.clustering = clustering
        self.subclusters = []
        self.representatives = [] #Index into subclusters

        self.final = [] #store final clustering 

        self.distance = distance #function to determine distance between instances can be set for custom domains
        

    def setInstanceDistance(func):
        self.distance = func

    def subcluster(self, **kwargs):
        self.subclusters = self.clustering

    def selectRepresenatives(self, **kwargs):
        self.representatives = map(lambda x: 0,self.subclusters)

    def queryUser(self, **kwargs):
        pass

    def mergeSubclusters(self, **kwargs):
        self.final = self.subclusters

    def run(self, **kwargs):
        self.subcluster(**kwargs)
        self.selectRepresenatives(**kwargs)
        self.queryUser(**kwargs)
        self.mergeSubclusters(**kwargs)

class OpticsINCREMENT(BaseINCREMENT):

    
    def subcluster(self, minPts=5, **kwargs):
        
        self.subclusters = []
        
        distances = map(lambda x:utils.pairwise(x,self.distance,True), self.clustering) #N^2 where N is the number of instances per cluster -- SLOW
        #print distances
        
        output = map(lambda d: optics.OPTICS(d, minPts), distances)
        separated = map(lambda o: optics.separateClusters(o, minPts), output)
        
        for c,sep in enumerate(separated):
            ids = map(lambda sc: map(lambda x: x._id, sc), sep)
            for sub in ids:
                clust = []
                for i in sub:
                    clust.append(self.clustering[c][i])
                
                self.subclusters.append(clust)
        print

class MatchingINCREMENT(BaseINCREMENT):

    def queryUser(self):
        pass


