import utils

#class Cluster:
    
#    def __init__(self,instances, representative = None): #, label=None):
#        self.instances = instances
#        self.representative = representative
        #self.label = label

class BaseINCREMENT(object):
    #Uses naive implementations of everything

    def __init__(self, clustering, distance=utils.EuclideanDistance):
        self.clustering = clustering
        self.subclusters = []
        self.representatives = []

        self.final = []

        self.distance = distance
        

    def setInstanceDistance(func):
        self.distance = func

    def subcluster(self):
        self.subclusters = self.clustering

    def selectRepresenatives(self):
        self.representatives = map(lambda x: 0,self.subclusters)

    def queryUser(self):
        pass

    def mergeSubclusters(self):
        self.final = self.subclusters

class OpticsINCREMENT(BaseINCREMENT):

    def subcluster(self):
        pass

class MatchingINCREMENT(BaseINCREMENT):

    def queryUser(self):
        pass


