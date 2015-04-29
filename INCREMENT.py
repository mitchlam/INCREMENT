import utils

class Instance:

    def __init__(self, data): #, label=None):
        self.data = data
        #self.label = label

    def distTo(other):
        return utils.EclideanDistance(data,other.data)

class Cluster:
    
    def __init__(self,instances, representative = None): #, label=None):
        self.instances = instances
        self.representative = representative
        #self.label = label

class BaseINCREMENT(object):
    #Uses naive implementations of everything

    def __init__(self, clustering):
        self.clustering = clustering
        self.subclusters = []
        

    def subcluster(self):
        self.subclusters = self.clustering

    def selectRepresenatives(self):
        map(lambda x: x.representative = x.instances[0], self.subclusters)

    def queryUser(self):
        pass

    def mergeSubclusters(self):
        pass

class OpticsINCREMENT(BaseINCREMENT):

    def subcluster(self):
        pass

class MatchingINCREMENT(BaseINCREMENT):

    def queryUser(self):
        pass


