
import utils
import optics


#class Cluster:
    
#    def __init__(self,instances, representative = None): #, label=None):
#        self.instances = instances
#        self.representative = representative
        #self.label = label

class BaseINCREMENT(object):
    #Uses naive implementations of everything

    def __init__(self, clustering, distance=utils.EuclideanDistance, symmetric_distance=True, **kwargs):
        self.clustering = clustering
        self.subclusters = []
        self.representatives = [] #actual points, one for each subcluster. Indexes should be aligned with subcluster
        self.feedback = [] #list of feedback. Should be clusterings of indexes of representative points

        self.final = [] #store final clustering 

        self.distance = distance #function to determine distance between instances can be set for custom domains
        self.symmetric_distance = symmetric_distance #Bool stating whether or not the distance is symmetric

    def setInstanceDistance(func):
        self.distance = func

    def subcluster(self, **kwargs):
        self.subclusters = self.clustering

    def selectRepresenatives(self, **kwargs):
        self.representatives = map(lambda x: x[0],self.subclusters)

    def generateFeedback(self, **kwargs):
        pass

    #Actually presents a set of points to the user, and returns the feedback
    #returns a list of clustered indexes into pts 
    def query(self, pts, **kwargs):
        return [[i] for i in range(len(pts))]

    def mergeSubclusters(self, **kwargs):
        self.final = self.subclusters

    def run(self, **kwargs):
        self.subcluster(**kwargs)
        self.selectRepresenatives(**kwargs)
        self.generateFeedback(**kwargs)
        self.mergeSubclusters(**kwargs)

class OpticsSubclustering(BaseINCREMENT):

    #Performs OPTICS to subcluster the current clustering
    def subcluster(self, minPts=5, **kwargs):
        
        self.subclusters = []
        
        distances = map(lambda x:utils.pairwise(x,self.distance, self.symmetric_distance), self.clustering) #N^2 where N is the number of instances per cluster -- SLOW
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

class ClosestPointFeedback(BaseINCREMENT):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, query_size=9, **kwargs):
        
        #include index to retrieve the actual point after sorting
        rep_distances = map(lambda d: zip(d, range(len(d))) ,utils.pairwise(self.representatives, self.distance, self.symmetric_distance) )
        
        queue = range(len(self.representatives))
        
        #list of indexes that have been queried
        queried = []
        
        feedback = []
        
        while len(queue) > 0:
            i = queue[0]
            del queue[0]
            
            #skip points already queried
            if i in queried:
                continue
            
            dist = sorted(rep_distances[i][:])
            
            #ensure we dont ask for too many points
            size = query_size
            if (len(dist) < size):
                size = len(dist)
            
            pt_idx = map(lambda d: d[1], dist[:query_size])
            pts = map(lambda x: self.representatives[x], pt_idx)
            
            q = self.query(pts, **kwargs)
            feedback.append(map(lambda c: map(lambda x: pt_idx[x], c), q)) #translate pt indexes to the indexes of the representatives
            
            #add pts to index to avoid querying again
            for idx in pt_idx:
                if( idx not in queried):
                    queried.append(idx)
    
        self.feedback = feedback
        
        print "Feedback: "
        for f in feedback:
            print "\t", f


class OracleMatchingINCREMENT(ClosestPointFeedback):
    
    #Cheats and looks at label. Simulates a perfect user.
    def query(self, labels = None, **kwargs):
        pass




class INCREMENT(OpticsSubclustering, ClosestPointFeedback):
    pass




















