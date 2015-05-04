
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

    def selectRepresentatives(self, **kwargs):
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
        print "Running INCREMENT"
        print
        
        self.subcluster(**kwargs)
        self.selectRepresentatives(**kwargs)
        self.generateFeedback(**kwargs)
        self.mergeSubclusters(**kwargs)

################################# Sub-Clustering ##########################################################
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
        
        print "Subclusters Formed:", len(self.subclusters)
        print 
        

################################# Representative Selection #################################################
class MedoidSelector(BaseINCREMENT):
    
    def selectRepresentatives(self, **kwargs):
        self.representatives = []
        
        distances = map(lambda sc: utils.pairwise(sc, self.distance, self.symmetric_distance), self.subclusters)
        
        reps = []
        
        for i, dist in enumerate(distances):
            sums = map(sum, dist)
            m = utils.arg_min(sums)
            reps.append(m)
            self.representatives.append(self.subclusters[i][m])
        
        print "Representatives:"
        print reps
        print
        
################################# Query Ordering #################################################

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
        
        print "Feedback:", len(feedback)
        for f in feedback:
            print "\t", f

################################# Query #################################################

class OracleMatching(BaseINCREMENT):
    
    #Cheats and looks at target. Simulates a perfect user.
    #labeler is a function that accepts an instance and returns its label
    def query(self, pts,  labeler=None, **kwargs):
        
        #if no labeling function is provided, default to parent implementation
        if labeler == None:
            return super(OracleMatching, self).query(pts, **kwargs)
        
        #dictionary from label to point index
        clusters = {}
        
        #list for unknown points
        unknown = []
        
        for i,p in enumerate(pts):
            label = labeler(p)
            
            if label == None:
                unknown.append(i)
                continue
            
            if label in clusters:
                clusters[label].append(i)
            else:
                clusters[label] = [i]
       
        #Separate from dictionary
       
        feedback = []
        
        for label, points in clusters.items():
            feedback.append(points)
        
        for point in unknown:
            feedback.append([point])
        
        return feedback
    
################################# Merging #################################################

class MergeSubclusters(BaseINCREMENT):
    
    def mergeSubclusters(self, **kwargs):
        self.final = []
        feedback = []
        
        for f in self.feedback:
            feedback += f
        
        
        #print "flattened Feedback:"
        #print "\t", feedback
        
        sets = map(set, feedback)
        
        changed = True
        while changed:
            changed = False
            for i,x in enumerate(sets):
                for j,y in enumerate(sets):
                    if i >= j:  
                        continue
                    
                    if(not x.isdisjoint(y)):
                        x.update(y)
                        del sets[j] #double check this. If j doesnt get updated correctly, will cause problem
                        changed = True
        
        feedback = []
        for s in sets:
            cluster = []
            f = []
            for i in s:
                f.append(i)
                cluster += self.subclusters[i]
            self.final.append(cluster)
            feedback.append(f)
        
        print "Merged Feedback:"
        print "\t", feedback
        print 


        
class INCREMENT(OpticsSubclustering, MedoidSelector, ClosestPointFeedback, OracleMatching, MergeSubclusters):
    pass




















