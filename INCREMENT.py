
import utils
import optics
import scipy.sparse.csgraph as csgraph


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
        
        self.num_queries = 0

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
        self.num_queries += 1
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
    def subcluster(self, minPts=5, display=False, **kwargs):
        
        self.subclusters = []
        
        print "Computing Distance"
        distances = map(lambda x:utils.pairwise(x,self.distance, self.symmetric_distance), self.clustering) #N^2 where N is the number of instances per cluster -- SLOW
        #print distances
        
        print "Running OPTICS: minPts = %d" % (minPts)
        output = map(lambda d: optics.OPTICS(d, minPts), distances)
        separated = map(lambda o: optics.separateClusters(o, minPts, display=display), output)
        
        print "Sub-Clustering:"
        
        for c,sep in enumerate(separated):
            ids = map(lambda sc: map(lambda x: x._id, sc), sep)
            lengths = []
            for sub in ids:
                clust = []
                for i in sub:
                    clust.append(self.clustering[c][i])
                lengths.append(len(clust))
                
                self.subclusters.append(clust)
            print "\t%d: %d => %s" %(c, len(lengths), lengths)
        
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

#Base Feedback class. Simply queries for the label of each point individually and sets feedback accordingly.
class AssignmentFeedback(BaseINCREMENT):
    
    def printFeedback(self, feedback):
        print "Feedback:", len(feedback)
        for f in feedback:
            print "\t", f
        
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):
        print "Assignment Query"
        labels = {}
        
        
        for r,rep in enumerate(self.representatives):           
            lbl = self.query([rep], **kwargs)
            if lbl not in labels:
                labels[lbl] = []
            
            labels[lbl].append(r)
            
        feedback = []
        
        for lbl, cluster in labels.items():
            feedback.append([cluster])
            
        self.feedback = feedback
        
        
        self.printFeedback(feedback)
        print
        print "Number of Assignement Queries: %d" % (self.num_queries)
        print
   
class MatchingFeedback(AssignmentFeedback):
    
    #Distances should be the pairwise distances between the representatives
    def generateFeedback(self, distances, query_size=9, times_presented=1, **kwargs):
        
        #can only perform matching if query_size > 1
        if(query_size == 1):
            super(MatchingFeedback, self).generateFeedback(**kwargs)
            return
        
         #include index to retrieve the actual point after sorting
        rep_distances = map(lambda d: zip(d, range(len(d))) , distances )
        
        queue = range(len(self.representatives))
        
        #list of indexes that have been queried
        queried = {}
        
        #initialize queried
        for i in queue:
            queried[i] = 0
        
        feedback = []
        
        while len(queue) > 0:
            i = queue[0]
            del queue[0]
            
            #skip points already queried
            if queried[i] >= times_presented:
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
                queried[idx] += 1
    
        self.feedback = feedback
        
        self.printFeedback(feedback)
        print 
        print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
        print
        
class ClosestPointFeedback(MatchingFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):    
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        super(ClosestPointFeedback,self).generateFeedback(distances, **kwargs)
       
    #distances should be the pairwise distances between the reps
    


class MinimumSpanningTreeFeedback(MatchingFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        mst = csgraph.minimum_spanning_tree(distances)
        distances = csgraph.shortest_path(mst,method="D", directed=False)
        
        super(MinimumSpanningTreeFeedback, self). generateFeedback(distances, **kwargs)
        

class MinimumDistanceFeedback(MatchingFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)

        distances = csgraph.shortest_path(distances,method="D", directed=self.symmetric_distance)
        
        super(MinimumDistanceFeedback, self). generateFeedback(distances, **kwargs)
        
################################# Query #################################################

#If only a single point is presented, return it's label 
class OracleMatching(BaseINCREMENT):
    
    #Cheats and looks at target. Simulates a perfect user.
    #labeler is a function that accepts an instance and returns its label
    def query(self, pts,  labeler=None, **kwargs):
        
        #if no labeling function is provided, default to parent implementation
        if labeler == None:
            raise("Labeler not provided.")
        
        self.num_queries += 1
        
        #If only 1 point, this is an assignment query. i.e. return it's label.
        if(len(pts) == 1):
            return  labeler(pts[0])
    
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
        
        #print "Query:"
        for label, points in clusters.items():
            #print "\tLabel [%s]: %s" % (str(label), str(points))
            feedback.append(points)
        
        #print "\tUnknown:", unknown
        
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


        
class ClosestINCREMENT(OpticsSubclustering, MedoidSelector, ClosestPointFeedback, OracleMatching, MergeSubclusters):
    pass

class TreeINCREMENT(OpticsSubclustering, MedoidSelector, MinimumSpanningTreeFeedback, OracleMatching, MergeSubclusters):
    pass
class PathINCREMENT(OpticsSubclustering, MedoidSelector, MinimumDistanceFeedback, OracleMatching, MergeSubclusters):
    pass



















