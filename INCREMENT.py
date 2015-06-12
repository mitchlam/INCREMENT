
import utils
import optics
import HMRF

import random
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

class CentroidINCREMENT(BaseINCREMENT):
    
    def __init__(self, clustering, aggregator, **kwargs):
        super(CentroidINCREMENT, self).__init__(clustering, **kwargs)
        self.aggregator = aggregator
        
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
        
class CentroidSelector(CentroidINCREMENT):
    
    def selectRepresentatives(self, **kwargs):
        self.representatives = []
        reps = map(self.aggregator, self.subclusters)
        self.representatives = reps
        
        '''
        print "Representatives:"
        print self.representatives
        print
        '''
        
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
   
class MinimumDistanceFeedback(AssignmentFeedback):
    
    #Distances should be the pairwise distances between the representatives
    def generateFeedback(self, distances, query_size=9, times_presented=1, num_queries=None, **kwargs):
        
        if times_presented == None:
            times_presented=1
            
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
        
        while len(queue) > 0 and (num_queries == None or len(feedback) < num_queries):
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

class LinkFeedback(AssignmentFeedback):
    
    
    def completeLink(self, distances, pt, group):
        dist = []
        for g in group:
            dist.append(distances[g])
        
        return max(dist)
        
    #Distances should be the pairwise distances between the representatives
    def generateFeedback(self, distances, query_size=9, times_presented=2, num_queries=None, **kwargs):
        
        if times_presented == None:
            times_presented = 2
            
        #can only perform matching if query_size > 1
        if(query_size == 1):
            super(LinkFeedback, self).generateFeedback(**kwargs)
            return

        #include index to retrieve the actual point after sorting
        rep_distances = map(lambda d: zip(d, range(len(d))) , distances )


        feedback = []
    
        #How often a point has been presented
        presented = [0] * len(distances)
        
        focusPoints = []
        focus = random.choice(range(len(presented)))
        
        candidates = set([focus])
        
        while(num_queries == None or self.num_queries < num_queries):
            
            dist = sorted(rep_distances[focus][:])
            
            #Try to present what has not been presented too many times
            toPresent = filter(lambda p: presented[p[1]] < times_presented,dist)
            
            #print "Focus:", focus
            
            #ensure we dont ask for too many points
            size = query_size
            if (len(toPresent) < size):
                #print "Not enough unPresented Points"
                #if we need more points, take it from the presented points
                _presented = filter(lambda p: presented[p[1]] >= times_presented,dist)
                #print "unpresented:", map(lambda d: d[1], toPresent[:size])
                
                diff = size - len(toPresent)
                
                #Ensure we have enough points to present
                if(len(_presented) < diff):
                    diff = len(_presented)
                
                
                #print "Already presented:", map(lambda d: d[1], _presented[:diff])
                
                toPresent += _presented[:diff]
                size = len(toPresent)
            
            #translate point index to points
            pt_idx = map(lambda d: d[1], toPresent[:size])
            pts = map(lambda x: self.representatives[x], pt_idx)
            
            #print "pt_idx:", pt_idx
            
            #Query
            q = self.query(pts, **kwargs) 
            feedback.append(map(lambda c: map(lambda x: pt_idx[x], c), q)) #translate pt indexes to the indexes of the representatives
        
            #Count the points as presented
            for p in pt_idx:
                presented[p] += 1
                candidates.add(p)
            
            #print "presented:", presented
            #print "Times_Presented:", times_presented
            #update candidates
            update = filter(lambda p: presented[p] < times_presented, candidates)
            
            candidates = set(update)
            #print "Candidates:", candidates
            
            focusPoints.append(focus)    
            
            #print "FocusPoints:", focusPoints
            
            if(len(update) == 0):
                left = filter(lambda p: presented[p] < times_presented, range(len(presented)))
                if len(left) == 0:
                    break
                
                focus = random.choice(left)
                continue
            
            #find new focus
            linkDist = sorted(map(lambda p: (self.completeLink(distances[p],p, focusPoints),p), candidates))
            focus = linkDist[-1][1]
        
            '''
            print "LinkDist:", linkDist
            print "NextFocus:", focus
        
            print
            print
            '''
        self.feedback = feedback
        self.printFeedback(feedback)
        print 
        print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
        
        left = filter(lambda p: presented[p] < times_presented, range(len(presented)))
        
        if len(left) != 0:
            print "Missed Points:", left
            
        print
        
        
            

class RandomMatchingFeedback(AssignmentFeedback):
    
    def generateFeedback(self, query_size=9, num_queries=15, **kwargs):
        if(query_size == 1):
            super(MatchingFeedback, self).generateFeedback(**kwargs)
            return
        
        feedback = []
        
        for i in range(num_queries):
            pt_idx = random.sample(range(len(self.representatives)), query_size)
            pts = map(lambda x: self.representatives[x], pt_idx)
            
            q = self.query(pts, **kwargs)
            feedback.append(map(lambda c: map(lambda x: pt_idx[x], c), q)) #translate pt indexes to the indexes of the representatives
 
        self.feedback = feedback
        self.printFeedback(feedback)
        
        print
        print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
        
class ClosestPointFeedback(MinimumDistanceFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):    
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        super(ClosestPointFeedback,self).generateFeedback(distances, **kwargs)
       
    #distances should be the pairwise distances between the reps

class FarthestLinkFeedback(LinkFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):    
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        super(FarthestLinkFeedback,self).generateFeedback(distances, **kwargs)    


class MinimumSpanningTreeFeedback(MinimumDistanceFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def generateFeedback(self, **kwargs):
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        mst = csgraph.minimum_spanning_tree(distances)
        distances = csgraph.shortest_path(mst,method="D", directed=False)
        
        super(MinimumSpanningTreeFeedback, self). generateFeedback(distances, **kwargs)
        

class DistanceFeedback(MinimumDistanceFeedback):
    
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
    
    def mergeFeedback(self, feedback):
        flattened = []
        for f in feedback:
            flattened += f
        
        
        #print "flattened Feedback:"
        #print "\t", feedback
        
        sets = map(set, flattened)
        
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
            f = list(s)
            feedback.append(f)
            
        return feedback
    
    def mergeSubclusters(self, **kwargs):
        self.final = []
        
        feedback = self.mergeFeedback(self.feedback)
        
        for f in feedback:
            cluster = []
            for i in f:
                cluster += self.subclusters[i]
            self.final.append(cluster)
            
        flattened = [x for i in feedback for x in i]
        r = range(len(self.representatives))
        
        for i in r:
            if i not in flattened:
                self.final.append(self.subclusters[i])
        
        print "Merged Feedback:"
        print "\t", sorted(map(sorted,feedback))
        print 

class HRMFMerge(CentroidINCREMENT,MergeSubclusters):
    
    def mergeSubclusters(self, **kwargs):
        M = []
        C = []
        
        for f in self.feedback:
            for i,x in enumerate(f):
                for j,y in enumerate(f):
                    #only handle symmetry
                    if j < i:
                        continue
    
                    if i == j:
                        #Must Links
                        for s,a in enumerate(x):
                            for t,b in enumerate(y):
                                if s <= t:
                                    continue
                                if([a,b] not in M and [b,a] not in M):
                                    M.append([a,b])
                    else:
                        #Connot Links
                        for s,a in enumerate(x):
                            for t,b in enumerate(y):
                                if([b,a] not in C and [a,b] not in C):
                                    C.append([a,b])
                                
        
        #print "M:", sorted(map(sorted,M))
        #print
        #print "C:", sorted(map(sorted,C))
        #print
        feedback = self.mergeFeedback(self.feedback)
        print "Merged Feedback:"
        print "\t", sorted(map(sorted,feedback))
        print 
        
        hmrf = HMRF.HMRF(self.distance, self.aggregator)
        
        clusters = hmrf.cluster(self.representatives,M,C, feedback)
        
        print
        print "Clustered Representatives:", sorted(map(sorted,clusters))
        print
        self.final = []
        for i in clusters:
            cluster = []
            for x in i:
                cluster += self.subclusters[x]
            self.final.append(cluster)
        
        

   
class HRMFINCREMENT(OpticsSubclustering, CentroidSelector, ClosestPointFeedback, OracleMatching, HRMFMerge):
    pass

class MergeINCREMENT(OpticsSubclustering, CentroidSelector, ClosestPointFeedback, OracleMatching, MergeSubclusters):
    pass

class RandomINCREMENT(OpticsSubclustering, CentroidSelector, FarthestLinkFeedback, OracleMatching, MergeSubclusters):
    pass

class PathINCREMENT(OpticsSubclustering, MedoidSelector, DistanceFeedback, OracleMatching, MergeSubclusters):
    pass



















