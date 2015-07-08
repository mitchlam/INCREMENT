
import utils
import optics
import HMRF

import random
import scipy.sparse.csgraph as csgraph
import numpy as np
import sys

#class Cluster:
    
#    def __init__(self,instances, representative = None): #, label=None):
#        self.instances = instances
#        self.representative = representative
        #self.label = label

class BaseINCREMENT(object):
    #Uses naive implementations of everything

    def __init__(self, clustering, distance=utils.EuclideanDistance, symmetric_distance=True, verbose=True, **kwargs):
        self.clustering = clustering
        self.subclusters = []
        self.representatives = [] #actual points, one for each subcluster. Indexes should be aligned with subcluster
        self.feedback = [] #list of feedback. Should be clusterings of indexes of representative points

        self.final = [] #store final clustering 

        self.distance = distance #function to determine distance between instances can be set for custom domains
        self.symmetric_distance = symmetric_distance #Bool stating whether or not the distance is symmetric
        
        self.num_queries = 0
        
        self.verbose = verbose

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
        if self.verbose:
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

    #performs and subclusters a single cluster of points
    def performOPTICS(self, distance, minPts, display):
        out = optics.OPTICS(distance, minPts)
        sep = optics.separateClusters(out, minPts, display=display)
        
        return out, sep
    
    def breakdown(self, output, separated, indent=""):
        reachability = map(lambda c: map(lambda x: x.reachability, c), output)
        avgs = map(lambda c:sum(c)/len(c), reachability)
        stds = map(lambda c: np.std(c), reachability)
            
        subavgs = []
        substds = []
        for c, sep in enumerate(separated):
            idx = 0
            subavgs.append([])
            substds.append([])
            for i, s in enumerate(sep):
                tmp = reachability[c][idx:idx+len(s)]
                avg = sum(tmp)/len(tmp)
                subavgs[c].append(avg)
                substds[c].append(np.std(tmp))
                idx += len(s)
                
        idx = 0
        
        print
        if indent == "":
            print "Subcluster Breakdown:"
            
        for c, sub in enumerate(subavgs):
            print indent + "\t%d: %f (%d)" % (c, avgs[c], sum(map(len, separated[c])))
            for i, a in enumerate(sub):
                print indent + "\t\t%d: %f -- %f  (%d)" % (idx, a, substds[c][i], len(separated[c][i]))
                idx += 1
            print indent + "\t--> std: %f -- %f" % (stds[c], np.std(substds[c]))
            print
            
        print indent + "\tAvg: %f -- %f " % (sum(avgs)/len(avgs), np.std(avgs))
        print indent + "\tStd: %f -- %f " % (sum(stds)/len(stds), np.std(stds))
        print indent + "Subclusters Formed:", len(self.subclusters)
        print


    # assumes separated in of the form [Partition][subcluster][OPTICS POINT]
    def mapSeparated(self, separated):
        subclusters = []
        
        for c,sep in enumerate(separated):
            ids = map(lambda sc: map(lambda x: x._id, sc), sep)
            lengths = []
            for sub in ids:
                clust = []
                for i in sub:
                    clust.append(self.clustering[c][i])
                lengths.append(len(clust))
                
                subclusters.append(clust)
                
        return subclusters
    
    
    #Performs OPTICS to subcluster the current clustering
    def subcluster(self, minPts=5, display=False, **kwargs):
        
        self.subclusters = []
        
        if self.verbose:
            print "Computing Distance"
            
        distances = map(lambda x:utils.pairwise(x,self.distance, self.symmetric_distance), self.clustering) #N^2 where N is the number of instances per cluster -- SLOW
        
        if self.verbose:
            print "Running OPTICS: minPts = %d" % (minPts)
        
        output, separated = zip(*map(lambda d: self.performOPTICS(d, minPts, display), distances))
        
        self.subclusters = self.mapSeparated(separated)
        
        
        if self.verbose:
            self.breakdown(output,separated)


class RecursiveOPTICS(OpticsSubclustering):
    
    #Return of the format Output, subclusters
    #subclusters: [Subcluster][OPTIC POINT]
    def performOPTICS(self, distance, minPts, display, level = 0, minPtsMin = 3):
        
        #minPts = len(distance)/10
        
        if minPts < minPtsMin:
            minPts = minPtsMin
        
        indent = "\t"*level
        start = len(distance)
        
        curried = lambda d: super(RecursiveOPTICS, self).performOPTICS(d, minPts, display) 
        
        output, subclusters = curried(distance) # Has bug, if there is only a single point, it isnt put in a subcluster
        
        if self.verbose:
            print indent + "{%d} Begin (%d): %d" % (level, start, minPts)
        
        if len(subclusters) == 0:
            if self.verbose:
                print indent + "{%d} End Single" % (level)
            return output, [output[:]]
        
        #Base Case -- Return when there is only a single subcluster
        if len(subclusters) == 1:
            #return output, subclusters # Uncomment to recurse a single subclsuter
        
            if minPts <= minPtsMin or start < 2 or level > 10:
                if self.verbose:
                    print indent + "{%d} Indivisable" % (level)
                    
                return output, subclusters
            else:
                
                #return super(RecursiveOPTICS, self).performOPTICS(distance, minPts/2, display)
                output, subclusters = self.performOPTICS(distance, minPts/2, display, level + 1)
                
                if self.verbose:
                    print indent + "{%d} End Reduce (%d) : %d" % (level, start, minPts)
                    
                return output, subclusters
        
        
        #Intermediate case
        
        #parse out ids
        idxs = map(lambda sub: map(lambda s: s._id, sub), subclusters)
        
        distance = np.array(distance)
        
        dist = []
        #Filter distances
        for sub in idxs:
            dist.append(distance[np.ix_(sub,sub)])
        
        #if minPts > 2:
        #    minPts = minPts/2
        
        #Recursive step
        try:
            out, sep = zip(*map(lambda d: self.performOPTICS(d, minPts, display, level+1), dist))
        except ValueError:
            print "Value Error"
            print "distance", len(distance)
            print "Dists:", map(len,dist)
            print "Minpts:", minPts
            print "output:", output
            print "subclusters:", subclusters
            
            sys.exit()
            
        
        #translate sep indexes back
        sep = list(sep)
    
        #if self.verbose:
        #    self.breakdown(out, sep, indent=indent)
        #print "idxs", idxs
    
        #print sep
    
        result = []
        for i, p in enumerate(sep):
            for sub in p:
                for s in sub:
                    s._id = idxs[i][s._id]
                result.append(sub)
        
                
        #print "result", result
        
        out = [i for x in out for i in x]
        
        lens = map(len, result)
        end = sum(lens)
        
        

        if self.verbose:
            print indent + "{%d} End (%d)" % (level, end)
        
        if start != end:
            print indent + "Points Given:", start
            print indent + "Points Returning:", end
            
            print indent + "Error: Missing Points"
            
            indent += "\t"
            
            print indent + "idsx", idxs
            print indent + "sep", sep
            print indent + "dists:", map(len,dist)
            print indent + "results:", lens
        
        
        return out, result
    

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
            
        if self.verbose:
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
        if self.verbose:
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
        
        if self.verbose:
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
            super(MinimumDistanceFeedback, self).generateFeedback(**kwargs)
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
        
        if self.verbose:
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
                left = filter(lambda p: presented[p] < times_presented/2, range(len(presented)))
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
        
        if self.verbose:
            self.printFeedback(feedback)
            print 
            print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
        
        left = filter(lambda p: presented[p] < times_presented, range(len(presented)))
        
        if self.verbose:
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
        
        if self.verbose:
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
        
        if self.verbose:
            print "Merged Feedback:"
            for i, f in enumerate(sorted(map(sorted,feedback))):
                print "\t", i, ":", f
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
        
        if self.verbose:
            print "Merged Feedback:"
            print "\t", sorted(map(sorted,feedback))
            print 
        
        hmrf = HMRF.HMRF(self.distance, self.aggregator)
        
        clusters = hmrf.cluster(self.representatives,M,C, feedback)
        
        if self.verbose:
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

class MergeINCREMENT(RecursiveOPTICS, CentroidSelector, FarthestLinkFeedback, OracleMatching, MergeSubclusters):
    pass

class OtherINCREMENT(RecursiveOPTICS, CentroidSelector, FarthestLinkFeedback, OracleMatching, MergeSubclusters):
    pass

class PathINCREMENT(RecursiveOPTICS, MedoidSelector, DistanceFeedback, OracleMatching, MergeSubclusters):
    pass

class AssignmentINCREMENT(RecursiveOPTICS, CentroidSelector, AssignmentFeedback, OracleMatching, MergeSubclusters):
    pass



















