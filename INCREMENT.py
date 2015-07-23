
import os
import incUtils as utils
import incOptics as optics
import HMRF
import caffe

import random
import scipy.sparse.csgraph as csgraph
import numpy as np
from sklearn.cluster import KMeans

import sys

_VERBOSE_SILENT = 3
_VERBOSE_DEFAULT = 2
_VERBOSE_INFO = 1
_VERBOSE_DEBUG = 0

#class Cluster:
    
#    def __init__(self,instances, representative = None): #, label=None):
#        self.instances = instances
#        self.representative = representative
        #self.label = label

class BaseINCREMENT(object):
    #Uses naive implementations of everything

    def __init__(self, clustering, distance=utils.EuclideanDistance, as_array=None, symmetric_distance=True, verbose=True, **kwargs):
        self.clustering = clustering
        self.subclusters = []
        self.representatives = [] #actual points, one for each subcluster. Indexes should be aligned with subcluster
        self.feedback = [] #list of feedback. Should be clusterings of indexes of representative points

        self.final = [] #store final clustering 

        self.distance = distance #function to determine distance between instances can be set for custom domains
        self.symmetric_distance = symmetric_distance #Bool stating whether or not the distance is symmetric
        self.as_array = as_array #Used to convert data to a np array
        
        self.num_queries = 0
        
        self.verbose = verbose

    def setInstanceDistance(func):
        self.distance = func
        
    def subcluster(self, **kwargs):
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Subclustering:"
        result = self._subcluster(**kwargs)
        
        print "Subclusters Formed:", len(self.subclusters)
        print
        
        return result

    def selectRepresentatives(self, **kwargs):
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Selecting Representatives:"
        result = self._selectRepresentatives(**kwargs)
        
        print
        
        return result
        
    def generateFeedback(self, **kwargs):
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Generating Feedback:"
        result =  self._generateFeedback(**kwargs)
        
        print "Number of Queries:", self.num_queries
        print
        
        return result
        
    def query(self, *args, **kwargs):
        return self._query(*args, **kwargs)
        
    def mergeSubclusters(self, **kwargs):
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Merging Subclusters:"
        
        result = self._mergeSubclusters(**kwargs)
        
        print 
        
        return result
        
    def _subcluster(self, **kwargs):
        self.subclusters = self.clustering

    def _selectRepresentatives(self, **kwargs):
        self.representatives = map(lambda x: x[0],self.subclusters)

    def _generateFeedback(self, **kwargs):
        pass

    #Actually presents a set of points to the user, and returns the feedback
    #returns a list of clustered indexes into pts 
    def _query(self, pts, **kwargs):
        self.num_queries += 1
        return [[i] for i in range(len(pts))]

    def _mergeSubclusters(self, **kwargs):
        self.final = self.subclusters

    def run(self, **kwargs):
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Running INCREMENT:"
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
        #print indent + "Subclusters Formed:", len(self.subclusters)
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
    def _subcluster(self, minPts=5, display=False, **kwargs):
        
        self.subclusters = []
        
        if self.verbose <= _VERBOSE_INFO:
            print "Computing Distance"
            
        distances = map(lambda x:utils.pairwise(x,self.distance, self.symmetric_distance), self.clustering) #N^2 where N is the number of instances per cluster -- SLOW
        
        if self.verbose <= _VERBOSE_INFO:
            print "Running OPTICS: minPts = %d" % (minPts)
        
        output, separated = zip(*map(lambda d: self.performOPTICS(d, minPts, display), distances))
        
        self.subclusters = self.mapSeparated(separated)
        
        
        if self.verbose <= _VERBOSE_INFO:
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
        
        if self.verbose <= _VERBOSE_DEBUG:
            print indent + "{%d} Begin (%d): %d" % (level, start, minPts)
        
        if len(subclusters) == 0:
            if self.verbose <= _VERBOSE_DEBUG:
                print indent + "{%d} End Single" % (level)
            return output, [output[:]]
        
        #Base Case -- Return when there is only a single subcluster
        if len(subclusters) == 1:
            #return output, subclusters # Uncomment to recurse a single subclsuter
        
            if minPts <= minPtsMin or start < 2 or level > 10:
                if self.verbose <= _VERBOSE_DEBUG:
                    print indent + "{%d} Indivisable" % (level)
                    
                return output, subclusters
            else:
                
                #return super(RecursiveOPTICS, self).performOPTICS(distance, minPts/2, display)
                output, subclusters = self.performOPTICS(distance, minPts/2, display, level + 1)
                
                if self.verbose <= _VERBOSE_DEBUG:
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
    
        #if self.verbose <= _VERBOSE_INFO:
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
        
        

        if self.verbose <= _VERBOSE_DEBUG:
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
    
    def _selectRepresentatives(self, **kwargs):
        self.representatives = []
        
        distances = map(lambda sc: utils.pairwise(sc, self.distance, self.symmetric_distance), self.subclusters)
        
        reps = []
        
        for i, dist in enumerate(distances):
            sums = map(sum, dist)
            m = utils.arg_min(sums)
            reps.append(m)
            self.representatives.append(self.subclusters[i][m])
            
        if self.verbose <= _VERBOSE_INFO:
            print "Representatives:"
            print reps
            print
    
class RandomSelector(BaseINCREMENT):
    
    def selectRepresentattives(self, **kwargs):
        self.representatives = []
        
        for sub in self.subclusters:
            self.representatives.appen(random.choice(sub))
        
        if self.verbose <= _VERBOSE_INFO:
            print "Representatives:"
            print reps
            print

class CentroidSelector(CentroidINCREMENT):
    
    def _selectRepresentatives(self, **kwargs):
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
    def _generateFeedback(self, **kwargs):
        if self.verbose <= _VERBOSE_INFO:
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
        
        if self.verbose <= _VERBOSE_INFO:
            self.printFeedback(feedback)
            print
            print "Number of Assignement Queries: %d" % (self.num_queries)
            print
   
class MinimumDistanceFeedback(AssignmentFeedback):
    
    #Distances should be the pairwise distances between the representatives
    def _generateFeedback(self, distances, query_size=9, times_presented=1, num_queries=None, **kwargs):
        
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
        
        if self.verbose <= _VERBOSE_INFO:
            self.printFeedback(feedback)
            print 
            print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
            print
            
            
class FarthestFirstFeedback(AssignmentFeedback):
    
    def __init__(self, *args, **kwargs):
        super(FarthestFirstFeedback, self).__init__(*args, **kwargs)
        self.overlap=True
    
    def singleLink(self, distances, group):
        dist = []
        for g in group:
            dist.append(distances[g])
        
        return min(dist)
    
    def findMax(self, presented, unpresented, paired_distances):
        distances = map(lambda p: (self.singleLink(paired_distances[p], presented), p),unpresented)
        
        distances.sort(reverse=True)
        
        return distances[0][1], distances[0][0][1]
    
    def presentQuery(self, pt_idx, **kwargs):
        pts = map(lambda x: self.representatives[x], pt_idx)
            
        response = self.query(pts, **kwargs)
        return map(lambda c: map(lambda x: pt_idx[x], c), response) #translate pt indexes to the indexes of the representatives
    
    
    def postProcess(self, feedback):
        return feedback
    
    
    def _generateFeedback(self, query_size=9, num_queries=None, **kwargs):        
        if(query_size == 1):
            super(FarthestFirstFeedback, self).generateFeedback(**kwargs)
            return
        
        if (self.verbose <= _VERBOSE_INFO):
            print
            print "Farthest First"
        
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        rep_distances = map(lambda d: zip(d, range(len(d))) , distances)
        
        feedback = []
        
        toPresent = set()
        presented = set()
        unpresented = range(len(self.representatives))
        
        q = 0
        
        while (len(unpresented) > 0 and (num_queries == None) or (q < num_queries)):
            if len(presented) == 0:
                pt = random.choice(unpresented)
                unpresented.remove(pt)
                
                toPresent.add(pt)
                presented.add(pt)
            else:
                pt, closest = self.findMax(presented, unpresented, rep_distances)
                
                unpresented.remove(pt)
                presented.add(pt)
                toPresent.add(pt)
                if self.overlap and len(toPresent) < query_size:
                    toPresent.add(closest)
        
            if len(toPresent) % query_size == 0:
                feedback.append(self.presentQuery(list(toPresent), **kwargs))
                q += 1
                toPresent = set()
        
        if (len(toPresent) > 0 and (num_queries == None) or (q < num_queries)):
            feedback.append(self.presentQuery(list(toPresent), **kwargs))
            q += 1
            toPresent = set()
        
        self.num_queries = q
        self.feedback = self.postProcess(feedback)
        
        if self.verbose <= _VERBOSE_INFO:
            self.printFeedback(feedback)
            print 
            print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
            print
            
class FarthestLabelFeedback(FarthestFirstFeedback):
    
    def __init__(self, *args, **kwargs):
        super(FarthestLabelFeedback, self).__init__(*args, **kwargs)
        self.overlap=False
        
    def presentQuery(self, pt_idx, **kwargs):
        pts = map(lambda x: self.representatives[x], pt_idx)
            
        lbls = {}
        for i,pt in enumerate(pts):
            response = self.query([pt], **kwargs)
            if response not in lbls:
                lbls[response] = []
            lbls[response].append(pt_idx[i])
        
        return lbls
    
    def postProcess(self, feedback):
        lbls = {}
        
        for f in feedback:
            for k,v in f.items():
                if k not in lbls:
                    lbls[k] = set()
                
                lbls[k].update(v)
        
        response = []
        
        for k,s in lbls.items():
            response.append(list(s))
        
        return [response]
    
    
class LinkFeedback(AssignmentFeedback):
    
    
    def completeLink(self, distances, pt, group):
        dist = []
        for g in group:
            dist.append(distances[g])
        
        return max(dist)
        
    #Distances should be the pairwise distances between the representatives
    def _generateFeedback(self, distances, query_size=9, times_presented=2, num_queries=None, **kwargs):
        
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
        
        if self.verbose <= _VERBOSE_INFO:
            self.printFeedback(feedback)
            print 
            print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
        
        left = filter(lambda p: presented[p] < times_presented, range(len(presented)))
        
        if self.verbose <= _VERBOSE_INFO:
            if len(left) != 0:
                print "Missed Points:", left
            
            print
        
        
            

class RandomMatchingFeedback(AssignmentFeedback):
    
    def _generateFeedback(self, query_size=9, num_queries=15, **kwargs):
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
        
        if self.verbose <= _VERBOSE_INFO:
            self.printFeedback(feedback)
        
            print
            print "Number of Queries: %d of size %d" % (self.num_queries, query_size)
        
class ClosestPointFeedback(MinimumDistanceFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def _generateFeedback(self, **kwargs):    
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        super(ClosestPointFeedback,self).generateFeedback(distances, **kwargs)
       
    #distances should be the pairwise distances between the reps

class FarthestLinkFeedback(LinkFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def _generateFeedback(self, **kwargs):    
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        super(FarthestLinkFeedback,self).generateFeedback(distances, **kwargs)    


class MinimumSpanningTreeFeedback(MinimumDistanceFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def _generateFeedback(self, **kwargs):
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)
        
        mst = csgraph.minimum_spanning_tree(distances)
        distances = csgraph.shortest_path(mst,method="D", directed=False)
        
        super(MinimumSpanningTreeFeedback, self). generateFeedback(distances, **kwargs)
        

class DistanceFeedback(MinimumDistanceFeedback):
    
    #Organizes and manages the presentation of representatives and user feedback
    def _generateFeedback(self, **kwargs):
        distances = utils.pairwise(self.representatives, self.distance, self.symmetric_distance)

        distances = csgraph.shortest_path(distances,method="D", directed=self.symmetric_distance)
        
        super(MinimumDistanceFeedback, self). generateFeedback(distances, **kwargs)
        
################################# Query #################################################

#If only a single point is presented, return it's label 
class OracleMatching(BaseINCREMENT):
    
    #Cheats and looks at target. Simulates a perfect user.
    #labeler is a function that accepts an instance and returns its label
    def _query(self, pts,  labeler=None, **kwargs):
        
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
    
    def _mergeSubclusters(self, **kwargs):
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
        
        if self.verbose <= _VERBOSE_INFO:
            print "Merged Feedback:"
            for i, f in enumerate(sorted(map(sorted,feedback))):
                print "\t", i, ":", f
            print 

class HRMFMerge(CentroidINCREMENT,MergeSubclusters):
    
    def _mergeSubclusters(self, **kwargs):
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
        
        if self.verbose <= _VERBOSE_INFO:
            print "Merged Feedback:"
            print "\t", sorted(map(sorted,feedback))
            print 
        
        hmrf = HMRF.HMRF(self.distance, self.aggregator)
        
        clusters = hmrf.cluster(self.representatives,M,C, feedback)
        
        if self.verbose <= _VERBOSE_INFO:
            print
            print "Clustered Representatives:", sorted(map(sorted,clusters))
            print
        
        
        self.final = []
        for i in clusters:
            cluster = []
            for x in i:
                cluster += self.subclusters[x]
            self.final.append(cluster)
        
class SiameseMerging (MergeSubclusters):
    
    def __init__(self, *args, **kwargs):
        super(SiameseMerging, self).__init__(*args, **kwargs)
        self.batch_size = 10
        self.output_size = 100
    
    def findConstraints(self, merged):
        feedback = self.feedback
    
        cannotLink = []
        
        for sc in merged:
            c = set()
            for rep in sc:
                for query in feedback:
                    tmp = set()
                    found = False
                    for group in query:
                        if rep not in group:
                            tmp.update(group)
                        else:
                            found = True
                    
                           
                    '''
                    print "Point:", rep
                    print "Found:", found
                    print "Query:", query
                    print "Tmp:", tmp
                    print
                    '''
                    if found:
                        c.update(tmp)
            
            cannotLink.append(c)
    
        constraints = []
        for constr in cannotLink:
            tmp = set()
            for pt in constr:
                for c,sc in enumerate(merged):
                    if pt in sc:
                        tmp.add(c)
                        break
            constraints.append(tmp)
        
        if self.verbose <= _VERBOSE_INFO:
            print "Merged Feedback:"
            print "\t", merged, len(merged)
            print
            
            print "Connot Link Subcluster Constraints:"
            print "\t", constraints, len(constraints)
        
        return constraints
        
    def getTrainData(self, feedback, **kwargs):
        train_data = []
        labels = []
        
        for l,sc in enumerate(feedback):
            for idx in sc:
                d = np.array(self.as_array(self.representatives[idx]), dtype=np.float32)
                train_data.append(d)
                labels.append(l)
    
        return np.array(train_data), np.array(labels), self.findConstraints(feedback)
    
    def getData(self, **kwargs):
        data = []
        targets = []
        lookUp = []
        for i,r in enumerate(self.representatives):
            d = np.array(self.as_array(r), dtype=np.float32)
            if('labeler' in kwargs):
                t = kwargs['labeler'](r)
                #print t
                targets.append(t)
                
            data.append(d)
            lookUp.append(self.subclusters[i])
            
        data = np.array(data)
        targets = np.array(utils.enumeratation(targets))
        
        return data, targets, lookUp
    
    
    def _mergeSubclusters(self, **kwargs):
        feedback = self.mergeFeedback(self.feedback)
        K = len(feedback)
        
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Generating Data"
        train_data, labels, constraints = self.getTrainData(feedback, **kwargs)
        data, targets, reverseIndex = self.getData(**kwargs)
        
            
        net = self.siameseNet("_train", "_deploy", train_data, labels, constraints, data)
        
        data = data[:,np.newaxis,:,np.newaxis] 
        train_data = train_data[:,np.newaxis,:,np.newaxis]
        
        feat = np.copy(net.forward()['feat'])
        #train_feat = np.copy(utils.feedData(net, train_data)['feat'])
        
        if self.verbose <= _VERBOSE_INFO:
            print "data", data.shape
            print "targets:", targets.shape
        
            print "K:", K
        
        #super(SiameseMerging, self).mergeSubclusters(**kwargs)
        

        #print "Feat:", feat.shape
        #print feat
        
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Reclustering"
            
        kmeans = KMeans(n_clusters=K, precompute_distances=True, n_jobs=-1)
        
        kmeans.fit(feat)
        lbls = kmeans.labels_
        
        k = len(set(lbls))
    
        clusters = []
        
        for i in range(k):
            clusters.append([])
            
        for i,x in enumerate(feat):
            l = lbls[i]
            clusters[l] += reverseIndex[i]
        
        self.final = clusters
        
        if self.verbose <= _VERBOSE_INFO and self.output_size <= 3:
            utils.clearPlots()
            #utils.displayPlot(train_feat, labels, title="Train Data", block=False)
            utils.displayPlot(feat, lbls, title="Final Clustering", block=False)
            utils.displayPlot(feat, targets, title="Targets")
        

    def siameseNet(self, trainName, deployName, train_data, labels, constraints, data):
        batch_size = self.batch_size
        outSize = self.output_size
            
        
        if self.verbose <= _VERBOSE_DEFAULT:
            print "Creating Pairs"
            
        pair_data, sims = self.createPairs(train_data, labels, constraints, batch_size)
        
        data = data[:, np.newaxis, :, np.newaxis]
        
        if self.verbose <= _VERBOSE_INFO:
            print "Train_data:", train_data.shape
            print "Data:", data.shape
            print "pairs:", pair_data.shape
            print "sims:", sims.shape
        
        if self.verbose <= _VERBOSE_INFO:
            print "Creating files for:", trainName
        
        utils.writeH5(trainName, pair_data=pair_data, sims=sims)
        
        if self.verbose <= _VERBOSE_INFO:
            print "Creating files for:", deployName
            
        utils.writeH5(deployName, data=data)
        
        TRAIN_MODEL = "_TRAIN_NET.prototxt"
        DEPLOY_MODEL = "_DEPLOY_NET.prototxt"
        SOLVER_FILE = "solver.prototxt"
        
        size = data.shape[2]
        
        #Write training model prototxt
        with open(TRAIN_MODEL, "w") as f:
            f.write('name: "train"\n')
            f.write(str(utils.createTrainSiamese(source = trainName + ".txt", batch_size=batch_size, vector_size = size, output_size=outSize)))
    
        with open(DEPLOY_MODEL, "w") as f:
            f.write('name: "deploy"\n')
            f.write(str(utils.createDeploySiamese(source = deployName + ".txt", batch_size=data.shape[0], output_size=outSize))) # leave batch_size =1 here. Causes weird errors otherwise

        if self.verbose <= _VERBOSE_DEFAULT:
            print "Training siamese network"
            
        solver = caffe.SGDSolver(SOLVER_FILE)
        #solver.net.set_input_arrays(pair_data,sims)
        solver.solve()
        
        net = caffe.Net(DEPLOY_MODEL, "_iter_" + str(solver.iter) + ".caffemodel", caffe.TEST)
        
        return net

    def createPairs(self, data, targets, constraints, batch_size):
        return utils.generatePairs(data, targets, constraints, batch_size=batch_size)

class SiameseTrainAll(SiameseMerging):
    
    def getTrainData(self,feedback, **kwargs):
        train_data = []
        labels = [] 
        
        for l,sc in enumerate(feedback):
            for idx in sc:
                for pt in self.subclusters[idx]:
                    d = np.array(self.as_array(pt), dtype=np.float32)
                    train_data.append(d)
                    labels.append(l)
    
        return np.array(train_data), np.array(labels), self.findConstraints(feedback)
    
    
    def createPairs(self, data, targets, constraints, batch_size, num_pairs=500000):
        n = data.shape[0]
        
        if n*(n-1)/2 > num_pairs:
            return utils.generateRandomPairs(data, targets, num_pairs, constraints, batch_size=batch_size)
        
        return utils.generatePairs(data, targets, constraints, batch_size=batch_size)
         
class SiameseTestAll(SiameseMerging):
    
    def getData(self, **kwargs):
        data = []
        targets = [] 
        lookUp = []
        
        for sc in self.subclusters:
            for pt in sc:
                d = np.array(self.as_array(pt), dtype=np.float32)
                data.append(d)
                if('labeler' in kwargs):
                    t = kwargs['labeler'](pt)
                    #print t
                    targets.append(t)
                lookUp.append([pt])
    
        return np.array(data), np.array(utils.enumeratation(targets)), lookUp
    
class Siamese(SiameseTrainAll, SiameseTestAll):
    pass

class HRMFINCREMENT(OpticsSubclustering, CentroidSelector, ClosestPointFeedback, OracleMatching, HRMFMerge):
    pass

class MergeINCREMENT(RecursiveOPTICS, MedoidSelector, FarthestLabelFeedback, OracleMatching, Siamese):
    pass

class OtherINCREMENT(RecursiveOPTICS, CentroidSelector, FarthestLinkFeedback, OracleMatching, MergeSubclusters):
    pass

class PathINCREMENT(RecursiveOPTICS, MedoidSelector, DistanceFeedback, OracleMatching, MergeSubclusters):
    pass

class AssignmentINCREMENT(RecursiveOPTICS, CentroidSelector, AssignmentFeedback, OracleMatching, MergeSubclusters):
    pass



















