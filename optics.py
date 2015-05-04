
import random
import heapq
import itertools
import numpy as np
import matplotlib.pyplot as pyplot

from image.signalutils import blur_bilateral

class heap:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
        #print "Init"
        
    def push(self, task, priority =0):
        'Add a new task or update the priority of an existing task'
        #print "Pushing"
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)
        
    def remove(self,task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED
        
    
    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
    
    def isEmpty(self):
        return len(self.entry_finder) == 0

class dataPoint:
    def __init__(self, _id):
        self._id = _id
        self.processed = False
        self.reachability = -1
        self.core = -1
        
    def __eq__(self, other):
        return self._id == other._id
    
    def core_distance(self, distances, minPts):
        if(self.core == -1):
            d = sorted(distances)
            self.core = d[ minPts ] if (minPts < len(d)) else -1
        
        return self.core
    
    def __hash__(self):
        return hash(self._id)


def OPTICS_update(distances, N, p, seeds, minPts):
    coredist = p.core_distance(distances, minPts)
    for o in N:
        if(not o.processed):
            new_reach = max(coredist, distances[o._id])
            if(o.reachability == -1):
                o.reachability = new_reach
                seeds.push(o, new_reach)
            elif (new_reach < o.reachability):
                o.reachability = new_reach
                seeds.push(o, new_reach)

                

def OPTICS(distances, minPts, cluster=None):
    
    points = map(dataPoint, range(len(distances)))
    
    #random.shuffle(points)
    output = []
    
    for p in points:
        if p.processed:
            continue
        
        N = points[:]
        N.remove(p)
        p.processed = True
        output.append(p)
        
        if(p.core_distance != -1):
            seeds = heap()
            OPTICS_update(distances[p._id], N, p, seeds, minPts)
            while (not seeds.isEmpty()):
                q = seeds.pop()
                q.processed = True
                output.append(q)
                if(q.core_distance(distances[q._id], minPts) != -1):
                    OPTICS_update(distances[q._id], N, q, seeds, minPts)
        
    output[0].reachability = 0
    return output

    #return chooseMax(output)
'''
def displayTruePlot(output, cluster, minPts, display=9):
    labels = [cluster.center] + map(lambda x: x.label, cluster.members)
    ids = map(lambda x: x._id, output)
    reach = map(lambda x: x.reachability, output)
    reach[0] = 0
    
    colors = ['r^', 'g^', 'm^', 'c^','b^', 'y^', 'k^']
    color = {}
    c = 0
    
    for lbl in labels:
        if lbl not in color:
            color[lbl] = colors[c % len(colors)]
            c += 1
    
    
    
    fig = pyplot.figure(1)
    pyplot.clf()
    
    pyplot.title("True Values")
    
    pyplot.plot(reach, "k")
    
    l = zip(ids,reach)
    x = 1
    
    centerColor = 'k^' if (cluster.label == None) else color[cluster.label]
    pyplot.plot(0,0, centerColor)
    
    for i, r in l[1:]:
        pyplot.plot(x, r, color[cluster.members[i-1].label])
        x+=1
    
    minimum = selectMinimum(output,minPts)
    
    for m in minimum[:display-1]:
        pyplot.plot(output.index(m), m.reachability, 'kD')
    
    fig.show()
 '''   
    
def selectMinimum(output, minPts):
    
    reachability = np.array(map(lambda x: x.reachability, output))
    fEdge = np.array((0, 1, -1))
    
    reach = reachability
    kernel = fEdge
    forward = np.correlate(reach, kernel, mode='same')
    
    minimum = []
    assert(len(forward) == len(output))
    
    dist = minPts*2
    for i,o in enumerate(output):
        start = i - dist
        end = i
        
        if (start < 0):
            start = 0 

        

        threshold = 0.0
        if (((i != 0) and (i != len(output))) and ((forward[i] < threshold) and (forward[i-1] >= threshold))):
            area = reachability[start:end]
            std = max(area)
            minimum.append((std - reachability[i], i))

    minimum.sort()
    
    minimum.reverse()
    
    ret = map(lambda i: output[i[1]], minimum)
    
    return ret

def separateClusters(output, minPts, display=False):
    output[0].reachability = 0
    reachability = np.array(map(lambda x: x.reachability, output))
    
    if len(output) < minPts:
        #print "Sub-clusters: 1 [%d]" %(len(output))
        return [output]
    
    fEdge = np.array((0, 1, -1))
    avg = np.array([1/float(minPts)] * minPts)


    space_sig = 1
    value_sig = 0.001
    
    reach = reachability
    #reach = blur_bilateral(reachability, minPts, space_sig, value_sig)
    
    
    kernel = fEdge
    #kernel = np.correlate(avg, fEdge, mode='full')
    
    forward = np.correlate(reach, kernel, mode='same')

    std = np.std(forward)
    
    fig = pyplot.figure(3)
    if display:
        pyplot.clf()
        pyplot.title("Derivative")
    
        pyplot.plot(forward, color='g')   
    

    clusters = []
    current = []
    dist = minPts*2
    for i, o in enumerate(output):
        
        start = i - dist
        end = i + dist
        
        if (start < 0):
            start = 0 
        if (end > (len(reach)-1)):
            end = len(reach) -1
        
        area = reach[start:end]
        
        var = np.std(area)
        
        threshold = np.sqrt(var * std)
        
        if display:
            pyplot.plot(i,std, 'c.')
            pyplot.plot(i,threshold, 'b.')
            
        if (((i == 0) and (i != len(output)-1)) or ((forward[i] >= threshold) and (forward[i-1] < threshold))):
            current = []
            clusters.append(current)
            
    
        current.append(o)
          
    if display:
        fig.show()

    while(len(clusters) > 1 and len(clusters[0]) < minPts):
        if (len(clusters[0]) < minPts and len(clusters) > 1):
            clusters[1] = clusters[0] + clusters[1]
            del clusters[0]
    
    enum = reversed(zip(range(len(clusters)), clusters))
    
    for i,cl in enum:
        if (i ==0):
            continue
        
        if(len(cl) < minPts):
            clusters[i-1] += cl
            del clusters[i]



    fig = pyplot.figure(2)
    if display:
        pyplot.clf()
    
        pyplot.title("Clustering")
    
        pyplot.plot(reachability, "c")
        pyplot.plot(reach, "k")
    
        colors =['b^','r^', 'g^', 'm^', 'c^']
        c = 0

        i = 0
        for cl in clusters:
            c = (c+1) % len(colors)
            for o in cl:
                pyplot.plot(i,o.reachability, colors[c])
                i += 1
        fig.show()
    
    #print "Sub-Clusters:", len(clusters), map(lambda x: len(x), clusters)
    
    
    #print "variance:", np.var(reach),",", np.var(forward)
    
    if display:
        raw_input()
    
    return clusters

def chooseMax(output):
    reach =  map(lambda x: x.reachability, output)

    reach[0] = 0.0

    s = zip(reach, range(len(reach)))
    s.sort()
        
    m = s[-8:]
    
    reps, idx = zip(*m)

    fig = pyplot.figure(2)
    pyplot.clf()
    
    
    pyplot.plot(reach, color='b')   
    pyplot.plot(idx, reps, "g^")
    fig.show()
    
    idx = [0] + sorted(idx)
    
    return map(lambda i: output[i]._id, idx)

##