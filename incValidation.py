import incUtils as utils
import numpy as np
import sklearn.metrics

def contingency(clustering):
    labels = [-1]*len(clustering)

    cont = {}

    for i,cluster in enumerate(clustering):
        trueVals = map(lambda x: x.label, cluster)
        label = utils.mode(trueVals)
        
        for l in trueVals:
            if l not in cont:
                cont[l] = [0]*len(labels)
            
            cont[l][i] += 1

    final = []
    for key,value in cont.items():
        final.append(value)

    #print "Final Error : %d of %d" % (error, total)
    return final


#computes and returns the homogeneity
def homogeneity(cont,eps=10e-10):
    if(len(cont) == 1):
        return 1.0
    
    x = 0.0
    label_counts = map(sum,cont)
    N = float(sum(label_counts))
    cluster_sizes = [0.0]*len(cont[0])
    num_labels = len(cont)
    
    #H(C|K)
    for label, row in enumerate(cont):
        for cluster, val in enumerate(row):
                cluster_sizes[cluster] += val
    
    for label, row in enumerate(cont):
        for cluster, val in enumerate(row):
            one = val/float(N)
            two = val/ cluster_sizes[cluster]
            if (two + eps != 0.0):
                x += one * np.log(two+eps)
    
    x *= -1
    
    y = 0.0
    
    #H(C)
    for count in label_counts:
        num = eps + count / N
        if(num != 0.0):
            y += (num) * np.log( num ) #different from paper
    
    y *= -1
    
    return 1.0 - (x / y)

#computes and returns the completeness score
def completeness(cont, eps=10e-10):
    if(len(cont[0]) == 1):
        return 1.0
    
    x = 0.0
    label_counts = map(sum,cont)
    N = sum(label_counts)
    cluster_sizes = [0.0]*len(cont[0])
    num_labels = len(cont)
    
    
    for label, row in enumerate(cont):
        for cluster, val in enumerate(row):
            #H(K|C)
            one = val/float(N)
            two = val/float(label_counts[label])
            if (two+eps != 0.0):
                x += one * np.log(two+eps)
            
            #H(K)
            cluster_sizes[cluster] += val
    
    x *= -1
    
    y = 0.0
    
    for size in cluster_sizes:
        num = eps + size / N
        if(num != 0.0):
            y += (num) * np.log( num ) #Different from paper
    
    y *= -1
    
    return 1.0 - (x / y)

#computes and returns the V-measure
def V_measure(cont, B=1):
    h = homogeneity(cont)
    c = completeness(cont)
    
    v = ((1 + B) * h * c) / (( B * h) + c)
    
    return v

#returns Homogeneity, Completeness, and V-Measure
def All_measures(cont, B=1):
    h = homogeneity(cont)
    c = completeness(cont)
    
    v = ((1 + B) * h * c) / (( B * h) + c)
    
    return (h,c,v)

# returns the accuracy based off the contingency table.
def checkAccuracy(cont):

    total = 0.0
    correct =  0.0
    
    m = [0] * len(cont[0])
    idx = [0] * len(cont[0])
    
    for i,r in enumerate(cont):
        for j,v in enumerate(r):
            if i == j:
                total += v
            else:
                total += v
            
            if(v > m[j]):
                idx[j] = i
                m[j] = v
    
    for i,r in enumerate(cont):
        for j,v in enumerate(r):
            if i == idx[j]:
                correct += v

    #print "Final Error : %d of %d" % (error, total)
    return (correct, total, correct/total * 100)



def jaccard(clustering):
    true = map(lambda c: map(lambda x: x.label, c), clustering)
    true = [x for y in true for x in y]

    pred = []
    for i,y in enumerate(clustering):
        pred += [i]*len(y)


    ss = 0
    sd = 0
    ds = 0

    for i in range(len(true)):
        for j in range(i+1,len(true)):
            one = true[i] == true[j]
            two = pred[i] == pred[j]
            if (one and two):
                ss += 1
            elif (one):
                sd += 1
            elif (two):
                ds += 1

    return float(ss) / float(ss + sd + ds)


def printMetrics(cluster):
	
	#clustering = sorted(cluster, key=lambda x: len(x), reverse=True)
	cont = contingency(cluster)
	
	print "Accuracy: %d of %d: %.3f %%" % (checkAccuracy(cont))
        print "H: %f C: %f V: %f" % (All_measures(cont)) + " JCC: %f" % (jaccard(cluster))

	utils.print_cont(cont)
	
	print
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
