#!/usr/bin/python

import numpy as np
import scipy.spatial.distance as metric
import matplotlib.pyplot as pl
import sys

import incOptics as optics


means = [[7.0,5.0],[7.0,15.0],[12.0,12.0]]
covs = [[[ 1.6, 0.0],
	     [ 0.0, 1.3]],
		 
		[[ 2.0, 0.0],
         [ 0.0, 0.2]],
		 
		[[ 0.2, 0.0],
         [ 0.0, 2.5]]]


def getData(p=(100,300), mean=(3.0,17.0), scale=(0.1,2), dim=2, K=3):
	data = []
	targets = []
	
	for i in range(K):
		n = int(np.random.random()*(p[1] - p[0]) + p[0])
		m = means[i]
		cov = covs[i]
		
		#cov = np.zeros(shape=[dim,dim], dtype=float)
		#for d in range(dim):
		#	cov[d,d] = c[d]
		
		
		print "N:", n
		print "mean:"
		print m
		print "cov:"
		print cov
		
		pts = np.random.multivariate_normal(mean=m, cov=cov, size=n)	
		data.extend(pts)
			
		targets.extend([i]*n)

	return np.array(data), np.array(targets)


def main(args):
	data, targets = getData(K=3)
	
	print data.shape
	print targets.shape

	
	minPts = 80
	
	distances = metric.squareform(metric.pdist(data))
	print distances.shape
	
	out = optics.OPTICS(distances, minPts)
	sep = optics.separateClusters(out, minPts, display=True)
	
	style = ['b.','rx', 'g1', 'm.', 'c.']

	pl.figure()

	pl.ylim([0,20])
	pl.xlim([0,20])
	pl.axis('off')

	for i, c in enumerate(sep):
		idx = np.array(map(lambda x: x._id, c), dtype=int)
		#print idx.shape
		#print data[idx,0]
		pl.plot(data[idx,0], data[idx,1], style[i % len(style)])
	
	pl.show()

if __name__ == "__main__":
	main(sys.argv)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	