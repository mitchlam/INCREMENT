#!/usr/bin/python

import sys
import utils
import random
import numpy as np

from scipy.spatial import distance as Distance


class HMRF:

	def __init__(self, distance, aggregate, K = None):
		self.distance = distance
		self.aggregate = aggregate
		self.K = K
		self.representatives = None
		self.clusters = None
		self.maxDist = 0
		
	#Assume M and C are tuples of indexes into data
	def cluster(self, data, M, C, neighborhood=None):
		self.initialize(data,M,C, neighborhood)
		
		for i in range(100):
			self.Estep(data,M,C)
			self.Mstep(data,M,C)
		
		return self.clusters
		
	
	def initialize(self, data, M, C, neighborhood=None):
		if(neighborhood == None):
			neighborhood, M, C = self.getNeighborhoods(M,C)
		'''
		print
		print "Merge:", sorted(map(sorted,M))
		'''
		print
		print "Neighborhood:", sorted(map(sorted,neighborhood))
		
		n = map(lambda a: map(lambda x: data[x], a), neighborhood)
		#print n
		
		centroids = map(self.aggregate,n)
		#print centroids
		
		#print self.representatives
		L = len(neighborhood)
		
		if(self.K == None or L <= self.K):
			self.representatives = centroids
			self.clusters = neighborhood
		
		#TODO: Handle K being different from L
		
		return neighborhood
	
	def getNeighborhoods(self, M,C):
		
		sets = map(set, M)
		
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
		
		
		neighborhood = []
		for s in sets:
			f = list(s)
			neighborhood.append(f)
			
		return neighborhood, M, C

	
	
	#update cluster membership
	def Estep(self, data, M, C):
		#x and u are indexes
		def obj(x,u):

			value = self.distance(data[x],self.representatives[u])
			for i,j in M:
				if x != i and x != j:
					continue
				
				y = j
				if x == j:
					y = i
				
				indicator = 1	
				
				if y in self.clusters[u]:
					indicator = 0
				
				d = self.distance(data[x],data[y])
				
				self.maxDist = max(self.maxDist, d)
				
				value += d * indicator
				
			for i,j in C:
				if x != i and x != j:
					continue
				
				y = j
				if x == j:
					y = i
				
				indicator = 0	
				
				if y in self.clusters[u]:
					indicator = 1
					
				d = self.distance(data[x],data[y])
				
				self.maxDist = max(self.maxDist, d)
				
				value += (self.maxDist - d) * indicator
			
			return value
		
		
		unchanged = True
		
		while (unchanged):
			unchanged = False
			order = range(len(data))
			random.shuffle(order)
			
			for o in order:
				old = None
				for i,c in enumerate(self.clusters):
					if o in c:
						old = i
						c.remove(o)
						break
					
				idxs = range(len(self.clusters))
				
				objectives = map(lambda i: obj(o,i), idxs)
				
				new = utils.arg_min(objectives)
				self.clusters[new].append(o)
				
				unchanged |= (old != new)
	
	#update cluster representatives
	def Mstep(self, data, M,C):
		c = map(lambda x: map(lambda i: data[i], x), self.clusters)
		
		centroids = map(self.aggregate, c)
		self.representatives = centroids
		
		#TODO: Handle Parameterized distances



def main(args):
	data = [[1,0],[0,1],[.2,.3],[.6,.2],[.8,.2],[.5,.5],[.2,.7],[.5,.9],[.2,1]]
	M = [(0,4), (1,7), (6,7), (2,5)]
	#M = [(0,4), (1,5), (4,5)]
	C = [(0,1), (4,8), (3,6)]
	
	data = np.array(data)
	aggr = lambda x: np.mean(x, axis=0)
	hmrf = HMRF(Distance.euclidean, aggr)
	
	print hmrf.cluster(data, M, C)
	

if __name__ == "__main__":
	main(sys.argv)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	