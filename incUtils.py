import string
import os
import sys

import h5py
import random
import numpy as np

import caffe
from caffe import layers as L

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def EuclideanDistance(x,y):
    s = 0.0

    for i,j in zip(x,y):
        s += (i-j)*(i-j)

    return sqrt(s)

def mode(x):
    d = {}
    
    for i in x:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1

    pairs = sorted(d.items(), key= lambda x: x[1], reverse=True)
    
    return pairs[0][0]

def arg_min(x):
    m = min(x)
    return x.index(m)

def pairwise(args, func, symmetric=True):
	mat = []
	for x, arg1 in enumerate(args):
		row = []
		for y, arg2 in enumerate(args):
			if symmetric and y < x:
				val = mat[y][x]
			else:
				val = func(arg1, arg2)
			row.append(val)
		mat.append(row)
	return mat

def split_mat(mat, row_len):
	mats = []
	total_row_length = len(mat[0])
	start = 0
	end = row_len
	while start < total_row_length:
		new_mat = []
		for row in mat:
			new_row = row[start:end]
			new_mat.append(new_row)
		mats.append(new_mat)
		start += row_len
		end += row_len
	return mats

def insert_indices(mat, row_start=0, col_start=0):
	row0 = range(col_start, len(mat[0]) + col_start)
	row0.insert(0, " ")
	for x,row in enumerate(mat, row_start):
		row.insert(0, x)
	mat.insert(0, row0)

def print_cont(mat, clusters_per_mat=20):
    mats = split_mat(mat, clusters_per_mat)
    
    print
    print "Rows are labels, Columns are Clusters"
    print
    for x, m in enumerate(mats):
        insert_indices(m, col_start = clusters_per_mat*x)
        print_mat(m)
        print
    print
    
def print_mat(mat):
	max_lens = [max([len(str(r[i])) for r in mat])
					 for i in range(len(mat[0]))]

	print "\n".join(["".join([string.rjust(str(e), l + 2)
							for e, l in zip(r, max_lens)]) for r in mat])

def generateRandomPairs(data,labels, num_pairs, constraints, batch_size=1):
	pairs = []
	sims = []
	
	while (len(pairs) < num_pairs) or (len(pairs) % batch_size != 0):
		i = random.randint(0, len(data) - 1)
		j = random.randint(0, len(data) - 1)
		
		s = None
			
		if labels[i] == labels[j]:
			s = 1.0
		elif (labels[j] in constraints[labels[i]]) or (labels[i] in constraints[labels[j]]):
			s = 0.0
		else:
			continue
		
		x = data[i]
		y = data[j]
		p = np.zeros((2,x.shape[0]), dtype= np.float32)
		
		p[0] = x[:]
		p[1] = y[:]
			
		if s != None:
			pairs.append(p)
			sims.append(s)
		
		#print "%d %% %d = %d" % (len(pairs), batch_size, len(pairs)%batch_size)
		
	#print "Pairs Filled"
	
	return np.array(pairs)[:,:,:,np.newaxis], np.array(sims, dtype=np.float32)
	#return np.ascontiguousarray(np.array(pairs)[:,:,:,np.newaxis]), np.ascontiguousarray(sims, dtype=np.float32)
	
def generatePairs(data,labels, constraints, batch_size=1):
	pairs = []
	sims = []
	
	for i,x in enumerate(data):
		for j,y in enumerate(data):
			if j <= i:
				continue
			
			p = np.zeros((2,x.shape[0]), dtype= np.float32)
			p[0] = x[:]
			p[1] = y[:]
			
			s = None
			
			if labels[i] == labels[j]:
				s = 1.0
			elif (labels[j] in constraints[labels[i]]) or (labels[i] in constraints[labels[j]]):
				s = 0.0
			
			if s != None:
				pairs.append(p)
				sims.append(s)
	
	while len(pairs)%batch_size != 0:
		i = random.randint(0, len(data) - 1)
		j = random.randint(0, len(data) - 1)
		
		x = data[i]
		y = data[j]
		p = np.zeros((2,x.shape[0]), dtype= np.float32)
		
		p[0] = x[:]
		p[1] = y[:]
		pairs.append(p)
		
		s = 1.0 if labels[i] == labels[j] else 0.0
		sims.append(s)
		
		#print "%d %% %d = %d" % (len(pairs), batch_size, len(pairs)%batch_size)
		
	comb = zip(pairs, sims)
	random.shuffle(comb)
	pairs[:], sims[:] = zip(*comb)
	
	return np.array(pairs)[:,:,:,np.newaxis], np.array(sims, dtype=np.float32)
	#return np.ascontiguousarray(np.array(pairs)[:,:,:,np.newaxis]), np.ascontiguousarray(sims, dtype=np.float32)

def clearPlots():
	plt.close('all')

def displayPlot(feat, labels, title = None, block=True):

	c = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff', '#990000', '#999900', '#009900', '#009999', '#000099', '#990099', '#000000', '#999999', '#9900ff', '#ff0099', '#99ff00', '#ff9900', '#0099ff', '#00ff99']	
	s = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', '+', 'x', 'd', '_', '1', '2', '3', '4', '|']
	if (feat.shape[1] == 3):
		f = plt.figure()
		ax = f.add_subplot(111,projection='3d')
	
		#print feat[:5]
	
		lbls = set(labels)
		for i in lbls:
			idx = labels == i
			ax.plot(feat[idx,0].flatten(), feat[idx,1].flatten(), feat[idx,2].flatten(), s[i%len(s)], c=c[i%len(c)])
	
		if title != None:
			plt.title(title)
	
		plt.show(block=block)
		
	elif( feat.shape[1] == 2):
		f = plt.figure()
	
		#print feat[:5]
	
		lbls = set(labels)
		for i in lbls:
			idx = labels == i
			plt.plot(feat[idx,0].flatten(), feat[idx,1].flatten(), s[i%len(s)], c=c[i%len(c)])
	
		if title != None:
			plt.title(title)
	
		plt.show(block=block)

def writeH5(name, **kwargs):
    with h5py.File(name+".h5", "w") as f:
        for key,value in kwargs.items():
            f[key] = value

    with open(name+".txt", "w") as f:
        f.write(name+".h5")


def enumeratation(targets):
	#print targets
	lbls = set()
	lbls.update(targets)
	lbls = list(lbls)
	
	labels = map(lambda t: lbls.index(t), targets)
	
	return labels

def feedData(net, data):
	net.blobs['data'].reshape(*data.shape)
	net.reshape()
	
	net.blobs['data'].data[...] = data
	net.reshape()
	net.blobs['data'].data[...] = data
	
	return net.forward()


def createTrainSiamese(source, batch_size, vector_size, output_size=10):
	n = caffe.NetSpec()
	
	n.pair_data, n.sims = L.HDF5Data(source= source, batch_size=batch_size, ntop=2)
	#n.pair_data, n.sims = L.MemoryData(batch_size=batch_size, channels=2, height=vector_size, width=1, ntop=2)
	
	n.data, n.data_p = L.Slice(n.pair_data, axis=1, ntop=2)
	
	#Side One
	addMainLeg(n, output_size)
	
	#Side Two
	addPairedLeg(n, output_size)
	
	#Loss
	n.loss = L.ContrastiveLoss(n.feat, n.feat_p,n.sims, margin=1)
	
	return n.to_proto()

def createDeploySiamese(source, batch_size, output_size=10):
    n = caffe.NetSpec()

    n.data = L.HDF5Data(batch_size=batch_size, source=source)

    addMainLeg(n, output_size)

    return n.to_proto()


def addMainLeg(n, output_size):
	n.ip1 = L.InnerProduct(n.data, param=[dict(name="ip1_w", lr_mult=1), dict(name="ip1_b", lr_mult=2)], num_output=500 , weight_filler=dict(type='xavier'),  bias_filler=dict(type='gaussian', std=0.1))
	n.s1 = L.Sigmoid(n.ip1, in_place=True)
	
	n.ip2 = L.InnerProduct(n.ip1, param=[dict(name="ip2_w", lr_mult=1), dict(name="ip2_b", lr_mult=2)], num_output=250 ,weight_filler=dict(type='xavier'),  bias_filler=dict(type='constant'))
	n.s2 = L.Sigmoid(n.ip2, in_place=True)
	'''
	n.ip3 = L.InnerProduct(n.ip2, param=[dict(name="ip3_w", lr_mult=1), dict(name="ip3_b", lr_mult=2)], num_output=100 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	n.s3 = L.Sigmoid(n.ip3, in_place=True)
	'''
	n.feat = L.InnerProduct(n.ip2, param=[dict(name="feat_w", lr_mult=1), dict(name="feat_b", lr_mult=2)], num_output = output_size, weight_filler=dict(type="xavier"),  bias_filler=dict(type='constant'))
	
def addPairedLeg(n, output_size):
	n.ip1_p = L.InnerProduct(n.data_p, param=[dict(name="ip1_w", lr_mult=1), dict(name="ip1_b", lr_mult=2)], num_output=500 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='gaussian', std=0.1))
	n.s1_p = L.Sigmoid(n.ip1_p, in_place=True)
	
	n.ip2_p = L.InnerProduct(n.ip1_p, param=[dict(name="ip2_w", lr_mult=1), dict(name="ip2_b", lr_mult=2)], num_output=250 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	n.s2_p = L.Sigmoid(n.ip2_p, in_place=True)
	'''
	n.ip3_p = L.InnerProduct(n.ip2_p, param=[dict(name="ip3_w", lr_mult=1), dict(name="ip3_b", lr_mult=2)], num_output=100 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	n.s3_p = L.Sigmoid(n.ip3_p, in_place=True)
	'''
	n.feat_p = L.InnerProduct(n.ip2_p, param=[dict(name="feat_w", lr_mult=1), dict(name="feat_b", lr_mult=2)], num_output = output_size, weight_filler=dict(type="xavier"),  bias_filler=dict(type='constant'))
	
	

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    