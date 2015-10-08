import string
import os
import sys

import lmdb
import h5py
import random
import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P

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
	one = []
	two = []
	#pairs = []
	sims = []
	
	idx = 0
	
	#while (len(pairs) < num_pairs) or (len(pairs) % batch_size != 0):
	while (idx < num_pairs) or (idx % batch_size != 0):	
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
		#p = np.zeros((2,x.shape[0]), dtype= np.float32)
		
		#p[0] = x[:]
		#p[1] = y[:]
			
		if s != None:
			#pairs.append(p)
			one.append(x)
			two.append(y)
			sims.append(s)
			idx += 1
		
		#print "%d %% %d = %d" % (len(pairs), batch_size, len(pairs)%batch_size)
		
	#print "Pairs Filled"
	
	return np.array(one), np.array(two), np.array(sims, dtype=np.float32)
	#return np.ascontiguousarray(np.array(pairs)[:,:,:,np.newaxis]), np.ascontiguousarray(sims, dtype=np.float32)
	
def generatePairs(data,labels, constraints, batch_size=1):
	#pairs = []
	one = []
	two = []
	sims = []
	
	for i,x in enumerate(data):
		for j,y in enumerate(data):
			if j <= i:
				continue
			
			#p = np.zeros((2,x.shape[0]), dtype= np.float32)
			#p[0] = x[:]
			#p[1] = y[:]
			
			s = None
			
			if labels[i] == labels[j]:
				s = 1.0
			elif (labels[j] in constraints[labels[i]]) or (labels[i] in constraints[labels[j]]):
				s = 0.0
			
			if s != None:
				#pairs.append(p)
				one.append(x)
				two.append(y)
				sims.append(s)
	
	idx = len(sims)
	while idx%batch_size != 0:
		i = random.randint(0, len(data) - 1)
		j = random.randint(0, len(data) - 1)
		if i == j:
			continue
		
		x = data[i]
		y = data[j]
		#p = np.zeros((2,x.shape[0]), dtype= np.float32)
		
		#p[0] = x[:]
		#p[1] = y[:]
		#pairs.append(p)
		
		one.append(x)
		two.append(y)
		
		s = 1.0 if labels[i] == labels[j] else 0.0
		sims.append(s)
		idx+=1
		
		#print "%d %% %d = %d" % (len(pairs), batch_size, len(pairs)%batch_size)
	'''	
	comb = zip(pairs, sims)
	random.shuffle(comb)
	pairs[:], sims[:] = zip(*comb)
	'''
	return np.array(one), np.array(two), np.array(sims, dtype=np.float32)
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


def writeLMDB(name, data, labels = None, map_size=2147483647, debug=False):
	env = lmdb.open(name, map_size=map_size, writemap= True, map_async=True, meminit=False, metasync=False)
	
	if labels != None:
		datum = map(lambda d,l : caffe.io.array_to_datum(d.astype(float), int(l)).SerializeToString(), data, labels)
	else:
		datum = map(lambda d: caffe.io.array_to_datum(d.astype(float)).SerializeToString(),data)
	
	print "Writing datum"
	for i in xrange(len(datum)):
		'''
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = data.shape[1]
		datum.height = data.shape[2]
		datum.width = data.shape[3]
		datum.data = bytes(data[i])
		'''
		'''
		datum = caffe.io.array_to_datum(data[i].astype(float))
	
		if (labels != None):
			datum.label = int(labels[i])
		'''
		
		str_id = '{:08}'.format(i)
	
		with env.begin(write=True) as txn:
			err = txn.put(str_id.encode('ascii'), datum[i])
		
			if not err:
				print "Could not add", i, "to", name
	
MAX_INT = 2147483647/2
'''
class H5:
	def __init__ (self, name, shape):
		self.name = name
		self.buffer = {}
		self.split_size, self.instance_size = calculateSplitSize(shape)
		self.written = 0
		self.filenum = 0
		
		self.filename = lambda: self.name + "_" + str(self.filenum) + ".h5"
		#clear previous file
		with open(name + ".txt", "w") as f:
			pass
		
		self.openNew()

	#splits data into subsets with the first split being of size initial
	def split(self, data, size, initial = 0):
		print "splitting"
		print "data:", data
		print "size:", size
		print "initial:", initial 
		s = []
		s.append(data[:initial])
		
		data = data[initial:]
		for i in range(0, len(data), size):
			s.append(data[i:i+size])

	
		result= filter(lambda a: a.size > 0, s)
		print "Final:", result
		print 
		return result

	def flush(self):
		print "Flushing"
		filename = self.filename()
		with h5py.File(filename, 'a') as f:
			for key, value in self.buffer.items():
				print "Writing:", key, value
				if key in f:
					f[key] += value
				else:
					f[key] = value
	
	def openNew(self):
		self.filenum += 1
		filename = self.filename()
		self.written = 0
		
		#Clear file if exists
		with h5py.File(filename, 'w'):
			pass
		
		#Appand to file
		with open(self. name+".txt", 'a') as f:
			f.write(filename + "\n")
	

	def write(self, **kwargs):
		data = {}
		length = 1
		for key, value in kwargs.items():
			data[key] = self.split(value, self.split_size, self.split_size - self.written)
			length = len(data[key])
		
		print length
		for i in range(length):
			instances = 1
			for key, value in data.items():
				self.buffer[key] = value[i]
				instances = len(value[i])
			
			self.written += instances * self.instance_size
			self.flush()
			if self.written > (self.split_size * 0.75):
				self.openNew()
		
	def __del__(self):
		pass
	
'''

def writeH5(name, **kwargs):
	
	split_size = MAX_INT
	size = 1
	length = 0
	for key, value in kwargs.items():
		length = value.shape[0] #assuming the same number of instances
		ss, s = calculateSplitSize(value.shape)
		
		if ss < split_size:
			split_size = ss
			size = s

			
	#print "Size per split", split_size
	
	with open(name+".txt", "w") as txt:
		numFiles = 0
		for i in range(0, length, split_size):
			numFiles += 1
			filename = name + "_" + str(numFiles) + ".h5"
			txt.write(filename + "\n")
			with h5py.File(filename, "w") as f:
				for key,value in kwargs.items():
					#print "Shape of H5 Value:", value.shape
					f[key] = value[i:i+split_size]
		#print "Files created:", numFiles

def calculateSplitSize(shape):
	split_size = MAX_INT
	size = 1
	
	for l in reversed(shape[1:]):
		size *= l
		
	return MAX_INT/size, size

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


def createTrainSiamese(source, batch_size, output_size=10, convolution=False):
	n = caffe.NetSpec()
	
	n.data, n.data_p, n.sims = L.HDF5Data(source=source, batch_size=batch_size, shuffle=True, ntop=3)
	#n.pair_data, n.sims = L.Data(source = source, backend=P.Data.LMDB, batch_size=batch_size, ntop=2)
	#n.pair_data, n.sims = L.MemoryData(batch_size=batch_size, channels=2, height=vector_size, width=1, ntop=2)
	
	#n.data, n.data_p = L.Slice(n.pair_data, axis=1, ntop=2)
	
	if convolution:
		addMainConvLeg(n, output_size)
		addPairedConvLeg(n, output_size)
	else:
		addMainLeg(n, output_size)
		addPairedLeg(n, output_size)
	
	#Loss
	n.loss = L.ContrastiveLoss(n.feat, n.feat_p,n.sims, margin=1)
	
	return n.to_proto()

def createDeploySiamese(source, batch_size, output_size=10, convolution=False):
	n = caffe.NetSpec()
	
	n.data = L.HDF5Data(batch_size=batch_size, source=source)
	#n.data = L.Data(backend=P.Data.LMDB, batch_size=batch_size, source=source)
	
	if convolution:
		addMainConvLeg(n,output_size)
	else:
		addMainLeg(n, output_size)
	
	return n.to_proto()


def addMainLeg(n, output_size):
	n.ip1 = L.InnerProduct(n.data, param=[dict(name="ip1_w", lr_mult=1), dict(name="ip1_b", lr_mult=2)], num_output=500 , weight_filler=dict(type='xavier'),  bias_filler=dict(type='gaussian', std=0.1))
	n.s1 = L.Sigmoid(n.ip1, in_place=True)

	#n.ip2 = L.InnerProduct(n.ip1, param=[dict(name="ip2_w", lr_mult=1), dict(name="ip2_b", lr_mult=2)], num_output=250 ,weight_filler=dict(type='xavier'),  bias_filler=dict(type='constant'))
	#n.s2 = L.Sigmoid(n.ip2, in_place=True)
	'''
	n.ip3 = L.InnerProduct(n.ip2, param=[dict(name="ip3_w", lr_mult=1), dict(name="ip3_b", lr_mult=2)], num_output=100 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	n.s3 = L.Sigmoid(n.ip3, in_place=True)
	'''
	n.feat = L.InnerProduct(n.ip1, param=[dict(name="feat_w", lr_mult=1), dict(name="feat_b", lr_mult=2)], num_output = output_size, weight_filler=dict(type="xavier"),  bias_filler=dict(type='constant'))
	
def addPairedLeg(n, output_size):
	n.ip1_p = L.InnerProduct(n.data_p, param=[dict(name="ip1_w", lr_mult=1), dict(name="ip1_b", lr_mult=2)], num_output=500 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='gaussian', std=0.1))
	n.s1_p = L.Sigmoid(n.ip1_p, in_place=True)
	
	#n.ip2_p = L.InnerProduct(n.ip1_p, param=[dict(name="ip2_w", lr_mult=1), dict(name="ip2_b", lr_mult=2)], num_output=250 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	#n.s2_p = L.Sigmoid(n.ip2_p, in_place=True)
	'''
	n.ip3_p = L.InnerProduct(n.ip2_p, param=[dict(name="ip3_w", lr_mult=1), dict(name="ip3_b", lr_mult=2)], num_output=100 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	n.s3_p = L.Sigmoid(n.ip3_p, in_place=True)
	'''
	n.feat_p = L.InnerProduct(n.ip1_p, param=[dict(name="feat_w", lr_mult=1), dict(name="feat_b", lr_mult=2)], num_output = output_size, weight_filler=dict(type="xavier"),  bias_filler=dict(type='constant'))
	

def addMainConvLeg(n, output_size):
	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, param=[dict(name="conv1_w", lr_mult=1), dict(name="conv1_b", lr_mult=2)], weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, param=[dict(name="conv2_w", lr_mult=1), dict(name="conv2_b", lr_mult=2)], weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	n.ip1 = L.InnerProduct(n.pool2, param=[dict(name="ip1_w", lr_mult=1), dict(name="ip1_b", lr_mult=2)], num_output=500 , weight_filler=dict(type='xavier'),  bias_filler=dict(type='gaussian', std=0.1))
	n.r1 = L.ReLU(n.ip1, in_place=True)	

	n.feat = L.InnerProduct(n.ip1, param=[dict(name="feat_w", lr_mult=1), dict(name="feat_b", lr_mult=2)], num_output = output_size, weight_filler=dict(type="xavier"),  bias_filler=dict(type='constant'))
	
def addPairedConvLeg(n, output_size):
	n.conv1_p = L.Convolution(n.data_p, kernel_size=5, num_output=20, param=[dict(name="conv1_w", lr_mult=1), dict(name="conv1_b", lr_mult=2)], weight_filler=dict(type='xavier'))
	n.pool1_p = L.Pooling(n.conv1_p, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.conv2_p = L.Convolution(n.pool1_p, kernel_size=5, num_output=50, param=[dict(name="conv2_w", lr_mult=1), dict(name="conv2_b", lr_mult=2)], weight_filler=dict(type='xavier'))
	n.pool2_p = L.Pooling(n.conv2_p, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	n.ip1_p = L.InnerProduct(n.pool2_p, param=[dict(name="ip1_w", lr_mult=1), dict(name="ip1_b", lr_mult=2)], num_output=500 ,weight_filler=dict(type='xavier'), bias_filler=dict(type='gaussian', std=0.1))
	n.r1_p = L.ReLU(n.ip1_p, in_place=True)

	n.feat_p = L.InnerProduct(n.ip1_p, param=[dict(name="feat_w", lr_mult=1), dict(name="feat_b", lr_mult=2)], num_output = output_size, weight_filler=dict(type="xavier"),  bias_filler=dict(type='constant'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    