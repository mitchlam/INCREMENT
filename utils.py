import string

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    