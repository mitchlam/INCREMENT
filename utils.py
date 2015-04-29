

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
