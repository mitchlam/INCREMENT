#!/usr/bin/python

import sys
import matplotlib.pyplot as pl
import numpy as np
import re

def loadCSV(filename):
    data = []

    with open(filename,"r") as f:
        for line in f:
            l = [i.strip() for i in line.split(',')]
            data.append(l)

    return data[0], data[1:]


EXT = {"query": "queries.csv", "misc": "misc.csv"}
PLOT = ["Accuracy", "Homogeneity", "Completeness", "V-measure", "JCC"]
COLOR = ['c', 'g', 'b', 'r', 'y'] 

def main(args):
    filename = args[1]
    dataName = args[2]

    outfolder = None
    postfix = ""

    if len(args) > 3:
        outfolder = args[3]

    if len(args) > 4:
        postfix = "_" + args[4]

    header, data = loadCSV(filename + EXT["query"])
    initHeader, initData = loadCSV(filename + EXT["misc"])

    #print header
    data = np.array(data, dtype=float)
    idx = header.index(PLOT[0])
    data[:,idx] /= 100
   
    #print initHeader
    #print initData

    idx = initHeader.index(PLOT[0])
    initData[0][idx] = float(initData[0][idx])/100

    idx = initHeader.index('Alg')
    alg = initData[0][idx]
    
    if alg  == "pre":
        alg = "CONFIRM"


    title = "%s: %s" % (dataName,alg)
    pl.title(title)
    pl.ylim([-0.05,1.05])
    for p,c in zip(PLOT, COLOR):
        i = header.index(p)
        ii = initHeader.index(p)
        pl.plot(data[:,0], [float(initData[0][ii])] * data.shape[0], "--", color=c)
        pl.plot(data[:,0], data[:,i], label=header[i], color=c)

    pl.legend(loc= 'lower right')

    if outfolder == None:
        pl.show()
    else:
        outfile = "%s/%s_%s%s.png" %(outfolder,dataName.replace(" ", "_"), alg, postfix) 
        print outfile
        pl.savefig(outfile)


if __name__ == "__main__":
    main(sys.argv)
