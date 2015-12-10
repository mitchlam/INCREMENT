#!/usr/bin/python

import sys
import matplotlib.pyplot as pl
import numpy as np
import re


def printTable(name, init, final, improve):
    init = np.asarray(init)
    final = np.asarray(final)
    improve = np.asarray(improve)
    
    init_str = ""
    final_str = ""
    improve_str = ""

    for i in range(init.size):
        init_str += " & %.2f" % init[i]
        final_str += " & %.2f" % final[i]
        improve_str += " & %.f" % improve[i]

    print '\\hline'
    s = '''\\multirow{3]{*}{\\begin{turn}{90} ~%s~ \\end{turn}} & Initial %s \\\\
             & Final %s \\\\
             & Improvement %s \\%% \\\\''' % (name, init_str, final_str, improve_str)

    print s


def loadCSV(filename):
    data = []

    with open(filename,"r") as f:
        for line in f:
            l = [i.strip() for i in line.split(',')]
            data.append(l)

    return data[0], data[1:]


def percentImprovement(x,y):
    diff = y - x
    i = (diff/x) * 100

    i[x==0] = 0

    return i


EXT = {"query": "queries.csv", "misc": "misc.csv", "final": "final.csv"}
MEASURE = ["Accuracy", "Homogeneity", "Completeness", "V-measure", "JCC"]
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
    finalHeader, finalData = loadCSV(filename + EXT["final"])

    #print header
    data = np.array(data, dtype=float)
    idx = header.index(MEASURE[0])
    data[:,idx] /= 100
 

    idx = initHeader.index('Alg')
    alg = initData[0][idx]  

    if alg  == "pre":
        alg = "CONFIRM"
    
    
    initMeasures = []
    finalMeasures = []

    for m in MEASURE:
        i = initHeader.index(m)
        j = finalHeader.index(m)

        initMeasures.append(float(initData[0][i]))
        finalMeasures.append(float(finalData[0][j]))

    initMeasures = np.array(initMeasures)
    finalMeasures = np.array(finalMeasures)

    improvement = percentImprovement(initMeasures, finalMeasures)
    #print alg, initMeasures, improvement, finalMeasures

    idx = initHeader.index(MEASURE[0])
    initData[0][idx] = float(initData[0][idx])/100
    
    ind = np.arange(len(improvement))

    #pl.bar(ind,improvement)
    #pl.show()

    #print initHeader
    #print initData



    


    title = "%s: %s" % (dataName,alg)
    pl.title(title)
    pl.ylim([-0.05,1.05])
    for p,c in zip(MEASURE, COLOR):
        i = header.index(p)
        ii = initHeader.index(p)
        pl.plot(data[:,0], [float(initData[0][ii])] * data.shape[0], "--", color=c)
        pl.plot(data[:,0], data[:,i], label=header[i], color=c)

    pl.legend(loc= 'lower right')
    pl.xlabel("Queries")
    
    if outfolder == None:
        pl.show()
    else:
        outfile = "%s/%s_%s%s.png" %(outfolder,dataName.replace(" ", "_"), alg, postfix) 
        print outfile
        pl.savefig(outfile)

        outfile = "%s/%s_%s%s.csv" % (outfolder, dataName.replace(" ", "_"), alg, postfix)
       
        f = "%0.2f"

        initMeasures = map(lambda s: f % s, initMeasures)
        finalMeasures = map(lambda s: f % s, finalMeasures)
        improvement = map(lambda s: f % s, improvement)
    
        #for i,m in enumerate(MEASURE):
        #    printTable(m, initMeasures[i], finalMeasures[i], improvement[i])

        d = np.array(zip(initMeasures, finalMeasures, improvement))
        out = "%s, %s, %s" % (dataName, alg, ",".join(d.flatten()))
        print out
        with open(outfile, "w") as f:
            f.write(out)
            f.write("\n")


if __name__ == "__main__":
    main(sys.argv)
