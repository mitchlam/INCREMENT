import numpy as np
import sys
import re

MEASURES = ["Accuracy", "Homogeneity", "Completeness", "V-Measure", "JCC"]

def loadCSV(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            l = [i.strip() for i in line.split(",")]
            data.append(l)

    return np.array(data)

def printTable(name, init, final, improve, percentage=False):
    init = np.asarray(init)
    final = np.asarray(final)
    improve = np.asarray(improve)
    
    init_str = ""
    final_str = ""
    improve_str = ""

    for i in range(init.size):
        if float(init[i])  == 0.0:
            init_str += " & $\\sim$ 0.0%s" % (" \\%" if percentage else "")
            improve_str += " & N/A"
        else:
            init_str += " & %s%s" % (init[i], " \\%" if percentage else "")
            improve_str += " & %s \\%%" % improve[i]
        
        if float(final[i]) == 0.0:
            final_str += " & $\\sim$ 0.0%s" % (" \\%" if percentage else "")
        else:
            final_str += " & %s%s" % (final[i], " \\%" if percentage else "")

    print '\\hline'
    s = '''\\multirow{3}{*}{%s} & Initial %s \\\\
             & Final %s \\\\
             & Improvement %s \\\\''' % (name, init_str, final_str, improve_str)

    return s


def main(args):
    
    print "Reading:", args[1]
    data = loadCSV(args[1]).transpose()

    datasets, algs, measures = data[0,1:], data[1,1:], data[2:, 1:]

    #Create Header
    cols = measures.shape[1] + 2
    col_str = " c |" * cols

    print "\\begin{tabular}{%s}" % col_str
    #Create dataset names

    datasets = map(lambda d: re.sub("((Faces)|(Leeds) )|( 1000)", "", d), datasets)

    dataset_str = " & & " + " & ".join(datasets) + "\\\\"
    print dataset_str
    print "\\hline"
  
    #create algorithm header

    alg_str = " & & " + " & ".join(algs) + "\\\\"
    print alg_str

    for i, m in enumerate(MEASURES):
        tbl = printTable(m, measures[i*3], measures[i*3 + 1], measures[i*3 + 2])
        print tbl

    print "\\end{tabular}"

if __name__ == "__main__":
    main(sys.argv)
