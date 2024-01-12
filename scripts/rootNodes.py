#!/usr/bin/env python


import io, sys, numpy, scipy, struct, os
from scipy import io
from scipy import sparse
from numpy import inf
import numpy as np
import networkx as nx
# initialize access to UF SpM collection
# import yaUFget as uf


dramBase = 0x10000000


absolute_path = os.path.dirname(__file__)
relative_path = "dataset/"
relative_path_localroot = "txt/"
graphDataRoot = os.path.join(absolute_path, relative_path)
localRoot = os.path.join(absolute_path, relative_path_localroot)
relative_path_localtxt = "txt/"
relative_path_localmat = "matlab/"
localtxtFolder = os.path.join(absolute_path, relative_path_localtxt)




def makeGraphList():
    graphs = []
    # Get all the files ending with .mat extension 
    for file in os.listdir(localRoot):
    	if file.endswith(".txt"):
        	graphs += [file.rsplit( ".", 1 )[0]]
    # print(graphs)
    # graphs = ["rmat-19-32"]
    return graphs





def buildGraphManager(c):
    graphs = makeGraphList()
    for g in graphs:
        m = GraphMatrix()
        print ("Graph " + g + " with " + str(c) + " partitions")
        m.prepareGraph(g, c)
      
def buildGraphManagerSingle(name,c):
    g=name
    m = GraphMatrix()
    print ("Graph " + g + " with " + str(c) + " partitions")
    m.prepareGraph(g, c,False)
      

# as the G500 spec says we should also count self-loops,
# removeSelfEdges does not do anything.
def removeSelfEdges(graph):
    return graph


def loadGraph(matrix):
    name_matrix = str(matrix) + '.txt'
    path_to_go = localtxtFolder + name_matrix
    
    # print(path_to_go)
    if scipy.sparse.isspmatrix_csc(matrix) or scipy.sparse.isspmatrix_csr(matrix):
        # return already loaded matrix
        r = removeSelfEdges(matrix)
        # do not adjust dimensions, return directly
        return r
    else:
        # load matrix from local file system
        # r = scipy.io.loadmat(path_to_go)['M']
        arr = np.loadtxt(path_to_go, dtype=int)
        # print(r)
        # print(arr)
    # else:
    # load matrix from University of Florida sparse matrix collection
    # r=removeSelfEdges(uf.get(matrix)['A'])
    # graph must have rows==cols, clip matrix if needed
   
    test = np.ones((len(arr[:, 0]),), dtype=int)
    arr = sparse.csr_matrix(((test,((arr[:, 0],(arr[:, 1]))))))
    #print r
    rows = arr.shape[0]
    cols = arr.shape[1]
    # print(rows)
    #print cols
    if rows != cols:
        dim = min(rows, cols)
        arr = arr[0:dim, 0:dim]
    return arr


#################### Generate ROOT Nodes #################
# def buildRootNodes():
#     print("HERE")
#     graphs = makeGraphList()
#     rnl = dict()
#     for g in graphs:
#         print("Generating root nodes for " + str(g) + "\n===========================\n")
#         rnl[g] = generateRootNodes(g)
#     return rnl

def buildRootNodesSingle(name):
    rnl = dict()
    g=name
    print("Generating root nodes for " + str(g) + "\n===========================\n")
    rnl[g] = generateRootNodes(g)
    return rnl

def generateRootNodes(graph):
    A = loadGraph(graph)
    rootNodes = []
    G = nx.from_scipy_sparse_array(A)
    rootCandidate = max(dict(G.degree()).items(), key = lambda x : x[1])[0]
    bfsDepthMin = max(dict(G.degree()).items(), key = lambda x : x[1])[1]
    edges = len(list(nx.edge_bfs(G,source=rootCandidate)))
    print("# Neigbours :",bfsDepthMin,", ~Connected Edges: ",edges)
    rootNodes+=[rootCandidate]
    return rootNodes


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # if(sys.argv[1] == "all"):
        # num_partition = int(sys.argv[2])
        # buildGraphManager(num_partition)
        # if(sys.argv[2] == "genrootnodes"):
        #     print("TEST")
        #     emp_num = dict()
        #     emp_num = buildRootNodes()
        #     print(emp_num)
    # else:
        # num_partition = int(sys.argv[2])
        # buildGraphManagerSingle(sys.argv[1],num_partition)
        print("======== Generate Root Nodes =======")
        emp_num = dict()
        emp_num = buildRootNodesSingle(sys.argv[1])
        print(emp_num)