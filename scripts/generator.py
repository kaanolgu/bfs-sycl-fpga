#!/usr/bin/env python


import io, sys, numpy, scipy, struct, os
from scipy import io
from scipy import sparse
from numpy import inf
import random as rnd
import numpy as np
import sys
# initialize access to UF SpM collection
# import yaUFget as uf


dramBase = 0x10000000


absolute_path = os.path.dirname(__file__)
relative_path = "dataset/"
relative_path_localroot = "matlab/"
graphDataRoot = os.path.join(absolute_path, relative_path)
localRoot = os.path.join(absolute_path, relative_path_localroot)
relative_path_localtxt = "txt/"
relative_path_localmat = "matlab/"
localtxtFolder = os.path.join(absolute_path, relative_path_localtxt)



def makeGraphList():
    graphs = []
    # Get all the files ending with .mat extension 
    for file in os.listdir(localRoot):
    	if file.endswith(".mat"):
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
      
   

'''
def removeSelfEdges(graph):
  lil=scipy.sparse.lil_matrix(graph)
  lil.setdiag([0 for i in range(0,graph.shape[0])])
  if scipy.sparse.isspmatrix_csr(graph): 
    return scipy.sparse.csr_matrix(lil)
  elif scipy.sparse.isspmatrix_csc(graph):
    return scipy.sparse.csc_matrix(lil)
'''


# as the G500 spec says we should also count self-loops,
# removeSelfEdges does not do anything.
def removeSelfEdges(graph):
    return graph


class GraphMatrix:
    def __init__(self):
        self.copyCommandBuffer = []
        self.graphName = ""

    def resetCommandBuffer(self):
        self.copyCommandBuffer = []



    def serializeGraphData(self, graph, name, rootFolder, startAddr, startRow,index,PrevRowsValue):
        burst = 256 * 8
        A = loadGraph(graph)

        # save metadata
        fileName = rootFolder + "/" + name + "-meta.bin"
        metaDataFile = open(fileName, "wb")

      


        # save index pointers
        fileName = rootFolder + "/" + name + "-indptr.bin"
        indPtrFile = open(fileName, "wb")
        indPtrFile.write(A.indptr)
        indPtrFile.close()

        # save indptr data start into metadata
        


        # save indices
        fileName = rootFolder + "/" + name + "-inds.bin"
        indsFile = open(fileName, "wb")
        A.indices = A.indices + PrevRowsValue * index
        indsFile.write(A.indices)
        indsFile.close()
    
        # save inds data start into metadata

        print("Rows = " + str(A.shape[0]))
        print("Cols = " + str(A.shape[1]))
        print("NonZ = " + str(A.nnz))
        metaDataFile.write(struct.pack("I", A.shape[0]))
        metaDataFile.write(struct.pack("I", A.shape[1]))
        metaDataFile.write(struct.pack("I", A.nnz))
        metaDataFile.write(struct.pack("I", startRow))  
        metaDataFile.write(struct.pack("I", startAddr))
        metaDataFile.write(struct.pack("I", A.nnz))
        metaDataFile.close()
       
        # return the xmd command and new start address
        return [A.shape[0], A.nnz]



    def prepareGraph(self, graphName, partitionCount, csr=False):
        startAddr = dramBase
        # load the graph
        graph = loadGraph(graphName)
        # print(graph)
        if (csr):
            graphName += "-csr"
        else:
            graphName += "-csc"
            # SpMV BFS needs transpose of matrix
            graph = graph.transpose()

        graphName = graphName.replace("/", "-")

        # create the graph partitions
        partitions = horizontalSplit(graph, partitionCount)
        # add subfolder with name and part count
        targetDir = graphDataRoot + graphName + "-" + str(partitionCount)
        # create the target dir if it does not exist
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        # serialize the graph data and build commands
        i = 0

        # reserve the first 1024 bytes on startAddr for metadata pointers and
        # partition count (config.bin)
      
        configFile = open(targetDir + "/" + graphName + "-config.bin", "wb")
        # first word of config.bin is the number of partitions
        configFile.write(struct.pack("I", partitionCount))
#struct.pack converts integer("I") to binary
        startRow = 0
        startAddr = 0
        savedRows = 0
        for i in range(0, partitionCount):
            print ("Partition " + str(i))
            # write the metadata base ptr into
            configFile.write(struct.pack("I", 313122))
            
            if (csr):
                res = self.serializeGraphData(partitions[i].tocsr(), graphName + "-" + str(i),
                                              targetDir, startAddr, startRow,i,savedRows)
            else:
                res = self.serializeGraphData(partitions[i].tocsc(), graphName + "-" + str(i),
                                              targetDir, startAddr, startRow,i,savedRows)
            startRow += partitions[i].shape[0]
            startAddr = res[1]
            savedRows = res[0]
      
            # i += 1
        configFile.close()
        
        # add copy command for the config file
        print ("Graph " + graphName + " prepared with " + str(partitionCount) + " partitions")
        if csr:
            print("Matrix stored in row-major format")
        else:
            print ("Matrix stored in col-major format")
        print ("All data is located in " + targetDir + "\n")

        self.graphName = graphName


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


# Slice matrix along rows to create desired number of partitions
# Keeps number of rows equal, does not look at evenness of NZ distribution
def horizontalSplit(matrix, numPartitions):
    csr = loadGraph(matrix).tocsr()
    print(csr)
    print ('Test=================================================================')
    submatrices = []
    # align partition size (rowcount) to 64 - good for bursts and
    # updating the input vector bits independently
    stepSize = int(alignedIncrement(csr.shape[0] / numPartitions,0,64))
    print(csr.shape[0])
    print(numPartitions)
    print(stepSize)
    print("=======================\n")
    # create submatrices, last one includes the remainder
    currentRow = int(0)
    for i in range(0, numPartitions):
        if i != numPartitions - 1:
            submatrices += [csr[currentRow:currentRow + stepSize]]  # essentially this is initally csr[0:0+stepsize] = csr[0:stepsize]  
        else:
            submatrices += [csr[currentRow:]]
        currentRow += stepSize
    return submatrices


def createVerifyCommand(fileName, startAddr):
    addrString = ("0x%0.8X" % startAddr)
    cmd = "data_verify " + fileName + " " + addrString + "\n"
    return cmd


# increment base address by <increment> and ensure alignment to <align>
def alignedIncrement(base, increment, align):
    res = base + increment
    rem = res % align
    if rem != 0:
        res += align - rem
    return res










#################### Generate ROOT Nodes #################
def buildRootNodes(cnt):
    graphs = makeGraphList()
    rnl = dict()
    for g in graphs:
        print("Generating root nodes for " + str(g) + "\n===========================\n")
        rnl[g] = generateRootNodes(g, cnt, 5)
    return rnl

def buildRootNodesSingle(name,cnt):
    rnl = dict()
    g=name
    print("Generating root nodes for " + str(g) + "\n===========================\n")
    rnl[g] = generateRootNodes(g, cnt, 5)
    return rnl

def generateRootNodes(graph, desiredNumRootNodes, bfsDepthMin):
  
    A = loadGraph(graph)
    
    # seed the random number generator
    rnd.seed()
    rootNodes = []

    while(len(rootNodes) != desiredNumRootNodes):
        rootCandidate = rnd.randint(0, 2048)
        bfsDepth = getBFSDepth(A, rootCandidate)
        if(bfsDepth >= bfsDepthMin):  
            rootNodes += [rootCandidate]
    return rootNodes

def getBFSDepth(graph, root):
    l = ssspUnweighted(graph, root)
    l = filter(lambda x: x != inf, l)
    return max(l)

def ssspUnweighted(graph, root):
    A = loadGraph(graph)
    return scipy.sparse.csgraph.dijkstra(A, indices=root, unweighted=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if(sys.argv[1] == "all"):
        num_partition = int(sys.argv[2])
        buildGraphManager(num_partition)
        if(sys.argv[2] == "genrootnodes"):
            emp_num = dict()
            emp_num = buildRootNodes(16)
            print(emp_num)
    else:
        num_partition = int(sys.argv[2])
        buildGraphManagerSingle(sys.argv[1],num_partition)
        emp_num = dict()
        emp_num = buildRootNodesSingle(sys.argv[1],1)
        print(emp_num)
