from numba import njit
import numpy as np
import time
import heapq
import os
import re
import psutil

@njit
def bound(graphAdjacencyMatrix, solution):

    totalBound = 0
    numVerts = graphAdjacencyMatrix.shape[0]
    solutionSize = len(solution)
    
    if solutionSize > 1:
        
        for i in range(solutionSize - 1):
            totalBound += graphAdjacencyMatrix[solution[i], solution[i + 1]] * 2

        if solutionSize == numVerts:
            return int(np.ceil((totalBound + (graphAdjacencyMatrix[solution[-1], solution[0]] * 2)) / 2))
        
        allVertsExceptTheOneAfter0InThePath = np.array([i for i in range(numVerts) if i != solution[1]])
        secondEdgeCostForZero = np.partition(graphAdjacencyMatrix[0, allVertsExceptTheOneAfter0InThePath], 1)[0]
        totalBound += secondEdgeCostForZero
        
        allVertsExceptTheOneBeforeTheLastVertInThePath = np.array([i for i in range(numVerts) if i != solution[solutionSize - 2]])
        secondEdgeCostForLastVertInPath = np.partition(graphAdjacencyMatrix[0, allVertsExceptTheOneBeforeTheLastVertInThePath], 1)[0]
        totalBound += secondEdgeCostForLastVertInPath
    
    else:
        totalBound += np.sum(np.partition(graphAdjacencyMatrix[0], 1)[:2])

    vertsNotInSolution = np.array([i for i in range(numVerts) if i not in solution])

    for i in vertsNotInSolution:

        smallestEdges = np.partition(graphAdjacencyMatrix[i], 1)[:2]
        totalBound += np.sum(smallestEdges)

    return int(np.ceil(totalBound / 2))

class Node:
    def __init__(self, bound, level, cost, s):
        self.level = level
        self.cost = cost
        self.s = s
        self.bound = bound

    def __lt__(self, other):
        return self.bound < other.bound

    def __repr__(self):
        return f"Node(bound={self.bound}, level={self.level}, cost={self.cost}, s={self.s})"


def TSPBranchAndBoundOriginal(graphAdjacencyMatrix, numNodes, timeLimitMinutes=30):
    
    startTime = time.time()
    timeLimitSeconds = timeLimitMinutes * 60

    root = Node(bound(graphAdjacencyMatrix, [0]), 1, 0, [0])
    queue = [root]

    best = 0

    # Create a trivial solution: visit nodes in order 0, 1, 2, ..., n-1, 0
    sol = list(range(numNodes)) + [0]

    # Calculate the cost of the trivial solution
    for i in range(numNodes):
        best += graphAdjacencyMatrix[sol[i]][sol[i+1]]
    
    leafNodeReached = False
    cutsCount = 0

    while queue:
        executionDuration = time.time() - startTime
        if executionDuration > timeLimitSeconds:
            break
        node = heapq.heappop(queue)
        if node.level > numNodes:
            if best > node.cost:
                best = node.cost
                sol = node.s
            leafNodeReached = True
        elif node.bound < best:
            if node.level < numNodes:
                for k in range(numNodes):
                    if k not in node.s and graphAdjacencyMatrix[node.s[-1]][k] != np.inf and bound(graphAdjacencyMatrix, node.s + [k]) < best:
                        heapq.heappush(queue, Node(bound(graphAdjacencyMatrix, node.s + [k]), node.level + 1, node.cost + graphAdjacencyMatrix[node.s[-1]][k], node.s + [k]))
            elif graphAdjacencyMatrix[node.s[-1]][0] != np.inf and bound(graphAdjacencyMatrix, node.s + [0]) < best:
                if all(i in node.s for i in range(numNodes)):
                    heapq.heappush(queue, Node(bound(graphAdjacencyMatrix, node.s + [0]), node.level + 1, node.cost + graphAdjacencyMatrix[node.s[-1]][0], node.s + [0]))
        elif node.bound >= best:
            cutsCount += 1

    return best, sol, leafNodeReached, cutsCount


def readTspData(tspFilePath):

    with open(tspFilePath, 'r') as tspFile:
        coordinates = []
        for line in tspFile:
            lineStrip = line.strip()
            if lineStrip == 'NODE_COORD_SECTION':
                for line in tspFile:
                    lineStrip = line.strip()
                    if lineStrip == 'EOF':
                        break
                    nodeID, xCoord, yCoord = lineStrip.split()
                    coordinates.append((float(xCoord), float(yCoord)))
                break

    coordinatesArray = np.array(coordinates)

    # Calculate euclidian distance
    difference = coordinatesArray[:, np.newaxis, :] - coordinatesArray[np.newaxis, :, :]
    graphAdjacencyMatrix = np.sqrt(np.sum(difference**2, axis=2))
    np.fill_diagonal(graphAdjacencyMatrix, np.inf)

    return graphAdjacencyMatrix

def extractFolderNamesInDirectory(directory):
    
    folderNames = []
    
    for entry in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, entry)):
            folderNames.append(entry)

    return folderNames

def tspInstanceToNumOfNodesMapper(tspInstances):
    
    mapResult = {}
    pattern = re.compile(r"\d+")

    for tspInstance in tspInstances:
        matches = pattern.findall(tspInstance)
        for match in matches:
            mapResult[int(match)] = tspInstance

    return mapResult

def processTspInstance(tspInstance, outputFile, tspFolderPath):
    
    try:

        tspFilePath = os.path.join(tspFolderPath, tspInstance, tspInstance + ".tsp")
        startReadDataTime = time.time()
        graphAdjacencyMatrix = readTspData(tspFilePath)
        endReadDataTime = time.time()
        durationReadDataTime = endReadDataTime - startReadDataTime
        
        outputFile.write(f"Time taken to read the file: {durationReadDataTime} seconds\n")
        outputFile.write("-----------\n")

        startMemoryUsageInKB = psutil.virtual_memory().used / 1024
        startAlgorithmExecuteTime = time.time()

        bestCostFound, bestSolutionFound, leafNodeReached, cutCounts = TSPBranchAndBoundOriginal(graphAdjacencyMatrix, graphAdjacencyMatrix.shape[0], 30)

        endAlgorithmExecuteTime = time.time()
        endMemoryUsageInKB = psutil.virtual_memory().used / 1024
        durationAlgorithmExecutionTime = endAlgorithmExecuteTime - startAlgorithmExecuteTime
        memoryUsage = endMemoryUsageInKB - startMemoryUsageInKB

        outputFile.write(f"Time taken to execute: {durationAlgorithmExecutionTime} seconds\n")
        outputFile.write(f"Memory taken to execute: {memoryUsage} kilobytes\n")
        outputFile.write(f"Best cost: {bestCostFound}\n")
        outputFile.write(f"Cut count = {cutCounts}\n")
        if leafNodeReached:
            outputFile.write(f"Leaf node was reached\n")
        else:
            outputFile.write(f"Leaf node was NOT reached\n")
        outputFile.write(f"Solution path: {bestSolutionFound}\n")

        return "Successfully processed"

    except Exception as error:

        outputFile.write(f"An error occurred: {error}\n")
        return "!!!!!!!!!!!!!!An error occurred!!!!!!!!!!!!!!"

def runSingleTspInstance(tspInstance, tspFolderPath, outputFile):
    
    print(f"Started folder: {tspInstance} ...")

    outputFile.write(f"--------------------- Folder: {tspInstance} ---------------------\n")
    processTspInstanceErrorMessage = processTspInstance(tspInstance, outputFile, tspFolderPath)
    outputFile.write("\n\n\n\n")

    print(f"Finished folder: {tspInstance} --> {processTspInstanceErrorMessage}")

def runAllInstancesInOrderOfNodes(tspFolderPath, outputFolderPath):
    
    tspInstances = extractFolderNamesInDirectory(tspFolderPath)
    mapTspInstancesToNumOfNodes = tspInstanceToNumOfNodesMapper(tspInstances)

    for numberOfNodesInstance, tspInstance in sorted(mapTspInstancesToNumOfNodes.items()):
        numberOfNodesLastInstanceExecuted = 0
        if numberOfNodesInstance > numberOfNodesLastInstanceExecuted:
            with open(os.path.join(outputFolderPath, tspInstance + ".txt"), "w") as outputFile:
                runSingleTspInstance(tspInstance, tspFolderPath, outputFile)

tspFolderPath = "./data/tsps"
outputFolderPath = "./outputs/branchAndBound/trivialSolution/python"
runAllInstancesInOrderOfNodes(tspFolderPath, outputFolderPath)