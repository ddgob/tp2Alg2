import networkx as nx
import numpy as np
import os
import re
import time
import multiprocessing
import signal
import psutil

def TSPChristofides(graphAdjacencyMatrix):
    numVerts = graphAdjacencyMatrix.shape[0]
    G = nx.Graph()

    for i in range(numVerts):
        for j in range(i + 1, numVerts):
            G.add_edge(i, j, weight=graphAdjacencyMatrix[i][j])

    minSpanTree = nx.minimum_spanning_tree(G)

    oddDegreeNodes = []
    for vert, deg in minSpanTree.degree():
        if deg % 2 != 0:
            oddDegreeNodes.append(vert)
    oddSubgraph = G.subgraph(oddDegreeNodes)

    minWeightMatching = nx.min_weight_matching(oddSubgraph)

    minSpanTreeDupEdges = nx.MultiGraph(minSpanTree)
    minSpanTreeDupEdges.add_edges_from(minWeightMatching)

    eulerianCircuit = list(nx.eulerian_circuit(minSpanTreeDupEdges, source=0))

    visitedVerts = set()
    sol = [eulerianCircuit[0][0]]
    for _, vert in eulerianCircuit:
        if vert not in visitedVerts:
            sol.append(vert)
            visitedVerts.add(vert)

    best = 0
    for i in range(numVerts):
        best += graphAdjacencyMatrix[sol[i]][sol[i + 1]]

    return best, sol

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

def runTspWithTimeout(graphAdjacencyMatrix, timeout, resultQueue):
    def handler(signum, frame):
        raise TimeoutError("Function execution timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        best, sol = TSPChristofides(graphAdjacencyMatrix)
        resultQueue.put((best, sol, None))
    except Exception as e:
        resultQueue.put((None, None, e))
    finally:
        signal.alarm(0)  # Cancel the alarm

def processTspInstance(tspInstance, outputFile, tspFolderPath):
    
    tspFilePath = os.path.join(tspFolderPath, tspInstance, tspInstance + ".tsp")
    startReadDataTime = time.time()
    graphAdjacencyMatrix = readTspData(tspFilePath)
    endReadDataTime = time.time()
    durationReadDataTime = endReadDataTime - startReadDataTime

    outputFile.write(f"Time taken to read the file: {durationReadDataTime} seconds\n")
    outputFile.write("-----------\n")

    startMemoryUsageInKB = psutil.Process(os.getpid()).memory_info().rss / 1024
    startAlgorithmExecuteTime = time.time()

    # Set the timeout to 30 minutes (1800 seconds)
    algorithmExecutionTimeLimitSeconds = 1800

    resultQueue = multiprocessing.Queue()
    process = multiprocessing.Process(target=runTspWithTimeout, args=(graphAdjacencyMatrix, algorithmExecutionTimeLimitSeconds, resultQueue))
    process.start()
    process.join(algorithmExecutionTimeLimitSeconds)  # Wait for the process to finish or timeout

    if process.is_alive():
        # If the process is still running, terminate it
        process.terminate()
        process.join()
        outputFile.write("Time limit reached. No solution found.\n")
        endAlgorithmExecuteTime = time.time()
        endMemoryUsageInKB = psutil.Process(os.getpid()).memory_info().rss / 1024
        durationAlgorithmExecutionTime = endAlgorithmExecuteTime - startAlgorithmExecuteTime
        memoryUsage = endMemoryUsageInKB - startMemoryUsageInKB

        outputFile.write(f"Time taken to execute: {durationAlgorithmExecutionTime} seconds\n")
        outputFile.write(f"Memory taken to execute: {memoryUsage} kilobytes\n")
        return "Time limit reached. No solution found."

    bestCostFound, bestSolutionFound, error = resultQueue.get()

    endAlgorithmExecuteTime = time.time()
    endMemoryUsageInKB = psutil.Process(os.getpid()).memory_info().rss / 1024
    durationAlgorithmExecutionTime = endAlgorithmExecuteTime - startAlgorithmExecuteTime
    memoryUsage = endMemoryUsageInKB - startMemoryUsageInKB

    outputFile.write(f"Time taken to execute: {durationAlgorithmExecutionTime} seconds\n")
    outputFile.write(f"Memory taken to execute: {memoryUsage} kilobytes\n")

    if error:
        outputFile.write(f"An error occurred: {error}\n")
        return "!!!!!!!!!!!!!!An error occurred!!!!!!!!!!!!!!"

    outputFile.write(f"Best cost: {bestCostFound}\n")
    outputFile.write(f"Solution path: {bestSolutionFound}\n")

    return "Successfully processed"

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
outputFolderPath = "./outputs/christofides"
runAllInstancesInOrderOfNodes(tspFolderPath, outputFolderPath)