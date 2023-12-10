import networkx as nx
import numpy as np
import os
import re

def TSPTwiceAroundTheTree(graphAdjacencyMatrix):
    
    numVerts = graphAdjacencyMatrix.shape[0]
    G = nx.Graph()

    for i in range(numVerts):
        for j in range(i + 1, numVerts):
            G.add_edge(i, j, weight=graphAdjacencyMatrix[i][j])

    minSpanTree = nx.minimum_spanning_tree(G)

    sol = list(nx.dfs_preorder_nodes(minSpanTree, 0))
    sol.append(sol[0])

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