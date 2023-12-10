from numba import njit
import numpy as np
import time
import heapq

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

    best = float('inf')
    sol = []
    
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