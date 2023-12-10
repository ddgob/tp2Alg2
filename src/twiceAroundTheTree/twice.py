import networkx as nx

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