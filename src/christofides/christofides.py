import networkx as nx

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