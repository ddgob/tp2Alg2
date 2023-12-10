#include <vector>
#include <cmath>
#include <chrono>
#include <queue>
#include <iostream>
#include <algorithm>

double bound(const std::vector<int>& solution, const std::vector<std::vector<double>>& graphAdjacencyMatrix) {
    double totalBound = 0;
    int numVerts = graphAdjacencyMatrix.size();
    int solutionSize = solution.size();

    for (int i = 0; i < numVerts; ++i) {
        double smallestEdge = std::numeric_limits<double>::infinity();
        double secondSmallestEdge = std::numeric_limits<double>::infinity();
        bool oneEdgeFound = false;
        bool bothEdgesFound = false;

        if (solutionSize >= 2) {
            for (int k = 0; k < solutionSize; ++k) {
                if (i == solution[k]) {
                    if (k != 0) {
                        secondSmallestEdge = graphAdjacencyMatrix[i][solution[k - 1]];
                        bothEdgesFound = true;
                    }
                    if (k != solutionSize - 1) {
                        smallestEdge = graphAdjacencyMatrix[i][solution[k + 1]];
                        oneEdgeFound = true;
                    }
                    if (bothEdgesFound && !oneEdgeFound) {
                        smallestEdge = secondSmallestEdge;
                        bothEdgesFound = false;
                        oneEdgeFound = true;
                    }
                    break;
                }
            }
        }

        if (!bothEdgesFound) {
            for (int j = 0; j < numVerts; ++j) {
                if (i != j) {
                    if (graphAdjacencyMatrix[i][j] < smallestEdge && !oneEdgeFound) {
                        secondSmallestEdge = smallestEdge;
                        smallestEdge = graphAdjacencyMatrix[i][j];
                    } else if (graphAdjacencyMatrix[i][j] < secondSmallestEdge) {
                        secondSmallestEdge = graphAdjacencyMatrix[i][j];
                    }
                }
            }
        }

        totalBound += smallestEdge + secondSmallestEdge;
    }

    return std::ceil(totalBound / 2);
}

class Node {
public:
    int level;
    double cost;
    std::vector<int> s;
    double bound;

    Node(int level, double cost, const std::vector<int>& s, double bound)
        : level(level), cost(cost), s(s), bound(bound) {}

    bool operator<(const Node& other) const {
        return bound < other.bound;
    }
};

std::tuple<double, std::vector<int>, bool, int> TSPBranchAndBoundOriginal(const std::vector<std::vector<double>>& graphAdjacencyMatrix, int numNodes, int timeLimitMinutes = 30) {
    std::clock_t startTime = std::clock();
    double timeLimitSeconds = timeLimitMinutes * 60.0;
    Node root(1, 0.0, {0}, bound({0}, graphAdjacencyMatrix));
    std::priority_queue<Node> queue;
    queue.push(root);

    double best = std::numeric_limits<double>::infinity();
    std::vector<int> sol;

    bool leafNodeReached = false;
    int cutsCount = 0;

    while (!queue.empty()) {
        if ((std::clock() - startTime) / static_cast<double>(CLOCKS_PER_SEC) > timeLimitSeconds) {
            std::cout << "Time limit reached" << std::endl;
            break;
        }

        Node node = queue.top();
        queue.pop();

        if (node.level > numNodes) {
            if (best > node.cost) {
                best = node.cost;
                sol = node.s;
            }
            leafNodeReached = true;
        } else if (node.bound < best) {
            if (node.level < numNodes) {
                for (int k = 0; k < numNodes; ++k) {
                    if (std::find(node.s.begin(), node.s.end(), k) == node.s.end() && graphAdjacencyMatrix[node.s.back()][k] != std::numeric_limits<double>::infinity() && bound(node.s, graphAdjacencyMatrix) < best) {
                        std::vector<int> new_s = node.s;
                        new_s.push_back(k);
                        queue.push(Node(node.level + 1, node.cost + graphAdjacencyMatrix[node.s.back()][k], new_s, bound(new_s, graphAdjacencyMatrix)));
                    }
                }
            } else if (graphAdjacencyMatrix[node.s.back()][0] != std::numeric_limits<double>::infinity() && bound(node.s, graphAdjacencyMatrix) < best) {
                if (std::all_of(node.s.begin(), node.s.end(), [numNodes](int i) { return i >= 0 && i < numNodes; })) {
                    std::vector<int> new_s = node.s;
                    new_s.push_back(0);
                    queue.push(Node(node.level + 1, node.cost + graphAdjacencyMatrix[node.s.back()][0], new_s, bound(new_s, graphAdjacencyMatrix)));
                }
            }
        } else if (node.bound >= best) {
            cutsCount++;
        }
    }

    return std::make_tuple(best, sol, leafNodeReached, cutsCount);
}