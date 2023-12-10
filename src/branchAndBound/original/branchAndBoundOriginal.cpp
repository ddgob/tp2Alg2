#include <vector>
#include <cmath>
#include <chrono>
#include <queue>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>
#include <regex>
#include <sys/resource.h>

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

std::vector<std::vector<double>> readTspData(const std::string& tspFilePath) {
    std::ifstream tspFile(tspFilePath);
    std::vector<std::vector<double>> graphAdjacencyMatrix;

    if (tspFile.is_open()) {
        std::vector<std::pair<double, double>> coordinates;
        std::string line;

        while (std::getline(tspFile, line)) {
            if (line == "NODE_COORD_SECTION") {
                while (std::getline(tspFile, line) && line != "EOF") {
                    std::istringstream iss(line);
                    double nodeID, xCoord, yCoord;
                    iss >> nodeID >> xCoord >> yCoord;
                    coordinates.emplace_back(xCoord, yCoord);
                }
                break;
            }
        }

        int size = coordinates.size();
        graphAdjacencyMatrix.resize(size, std::vector<double>(size, 0.0));

        // Calculate euclidian distance
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double diffX = coordinates[i].first - coordinates[j].first;
                double diffY = coordinates[i].second - coordinates[j].second;
                graphAdjacencyMatrix[i][j] = std::sqrt(diffX * diffX + diffY * diffY);
            }
        }

        tspFile.close();
    }

    return graphAdjacencyMatrix;
}

std::vector<std::string> extractFolderNamesInDirectory(const std::string& directory) {
    std::vector<std::string> folderNames;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (std::filesystem::is_directory(entry)) {
            folderNames.push_back(entry.path().filename().string());
        }
    }
    return folderNames;
}

std::map<int, std::string> tspInstanceToNumOfNodesMapper(const std::vector<std::string>& tspInstances) {
    std::map<int, std::string> mapResult;
    std::regex pattern("\\d+");

    for (const auto& tspInstance : tspInstances) {
        std::sregex_iterator currentMatch(tspInstance.begin(), tspInstance.end(), pattern);
        std::sregex_iterator lastMatch;

        while (currentMatch != lastMatch) {
            std::smatch match = *currentMatch;
            mapResult[std::stoi(match.str())] = tspInstance;
            currentMatch++;
        }
    }

    return mapResult;
}

std::string processTspInstance(const std::string& tspInstance, std::ofstream& outputFile, const std::string& tspFolderPath) {
    try {
        std::string tspFilePath = tspFolderPath + "/" + tspInstance + "/" + tspInstance + ".tsp";
        std::clock_t startReadDataTime = std::clock();
        std::vector<std::vector<double>> graphAdjacencyMatrix = readTspData(tspFilePath);
        std::clock_t endReadDataTime = std::clock();
        
        double durationReadDataTime = static_cast<double>(endReadDataTime - startReadDataTime) / CLOCKS_PER_SEC;
        outputFile << "Time taken to read the file: " << durationReadDataTime << " seconds" << std::endl;
        outputFile << "-----------" << std::endl;

        struct rusage startMemoryUsageInKB, endMemoryUsageInKB;
        getrusage(RUSAGE_SELF, &startMemoryUsageInKB);
        std::clock_t startAlgorithmExecuteTime = std::clock();
        std::tuple<double, std::vector<int>, bool, int> result = TSPBranchAndBoundOriginal(graphAdjacencyMatrix, graphAdjacencyMatrix.size(), 30);
        std::clock_t endAlgorithmExecuteTime = std::clock();
        getrusage(RUSAGE_SELF, &endMemoryUsageInKB);
        double durationAlgorithmExecutionTime = static_cast<double>(endAlgorithmExecuteTime - startAlgorithmExecuteTime) / CLOCKS_PER_SEC;
        long memoryUsage = endMemoryUsageInKB.ru_maxrss - startMemoryUsageInKB.ru_maxrss;
        
        // Write the best cost and solution path to the output file
        outputFile << "Time taken to execute: " << durationAlgorithmExecutionTime << " seconds" << std::endl;
        outputFile << "Memory taken to execute: " << memoryUsage << " kilobytes" << std::endl;
        outputFile << "Best cost: " << std::get<0>(result) << std::endl;
        outputFile << "Cuts count: " << std::get<3>(result) << std::endl;
        if (std::get<2>(result)) {
            outputFile << "Leaf node was reached" << std::endl;
        } else {
            outputFile << "Leaf node was NOT reached" << std::endl;
        }
        outputFile << "Solution path: ";
        for (int i : std::get<1>(result)) {

            outputFile << i << " ";

        }
        outputFile << std::endl;

        return "Successfully processed";
    }
    catch (const std::exception& error) {
        outputFile << "An error occurred: " << error.what() << std::endl;
        return "!!!!!!!!!!!!!!An error occurred!!!!!!!!!!!!!!";
    }
}

void runSingleTspInstance(const std::string& tspInstance, const std::string& tspFolderPath, std::ofstream& outputFile) {
    std::cout << "Started folder: " << tspInstance << " ..." << std::endl;
    outputFile << "--------------------- Folder: " << tspInstance << " ---------------------" << std::endl;
    std::string processTspInstanceErrorMessage = processTspInstance(tspInstance, outputFile, tspFolderPath);
    outputFile << "\n\n\n\n" << std::endl;
    std::cout << "Finished folder: " << tspInstance << " --> " << processTspInstanceErrorMessage << std::endl;
}

void runAllInstancesInOrderOfNodes(const std::string& tspFolderPath, const std::string& outputFolderPath) {
    auto tspInstances = extractFolderNamesInDirectory(tspFolderPath);
    auto mapTspInstancesToNumOfNodes = tspInstanceToNumOfNodesMapper(tspInstances);

    for (const auto& pair : mapTspInstancesToNumOfNodes) {
        int numberOfNodesLastInstanceExecuted = 0;
        if (pair.first > numberOfNodesLastInstanceExecuted) {
            std::string tspInstance = pair.second;
            std::ofstream outputFile(outputFolderPath + "/" + tspInstance + ".txt");
            runSingleTspInstance(tspInstance, tspFolderPath, outputFile);
        }
    }
}

int main() {
    std::string tspFolderPath = "./data/test";
    std::string outputFold = "./outputs/branchAndBound/original/c++";
    runAllInstancesInOrderOfNodes(tspFolderPath, outputFold);
    return 0;
}