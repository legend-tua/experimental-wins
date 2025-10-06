/**
 * A* Pathfinding Algorithm Implementation
 * 
 * A* is an informed search algorithm that finds the shortest path between nodes in a graph.
 * It uses a heuristic function to guide the search towards the goal efficiently.
 * 
 * Time Complexity: O(E log V) where E is number of edges and V is number of vertices
 * Space Complexity: O(V) for storing nodes in open and closed sets
 */

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// Represents a position in 2D grid
struct Position {
    int x;
    int y;
    
    Position(int xCoord = 0, int yCoord = 0) : x(xCoord), y(yCoord) {}
    
    // Operator overloading for using Position in map and set
    bool operator<(const Position& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

// Node structure for A* algorithm
struct Node {
    Position position;           // Current position
    double gCost;               // Cost from start to current node
    double hCost;               // Heuristic cost from current node to goal
    double fCost;               // Total cost (g + h)
    Position parent;            // Parent node position for path reconstruction
    
    Node() : position(), gCost(0), hCost(0), fCost(0), parent() {}
    
    Node(Position pos, double g, double h, Position par) 
        : position(pos), gCost(g), hCost(h), fCost(g + h), parent(par) {}
    
    // Comparator for priority queue (min-heap based on fCost)
    bool operator>(const Node& other) const {
        return fCost > other.fCost;
    }
};

class AStarPathfinder {
private:
    vector<vector<int>> grid;           // 0 = walkable, 1 = obstacle
    int rows;
    int cols;
    
    // Possible movement directions: up, down, left, right, and diagonals
    vector<pair<int, int>> directions = {
        {-1, 0},  // Up
        {1, 0},   // Down
        {0, -1},  // Left
        {0, 1},   // Right
        {-1, -1}, // Up-Left
        {-1, 1},  // Up-Right
        {1, -1},  // Down-Left
        {1, 1}    // Down-Right
    };
    
    /**
     * Calculate heuristic cost using Euclidean distance
     * This is admissible and consistent, making A* optimal
     */
    double calculateHeuristic(const Position& current, const Position& goal) {
        int dx = abs(current.x - goal.x);
        int dy = abs(current.y - goal.y);
        return sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Calculate movement cost between two adjacent positions
     * Diagonal movement costs sqrt(2), orthogonal movement costs 1
     */
    double calculateMovementCost(const Position& from, const Position& to) {
        int dx = abs(to.x - from.x);
        int dy = abs(to.y - from.y);
        
        // Diagonal movement
        if (dx == 1 && dy == 1) {
            return 1.414; // sqrt(2)
        }
        // Orthogonal movement
        return 1.0;
    }
    
    /**
     * Check if a position is valid (within grid bounds and not an obstacle)
     */
    bool isValidPosition(const Position& pos) {
        return pos.x >= 0 && pos.x < rows && 
               pos.y >= 0 && pos.y < cols && 
               grid[pos.x][pos.y] == 0;
    }
    
    /**
     * Get all valid neighboring positions
     */
    vector<Position> getNeighbors(const Position& current) {
        vector<Position> neighbors;
        
        for (const auto& dir : directions) {
            Position neighbor(current.x + dir.first, current.y + dir.second);
            
            if (isValidPosition(neighbor)) {
                // For diagonal movement, check if adjacent cells are not blocked
                if (abs(dir.first) == 1 && abs(dir.second) == 1) {
                    Position adjacent1(current.x + dir.first, current.y);
                    Position adjacent2(current.x, current.y + dir.second);
                    
                    if (isValidPosition(adjacent1) || isValidPosition(adjacent2)) {
                        neighbors.push_back(neighbor);
                    }
                } else {
                    neighbors.push_back(neighbor);
                }
            }
        }
        
        return neighbors;
    }
    
    /**
     * Reconstruct the path from start to goal using parent pointers
     */
    vector<Position> reconstructPath(const map<Position, Position>& parentMap, 
                                     const Position& start, const Position& goal) {
        vector<Position> path;
        Position current = goal;
        
        // Backtrack from goal to start
        while (!(current == start)) {
            path.push_back(current);
            current = parentMap.at(current);
        }
        path.push_back(start);
        
        // Reverse to get path from start to goal
        reverse(path.begin(), path.end());
        return path;
    }

public:
    /**
     * Constructor
     * @param inputGrid 2D grid where 0 = walkable, 1 = obstacle
     */
    AStarPathfinder(const vector<vector<int>>& inputGrid) 
        : grid(inputGrid), rows(inputGrid.size()), cols(inputGrid[0].size()) {}
    
    /**
     * Find the shortest path from start to goal using A* algorithm
     * 
     * @param start Starting position
     * @param goal Goal position
     * @return Vector of positions representing the path (empty if no path exists)
     */
    vector<Position> findPath(const Position& start, const Position& goal) {
        // Validate start and goal positions
        if (!isValidPosition(start) || !isValidPosition(goal)) {
            cout << "Invalid start or goal position!" << endl;
            return vector<Position>();
        }
        
        // Priority queue for open set (nodes to be evaluated)
        priority_queue<Node, vector<Node>, greater<Node>> openSet;
        
        // Set of evaluated nodes
        set<Position> closedSet;
        
        // Map to store g-costs for each position
        map<Position, double> gCostMap;
        
        // Map to store parent positions for path reconstruction
        map<Position, Position> parentMap;
        
        // Initialize start node
        Node startNode(start, 0, calculateHeuristic(start, goal), start);
        openSet.push(startNode);
        gCostMap[start] = 0;
        parentMap[start] = start;
        
        // Main A* loop
        while (!openSet.empty()) {
            // Get node with lowest f-cost
            Node current = openSet.top();
            openSet.pop();
            
            // Skip if already evaluated
            if (closedSet.count(current.position)) {
                continue;
            }
            
            // Mark current node as evaluated
            closedSet.insert(current.position);
            
            // Check if goal is reached
            if (current.position == goal) {
                cout << "Path found!" << endl;
                return reconstructPath(parentMap, start, goal);
            }
            
            // Explore neighbors
            vector<Position> neighbors = getNeighbors(current.position);
            
            for (const Position& neighborPos : neighbors) {
                // Skip if already evaluated
                if (closedSet.count(neighborPos)) {
                    continue;
                }
                
                // Calculate tentative g-cost
                double movementCost = calculateMovementCost(current.position, neighborPos);
                double tentativeGCost = current.gCost + movementCost;
                
                // Check if this path to neighbor is better
                if (gCostMap.find(neighborPos) == gCostMap.end() || 
                    tentativeGCost < gCostMap[neighborPos]) {
                    
                    // Update costs and parent
                    gCostMap[neighborPos] = tentativeGCost;
                    parentMap[neighborPos] = current.position;
                    
                    // Calculate heuristic and create neighbor node
                    double hCost = calculateHeuristic(neighborPos, goal);
                    Node neighborNode(neighborPos, tentativeGCost, hCost, current.position);
                    
                    // Add to open set
                    openSet.push(neighborNode);
                }
            }
        }
        
        // No path found
        cout << "No path found!" << endl;
        return vector<Position>();
    }
    
    /**
     * Print the grid with the path marked
     */
    void printGridWithPath(const vector<Position>& path) {
        // Create a copy of the grid for display
        vector<vector<char>> displayGrid(rows, vector<char>(cols));
        
        // Fill display grid
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                displayGrid[i][j] = (grid[i][j] == 1) ? '#' : '.';
            }
        }
        
        // Mark path
        for (size_t i = 0; i < path.size(); i++) {
            if (i == 0) {
                displayGrid[path[i].x][path[i].y] = 'S'; // Start
            } else if (i == path.size() - 1) {
                displayGrid[path[i].x][path[i].y] = 'G'; // Goal
            } else {
                displayGrid[path[i].x][path[i].y] = '*'; // Path
            }
        }
        
        // Print grid
        cout << "\nGrid with path:\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << displayGrid[i][j] << " ";
            }
            cout << endl;
        }
    }
};

// Main function with example usage
int main() {
    // Create a sample grid (0 = walkable, 1 = obstacle)
    vector<vector<int>> grid = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 1, 0, 1, 0},
        {0, 0, 1, 0, 1, 0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0, 0, 1, 0, 1, 0},
        {0, 0, 0, 1, 1, 0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 1, 1, 1, 0, 1, 1, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
    
    // Create pathfinder instance
    AStarPathfinder pathfinder(grid);
    
    // Define start and goal positions
    Position start(0, 0);
    Position goal(9, 9);
    
    cout << "A* Pathfinding Algorithm Demo" << endl;
    cout << "=============================" << endl;
    cout << "Start: (" << start.x << ", " << start.y << ")" << endl;
    cout << "Goal: (" << goal.x << ", " << goal.y << ")" << endl;
    cout << "\nFinding path...\n" << endl;
    
    // Find path
    vector<Position> path = pathfinder.findPath(start, goal);
    
    // Display results
    if (!path.empty()) {
        cout << "\nPath length: " << path.size() << " steps" << endl;
        cout << "\nPath coordinates:" << endl;
        for (size_t i = 0; i < path.size(); i++) {
            cout << "Step " << i << ": (" << path[i].x << ", " << path[i].y << ")" << endl;
        }
        
        pathfinder.printGridWithPath(path);
    }
    
    return 0;
}
