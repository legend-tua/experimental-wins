#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

using namespace std;

// Structure to represent a single cell in the grid
struct Node {
    int x, y;              // Coordinates
    float gCost, hCost;    // Cost from start and heuristic cost to goal
    Node* parent;          // Pointer to the parent node (for path reconstruction)

    Node(int x, int y, Node* parent = nullptr)
        : x(x), y(y), gCost(0), hCost(0), parent(parent) {}

    float fCost() const {
        return gCost + hCost;
    }

    // Comparator for priority queue (min-heap based on fCost)
    bool operator>(const Node& other) const {
        return fCost() > other.fCost();
    }
};

// Function to calculate Manhattan Distance (heuristic)
float calculateHeuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

// Check if a cell is valid (within grid bounds)
bool isValidCell(int x, int y, int rows, int cols) {
    return (x >= 0 && x < rows && y >= 0 && y < cols);
}

// Retrieve path from goal node by traversing parents
vector<pair<int, int>> reconstructPath(Node* node) {
    vector<pair<int, int>> path;
    while (node != nullptr) {
        path.emplace_back(node->x, node->y);
        node = node->parent;
    }
    reverse(path.begin(), path.end());
    return path;
}

// A* Search Algorithm
vector<pair<int, int>> aStarSearch(
    vector<vector<int>>& grid,
    pair<int, int> start,
    pair<int, int> goal
) {
    int rows = grid.size();
    int cols = grid[0].size();

    vector<vector<bool>> closedList(rows, vector<bool>(cols, false));
    vector<vector<Node*>> allNodes(rows, vector<Node*>(cols, nullptr));

    // Min-heap priority queue for open nodes (based on lowest fCost)
    priority_queue<Node, vector<Node>, greater<Node>> openList;

    // Initialize start node
    Node startNode(start.first, start.second);
    startNode.gCost = 0;
    startNode.hCost = calculateHeuristic(start.first, start.second, goal.first, goal.second);

    openList.push(startNode);

    // 4 possible movement directions (up, down, left, right)
    vector<pair<int, int>> directions = {
        { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 }
    };

    while (!openList.empty()) {
        Node current = openList.top();
        openList.pop();

        int x = current.x;
        int y = current.y;

        if (closedList[x][y])
            continue;

        closedList[x][y] = true;

        // Goal reached
        if (x == goal.first && y == goal.second) {
            return reconstructPath(&current);
        }

        // Explore neighbors
        for (auto [dx, dy] : directions) {
            int nx = x + dx;
            int ny = y + dy;

            if (!isValidCell(nx, ny, rows, cols) || grid[nx][ny] == 1 || closedList[nx][ny])
                continue;

            float tentativeG = current.gCost + 1; // Cost between adjacent nodes

            if (allNodes[nx][ny] == nullptr) {
                allNodes[nx][ny] = new Node(nx, ny, new Node(current));
            }

            Node* neighbor = allNodes[nx][ny];
            if (tentativeG < neighbor->gCost || neighbor->parent == nullptr) {
                neighbor->gCost = tentativeG;
                neighbor->hCost = calculateHeuristic(nx, ny, goal.first, goal.second);
                neighbor->parent = new Node(current);
                openList.push(*neighbor);
            }
        }
    }

    // Return empty path if no route found
    return {};
}

// Driver function
int main() {
    // 0 = open cell, 1 = blocked cell
    vector<vector<int>> grid = {
        { 0, 0, 0, 0, 0 },
        { 1, 1, 0, 1, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 1, 1, 1, 0 },
        { 0, 0, 0, 0, 0 }
    };

    pair<int, int> start = { 0, 0 };
    pair<int, int> goal = { 4, 4 };

    auto path = aStarSearch(grid, start, goal);

    if (!path.empty()) {
        cout << "Path found:\n";
        for (auto [x, y] : path)
            cout << "(" << x << ", " << y << ") -> ";
        cout << "GOAL\n";
    } else {
        cout << "No path found.\n";
    }

    return 0;
}
