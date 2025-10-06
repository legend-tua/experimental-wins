/**
 * @file qStar.cpp
 * @brief Implementation of Q* (Q-Star) Algorithm
 * @details Q* combines Q-learning with A* search to find optimal paths in a grid-based environment
 *          while learning from experience. It uses both heuristic search and reinforcement learning.
 * @author GitHub Copilot
 * @date 2025-10-06
 */

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iomanip>

using namespace std;

// ============================
// Constants and Type Definitions
// ============================

const int GRID_SIZE = 10;
const double LEARNING_RATE = 0.1;      // Alpha: controls how much new information overrides old
const double DISCOUNT_FACTOR = 0.9;     // Gamma: importance of future rewards
const double EPSILON = 0.1;             // Exploration rate for epsilon-greedy policy
const int MAX_EPISODES = 1000;          // Maximum training episodes
const int MAX_STEPS_PER_EPISODE = 100;  // Maximum steps per episode

// Cell types in the grid
enum CellType {
    EMPTY = 0,
    OBSTACLE = 1,
    START = 2,
    GOAL = 3,
    PATH = 4
};

// Possible actions/movements
enum Action {
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,
    NUM_ACTIONS = 4
};

// ============================
// Data Structures
// ============================

/**
 * @struct Position
 * @brief Represents a 2D position in the grid
 */
struct Position {
    int row;
    int col;
    
    Position(int r = 0, int c = 0) : row(r), col(c) {}
    
    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }
    
    bool operator<(const Position& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

/**
 * @struct StateActionPair
 * @brief Represents a state-action pair for Q-value storage
 */
struct StateActionPair {
    Position state;
    Action action;
    
    StateActionPair(Position s, Action a) : state(s), action(a) {}
    
    bool operator<(const StateActionPair& other) const {
        if (state < other.state) return true;
        if (other.state < state) return false;
        return action < other.action;
    }
};

/**
 * @struct Node
 * @brief Node structure for A* priority queue
 */
struct Node {
    Position position;
    double gCost;           // Cost from start to current node
    double hCost;           // Heuristic cost from current node to goal
    double fCost;           // Total cost (g + h)
    double qValue;          // Q-value from Q-learning
    Position parent;
    
    Node(Position pos = Position(), double g = 0, double h = 0, double q = 0)
        : position(pos), gCost(g), hCost(h), qValue(q), parent(-1, -1) {
        fCost = gCost + hCost + qValue;  // Q* combines A* and Q-learning
    }
    
    bool operator>(const Node& other) const {
        return fCost > other.fCost;
    }
};

// ============================
// QStar Class
// ============================

/**
 * @class QStar
 * @brief Implementation of Q* algorithm combining Q-learning and A* search
 */
class QStar {
private:
    vector<vector<int>> grid;                    // Environment grid
    map<StateActionPair, double> qTable;         // Q-value table
    Position startPosition;
    Position goalPosition;
    
    /**
     * @brief Calculate Manhattan distance heuristic
     * @param from Starting position
     * @param to Target position
     * @return Manhattan distance between positions
     */
    double calculateHeuristic(const Position& from, const Position& to) const {
        return abs(from.row - to.row) + abs(from.col - to.col);
    }
    
    /**
     * @brief Check if a position is valid and not an obstacle
     * @param pos Position to check
     * @return true if position is valid, false otherwise
     */
    bool isValidPosition(const Position& pos) const {
        return pos.row >= 0 && pos.row < GRID_SIZE &&
               pos.col >= 0 && pos.col < GRID_SIZE &&
               grid[pos.row][pos.col] != OBSTACLE;
    }
    
    /**
     * @brief Get the next position based on current position and action
     * @param current Current position
     * @param action Action to take
     * @return New position after taking the action
     */
    Position getNextPosition(const Position& current, Action action) const {
        Position next = current;
        switch (action) {
            case UP:    next.row--; break;
            case DOWN:  next.row++; break;
            case LEFT:  next.col--; break;
            case RIGHT: next.col++; break;
            default: break;
        }
        return next;
    }
    
    /**
     * @brief Get reward for transitioning to a position
     * @param pos Position to evaluate
     * @return Reward value
     */
    double getReward(const Position& pos) const {
        if (!isValidPosition(pos)) return -10.0;          // Invalid move penalty
        if (pos == goalPosition) return 100.0;             // Goal reward
        if (grid[pos.row][pos.col] == OBSTACLE) return -10.0;  // Obstacle penalty
        return -1.0;                                       // Step penalty
    }
    
    /**
     * @brief Get Q-value for a state-action pair
     * @param state Current state/position
     * @param action Action to take
     * @return Q-value
     */
    double getQValue(const Position& state, Action action) const {
        StateActionPair pair(state, action);
        auto it = qTable.find(pair);
        return (it != qTable.end()) ? it->second : 0.0;
    }
    
    /**
     * @brief Update Q-value using Q-learning update rule
     * @param state Current state
     * @param action Action taken
     * @param reward Reward received
     * @param nextState Next state after taking action
     */
    void updateQValue(const Position& state, Action action, 
                     double reward, const Position& nextState) {
        // Find maximum Q-value for next state
        double maxNextQ = -numeric_limits<double>::infinity();
        for (int a = 0; a < NUM_ACTIONS; a++) {
            maxNextQ = max(maxNextQ, getQValue(nextState, static_cast<Action>(a)));
        }
        
        // Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        double currentQ = getQValue(state, action);
        double newQ = currentQ + LEARNING_RATE * 
                     (reward + DISCOUNT_FACTOR * maxNextQ - currentQ);
        
        StateActionPair pair(state, action);
        qTable[pair] = newQ;
    }
    
    /**
     * @brief Select action using epsilon-greedy policy
     * @param state Current state
     * @return Selected action
     */
    Action selectAction(const Position& state) const {
        // Epsilon-greedy: explore with probability EPSILON, exploit otherwise
        if ((double)rand() / RAND_MAX < EPSILON) {
            return static_cast<Action>(rand() % NUM_ACTIONS);
        }
        
        // Exploit: choose action with highest Q-value
        double maxQ = -numeric_limits<double>::infinity();
        Action bestAction = UP;
        
        for (int a = 0; a < NUM_ACTIONS; a++) {
            double q = getQValue(state, static_cast<Action>(a));
            if (q > maxQ) {
                maxQ = q;
                bestAction = static_cast<Action>(a);
            }
        }
        
        return bestAction;
    }
    
public:
    /**
     * @brief Constructor - Initialize the Q* algorithm
     * @param gridSize Size of the grid
     */
    QStar(int gridSize = GRID_SIZE) {
        grid.resize(gridSize, vector<int>(gridSize, EMPTY));
        srand(time(0));
    }
    
    /**
     * @brief Set up the environment grid
     * @param obstacles Vector of obstacle positions
     * @param start Starting position
     * @param goal Goal position
     */
    void setupEnvironment(const vector<Position>& obstacles,
                         const Position& start, const Position& goal) {
        // Clear grid
        for (auto& row : grid) {
            fill(row.begin(), row.end(), EMPTY);
        }
        
        // Set obstacles
        for (const auto& obs : obstacles) {
            if (isValidPosition(obs)) {
                grid[obs.row][obs.col] = OBSTACLE;
            }
        }
        
        startPosition = start;
        goalPosition = goal;
        grid[start.row][start.col] = START;
        grid[goal.row][goal.col] = GOAL;
    }
    
    /**
     * @brief Train the Q-learning component
     */
    void train() {
        cout << "Training Q* algorithm..." << endl;
        
        for (int episode = 0; episode < MAX_EPISODES; episode++) {
            Position currentPos = startPosition;
            int steps = 0;
            
            while (!(currentPos == goalPosition) && steps < MAX_STEPS_PER_EPISODE) {
                // Select action
                Action action = selectAction(currentPos);
                
                // Take action and observe result
                Position nextPos = getNextPosition(currentPos, action);
                
                // Ensure next position is valid
                if (!isValidPosition(nextPos)) {
                    nextPos = currentPos;
                }
                
                double reward = getReward(nextPos);
                
                // Update Q-value
                updateQValue(currentPos, action, reward, nextPos);
                
                currentPos = nextPos;
                steps++;
            }
            
            // Print progress every 100 episodes
            if ((episode + 1) % 100 == 0) {
                cout << "Episode " << (episode + 1) << "/" << MAX_EPISODES 
                     << " completed" << endl;
            }
        }
        
        cout << "Training completed!" << endl;
    }
    
    /**
     * @brief Find optimal path using Q* (combining A* and Q-learning)
     * @return Vector of positions representing the path
     */
    vector<Position> findPath() {
        priority_queue<Node, vector<Node>, greater<Node>> openSet;
        map<Position, bool> closedSet;
        map<Position, Position> cameFrom;
        
        // Initialize start node
        Node startNode(startPosition, 0, 
                      calculateHeuristic(startPosition, goalPosition),
                      getQValue(startPosition, UP));  // Use Q-value
        openSet.push(startNode);
        
        while (!openSet.empty()) {
            Node current = openSet.top();
            openSet.pop();
            
            // Goal reached
            if (current.position == goalPosition) {
                return reconstructPath(cameFrom, current.position);
            }
            
            // Skip if already processed
            if (closedSet[current.position]) continue;
            closedSet[current.position] = true;
            
            // Explore neighbors
            for (int a = 0; a < NUM_ACTIONS; a++) {
                Action action = static_cast<Action>(a);
                Position neighbor = getNextPosition(current.position, action);
                
                if (!isValidPosition(neighbor) || closedSet[neighbor]) {
                    continue;
                }
                
                double tentativeG = current.gCost + 1.0;
                double heuristic = calculateHeuristic(neighbor, goalPosition);
                double qValue = getQValue(current.position, action);
                
                Node neighborNode(neighbor, tentativeG, heuristic, qValue);
                cameFrom[neighbor] = current.position;
                openSet.push(neighborNode);
            }
        }
        
        return vector<Position>();  // No path found
    }
    
    /**
     * @brief Reconstruct path from goal to start
     * @param cameFrom Map of parent positions
     * @param current Current position (goal)
     * @return Vector of positions representing the path
     */
    vector<Position> reconstructPath(const map<Position, Position>& cameFrom,
                                    Position current) {
        vector<Position> path;
        path.push_back(current);
        
        while (cameFrom.find(current) != cameFrom.end()) {
            current = cameFrom.at(current);
            path.push_back(current);
        }
        
        reverse(path.begin(), path.end());
        return path;
    }
    
    /**
     * @brief Display the grid with path
     * @param path Path to display
     */
    void displayGrid(const vector<Position>& path = vector<Position>()) const {
        vector<vector<int>> displayGrid = grid;
        
        // Mark path
        for (const auto& pos : path) {
            if (displayGrid[pos.row][pos.col] == EMPTY) {
                displayGrid[pos.row][pos.col] = PATH;
            }
        }
        
        cout << "\nGrid Visualization:\n";
        cout << "  ";
        for (int i = 0; i < GRID_SIZE; i++) {
            cout << setw(2) << i << " ";
        }
        cout << "\n";
        
        for (int i = 0; i < GRID_SIZE; i++) {
            cout << setw(2) << i << " ";
            for (int j = 0; j < GRID_SIZE; j++) {
                switch (displayGrid[i][j]) {
                    case EMPTY:    cout << " . "; break;
                    case OBSTACLE: cout << " # "; break;
                    case START:    cout << " S "; break;
                    case GOAL:     cout << " G "; break;
                    case PATH:     cout << " * "; break;
                    default:       cout << " ? "; break;
                }
            }
            cout << "\n";
        }
        cout << "\nLegend: S=Start, G=Goal, #=Obstacle, *=Path, .=Empty\n";
    }
    
    /**
     * @brief Display Q-values for a specific state
     * @param state Position to display Q-values for
     */
    void displayQValues(const Position& state) const {
        cout << "\nQ-values for position (" << state.row << ", " << state.col << "):\n";
        cout << "UP:    " << fixed << setprecision(2) << getQValue(state, UP) << "\n";
        cout << "DOWN:  " << fixed << setprecision(2) << getQValue(state, DOWN) << "\n";
        cout << "LEFT:  " << fixed << setprecision(2) << getQValue(state, LEFT) << "\n";
        cout << "RIGHT: " << fixed << setprecision(2) << getQValue(state, RIGHT) << "\n";
    }
};

// ============================
// Main Function - Demonstration
// ============================

int main() {
    cout << "========================================\n";
    cout << "Q* Algorithm Implementation\n";
    cout << "========================================\n\n";
    
    // Create Q* instance
    QStar qstar(GRID_SIZE);
    
    // Define obstacles
    vector<Position> obstacles = {
        Position(2, 2), Position(2, 3), Position(2, 4),
        Position(5, 5), Position(5, 6), Position(6, 5),
        Position(7, 2), Position(8, 2), Position(9, 2)
    };
    
    // Set start and goal positions
    Position start(0, 0);
    Position goal(9, 9);
    
    // Setup environment
    qstar.setupEnvironment(obstacles, start, goal);
    
    cout << "Initial Grid Configuration:\n";
    qstar.displayGrid();
    
    // Train the Q-learning component
    qstar.train();
    
    // Find optimal path using Q*
    cout << "\nFinding optimal path using Q*...\n";
    vector<Position> path = qstar.findPath();
    
    if (path.empty()) {
        cout << "No path found!\n";
    } else {
        cout << "\nPath found with " << path.size() << " steps:\n";
        for (size_t i = 0; i < path.size(); i++) {
            cout << "Step " << i << ": (" << path[i].row << ", " 
                 << path[i].col << ")\n";
        }
        
        cout << "\nFinal Grid with Path:\n";
        qstar.displayGrid(path);
        
        // Display Q-values for start position
        qstar.displayQValues(start);
    }
    
    cout << "\n========================================\n";
    cout << "Q* Algorithm Completed Successfully!\n";
    cout << "========================================\n";
    
    return 0;
}
