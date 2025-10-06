/*
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Implementation
 * 
 * This implementation demonstrates the DBSCAN clustering algorithm, which groups together
 * points that are closely packed together (points with many nearby neighbors), marking as
 * outliers points that lie alone in low-density regions.
 * 
 * Algorithm Overview:
 * - Core Points: Points with at least minPoints neighbors within epsilon distance
 * - Border Points: Points within epsilon distance of a core point but with fewer than minPoints neighbors
 * - Noise Points: Points that are neither core nor border points
 * 
 * Author: Implementation for experimental-wins
 * Date: October 6, 2025
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <set>
#include <algorithm>
#include <iomanip>

using namespace std;

/**
 * @struct Point
 * @brief Represents a data point in n-dimensional space
 */
struct Point {
    vector<double> coordinates;  // Feature values for the point
    int clusterId;               // Cluster assignment (-1 for noise, 0 for unvisited)
    bool isVisited;              // Flag to track if point has been processed
    
    /**
     * @brief Constructor for Point
     * @param coords Vector of coordinate values
     */
    Point(const vector<double>& coords) 
        : coordinates(coords), clusterId(0), isVisited(false) {}
};

/**
 * @class DBSCANClusterer
 * @brief Implements the DBSCAN clustering algorithm
 * 
 * DBSCAN clusters data based on density, finding arbitrarily shaped clusters
 * and identifying outliers in the dataset.
 */
class DBSCANClusterer {
private:
    double epsilon;              // Maximum distance between two points to be considered neighbors
    int minPoints;               // Minimum number of points to form a dense region (core point)
    vector<Point> dataPoints;    // Collection of all data points
    int numberOfClusters;        // Total number of clusters formed
    
    /**
     * @brief Calculates Euclidean distance between two points
     * @param point1 First point
     * @param point2 Second point
     * @return Euclidean distance between the points
     */
    double calculateEuclideanDistance(const Point& point1, const Point& point2) {
        if (point1.coordinates.size() != point2.coordinates.size()) {
            cerr << "Error: Points have different dimensions" << endl;
            return -1.0;
        }
        
        double sumSquaredDifferences = 0.0;
        for (size_t i = 0; i < point1.coordinates.size(); i++) {
            double difference = point1.coordinates[i] - point2.coordinates[i];
            sumSquaredDifferences += difference * difference;
        }
        
        return sqrt(sumSquaredDifferences);
    }
    
    /**
     * @brief Finds all neighbors of a point within epsilon distance
     * @param pointIndex Index of the point to find neighbors for
     * @return Vector of indices of neighboring points
     */
    vector<int> findNeighbors(int pointIndex) {
        vector<int> neighbors;
        
        for (size_t i = 0; i < dataPoints.size(); i++) {
            if (i == pointIndex) continue; // Skip the point itself
            
            double distance = calculateEuclideanDistance(dataPoints[pointIndex], dataPoints[i]);
            if (distance <= epsilon) {
                neighbors.push_back(i);
            }
        }
        
        return neighbors;
    }
    
    /**
     * @brief Expands a cluster from a core point using BFS approach
     * @param pointIndex Index of the core point to expand from
     * @param neighbors Initial neighbors of the core point
     * @param clusterId ID to assign to points in this cluster
     */
    void expandCluster(int pointIndex, vector<int>& neighbors, int clusterId) {
        // Assign cluster to the initial core point
        dataPoints[pointIndex].clusterId = clusterId;
        
        // Use a queue for breadth-first expansion
        queue<int> pointsToProcess;
        for (int neighborIdx : neighbors) {
            pointsToProcess.push(neighborIdx);
        }
        
        // Process all points in the cluster
        while (!pointsToProcess.empty()) {
            int currentPointIdx = pointsToProcess.front();
            pointsToProcess.pop();
            
            // Skip if already processed
            if (dataPoints[currentPointIdx].isVisited) {
                continue;
            }
            
            dataPoints[currentPointIdx].isVisited = true;
            dataPoints[currentPointIdx].clusterId = clusterId;
            
            // Find neighbors of current point
            vector<int> currentNeighbors = findNeighbors(currentPointIdx);
            
            // If this point is also a core point, add its neighbors to the queue
            if (currentNeighbors.size() >= minPoints) {
                for (int neighborIdx : currentNeighbors) {
                    if (!dataPoints[neighborIdx].isVisited) {
                        pointsToProcess.push(neighborIdx);
                    }
                }
            }
        }
    }

public:
    /**
     * @brief Constructor for DBSCANClusterer
     * @param eps Maximum distance for neighborhood (epsilon parameter)
     * @param minPts Minimum points required to form a dense region
     */
    DBSCANClusterer(double eps, int minPts) 
        : epsilon(eps), minPoints(minPts), numberOfClusters(0) {
        if (eps <= 0) {
            cerr << "Warning: Epsilon should be positive" << endl;
        }
        if (minPts < 1) {
            cerr << "Warning: MinPoints should be at least 1" << endl;
        }
    }
    
    /**
     * @brief Adds a data point to the dataset
     * @param coordinates Vector of feature values for the point
     */
    void addDataPoint(const vector<double>& coordinates) {
        dataPoints.push_back(Point(coordinates));
    }
    
    /**
     * @brief Loads multiple data points into the dataset
     * @param points 2D vector where each row represents a data point
     */
    void loadDataPoints(const vector<vector<double>>& points) {
        dataPoints.clear();
        for (const auto& point : points) {
            addDataPoint(point);
        }
    }
    
    /**
     * @brief Performs DBSCAN clustering on the loaded data points
     * @return Number of clusters found (excluding noise)
     */
    int performClustering() {
        if (dataPoints.empty()) {
            cerr << "Error: No data points loaded" << endl;
            return 0;
        }
        
        numberOfClusters = 0;
        
        // Process each point
        for (size_t i = 0; i < dataPoints.size(); i++) {
            // Skip if already visited
            if (dataPoints[i].isVisited) {
                continue;
            }
            
            dataPoints[i].isVisited = true;
            
            // Find neighbors of current point
            vector<int> neighbors = findNeighbors(i);
            
            // Check if this is a core point
            if (neighbors.size() < minPoints) {
                // Mark as noise (will be reassigned if it's a border point)
                dataPoints[i].clusterId = -1;
            } else {
                // This is a core point - create a new cluster
                numberOfClusters++;
                expandCluster(i, neighbors, numberOfClusters);
            }
        }
        
        return numberOfClusters;
    }
    
    /**
     * @brief Retrieves the cluster assignments for all points
     * @return Vector of cluster IDs (-1 for noise, positive integers for clusters)
     */
    vector<int> getClusterAssignments() {
        vector<int> assignments;
        for (const auto& point : dataPoints) {
            assignments.push_back(point.clusterId);
        }
        return assignments;
    }
    
    /**
     * @brief Gets the number of clusters found (excluding noise)
     * @return Number of valid clusters
     */
    int getNumberOfClusters() {
        return numberOfClusters;
    }
    
    /**
     * @brief Counts the number of noise points
     * @return Number of points classified as noise
     */
    int countNoisePoints() {
        int noiseCount = 0;
        for (const auto& point : dataPoints) {
            if (point.clusterId == -1) {
                noiseCount++;
            }
        }
        return noiseCount;
    }
    
    /**
     * @brief Displays detailed clustering results
     */
    void displayClusteringResults() {
        cout << "\n========== DBSCAN Clustering Results ==========" << endl;
        cout << "Parameters:" << endl;
        cout << "  Epsilon (eps): " << epsilon << endl;
        cout << "  Minimum Points: " << minPoints << endl;
        cout << "  Total Data Points: " << dataPoints.size() << endl;
        cout << "  Number of Clusters: " << numberOfClusters << endl;
        cout << "  Noise Points: " << countNoisePoints() << endl;
        
        // Display points by cluster
        for (int clusterId = -1; clusterId <= numberOfClusters; clusterId++) {
            if (clusterId == 0) continue; // Skip unvisited (shouldn't exist after clustering)
            
            vector<int> clusterPoints;
            for (size_t i = 0; i < dataPoints.size(); i++) {
                if (dataPoints[i].clusterId == clusterId) {
                    clusterPoints.push_back(i);
                }
            }
            
            if (!clusterPoints.empty()) {
                if (clusterId == -1) {
                    cout << "\nNoise Points (" << clusterPoints.size() << "):" << endl;
                } else {
                    cout << "\nCluster " << clusterId << " (" << clusterPoints.size() << " points):" << endl;
                }
                
                for (int pointIdx : clusterPoints) {
                    cout << "  Point " << pointIdx << ": [";
                    for (size_t j = 0; j < dataPoints[pointIdx].coordinates.size(); j++) {
                        cout << fixed << setprecision(2) << dataPoints[pointIdx].coordinates[j];
                        if (j < dataPoints[pointIdx].coordinates.size() - 1) {
                            cout << ", ";
                        }
                    }
                    cout << "]" << endl;
                }
            }
        }
        cout << "==============================================" << endl;
    }
    
    /**
     * @brief Displays a summary of clustering results
     */
    void displaySummary() {
        cout << "\n========== Clustering Summary ==========" << endl;
        cout << "Total Points: " << dataPoints.size() << endl;
        cout << "Clusters Found: " << numberOfClusters << endl;
        cout << "Noise Points: " << countNoisePoints() << endl;
        
        // Count points per cluster
        for (int clusterId = 1; clusterId <= numberOfClusters; clusterId++) {
            int count = 0;
            for (const auto& point : dataPoints) {
                if (point.clusterId == clusterId) {
                    count++;
                }
            }
            cout << "Cluster " << clusterId << ": " << count << " points" << endl;
        }
        cout << "=======================================" << endl;
    }
    
    /**
     * @brief Resets the clustering state for re-clustering with different parameters
     */
    void resetClustering() {
        for (auto& point : dataPoints) {
            point.clusterId = 0;
            point.isVisited = false;
        }
        numberOfClusters = 0;
    }
    
    /**
     * @brief Updates clustering parameters
     * @param eps New epsilon value
     * @param minPts New minimum points value
     */
    void updateParameters(double eps, int minPts) {
        epsilon = eps;
        minPoints = minPts;
        resetClustering();
    }
};

/**
 * @brief Main function demonstrating the DBSCAN clustering algorithm
 */
int main() {
    cout << "========== DBSCAN Clustering Algorithm Demo ==========" << endl;
    
    // Create a sample dataset with 3 distinct clusters and some noise
    vector<vector<double>> dataset = {
        // Cluster 1: Points around (2, 2)
        {2.0, 2.0},
        {2.1, 2.1},
        {2.2, 2.0},
        {2.0, 2.2},
        {2.1, 1.9},
        {1.9, 2.1},
        
        // Cluster 2: Points around (8, 8)
        {8.0, 8.0},
        {8.1, 8.1},
        {8.0, 8.2},
        {7.9, 8.0},
        {8.2, 7.9},
        {8.1, 8.2},
        
        // Cluster 3: Points around (5, 2)
        {5.0, 2.0},
        {5.1, 2.1},
        {5.0, 1.9},
        {4.9, 2.0},
        {5.2, 2.0},
        
        // Noise points (isolated)
        {0.0, 0.0},
        {10.0, 0.0},
        {0.0, 10.0}
    };
    
    // Set DBSCAN parameters
    double epsilon = 0.5;      // Maximum distance for neighborhood
    int minPoints = 3;         // Minimum points to form a cluster
    
    cout << "\nDataset contains " << dataset.size() << " points" << endl;
    cout << "Using parameters: epsilon = " << epsilon << ", minPoints = " << minPoints << endl;
    
    // Create DBSCAN clusterer and load data
    DBSCANClusterer clusterer(epsilon, minPoints);
    clusterer.loadDataPoints(dataset);
    
    // Perform clustering
    cout << "\nPerforming DBSCAN clustering..." << endl;
    int clustersFound = clusterer.performClustering();
    cout << "Clustering complete! Found " << clustersFound << " clusters." << endl;
    
    // Display results
    clusterer.displayClusteringResults();
    
    // Display summary
    clusterer.displaySummary();
    
    // Demonstrate parameter adjustment
    cout << "\n========== Testing with Different Parameters ==========" << endl;
    cout << "Trying with epsilon = 1.0, minPoints = 2" << endl;
    clusterer.updateParameters(1.0, 2);
    clustersFound = clusterer.performClustering();
    cout << "Found " << clustersFound << " clusters with new parameters." << endl;
    clusterer.displaySummary();
    
    cout << "\n========== Demo Complete ==========" << endl;
    
    return 0;
}
