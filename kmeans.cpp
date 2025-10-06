/**
 * @file kmeans.cpp
 * @brief Implementation of K-Means Clustering Algorithm
 * @author AI Assistant
 * @date 2025-10-06
 * 
 * K-Means is an unsupervised learning algorithm that partitions data into K clusters.
 * Each data point belongs to the cluster with the nearest mean (centroid).
 * 
 * Algorithm Steps:
 * 1. Initialize K centroids randomly
 * 2. Assign each point to the nearest centroid
 * 3. Recalculate centroids as the mean of assigned points
 * 4. Repeat steps 2-3 until convergence
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <iomanip>

using namespace std;

/**
 * @class Point
 * @brief Represents a point in n-dimensional space
 */
class Point {
private:
    vector<double> coordinates;  // Coordinates of the point
    int clusterId;               // ID of the cluster this point belongs to
    
public:
    /**
     * @brief Constructor to create a point
     * @param coords Vector of coordinates
     */
    Point(const vector<double>& coords) : coordinates(coords), clusterId(-1) {}
    
    /**
     * @brief Get the coordinates of the point
     * @return Vector of coordinates
     */
    vector<double> getCoordinates() const {
        return coordinates;
    }
    
    /**
     * @brief Get the dimension of the point
     * @return Number of dimensions
     */
    int getDimension() const {
        return coordinates.size();
    }
    
    /**
     * @brief Set the cluster ID for this point
     * @param id Cluster ID
     */
    void setClusterId(int id) {
        clusterId = id;
    }
    
    /**
     * @brief Get the cluster ID of this point
     * @return Cluster ID
     */
    int getClusterId() const {
        return clusterId;
    }
    
    /**
     * @brief Calculate Euclidean distance to another point
     * @param other The other point
     * @return Euclidean distance
     */
    double distanceTo(const Point& other) const {
        if (coordinates.size() != other.coordinates.size()) {
            throw invalid_argument("Points must have the same dimension");
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < coordinates.size(); i++) {
            double diff = coordinates[i] - other.coordinates[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
};

/**
 * @class Cluster
 * @brief Represents a cluster with a centroid
 */
class Cluster {
private:
    Point centroid;              // Centroid of the cluster
    vector<Point*> points;       // Points assigned to this cluster
    int clusterId;               // Unique identifier for the cluster
    
public:
    /**
     * @brief Constructor to create a cluster
     * @param center Initial centroid position
     * @param id Cluster ID
     */
    Cluster(const Point& center, int id) : centroid(center), clusterId(id) {}
    
    /**
     * @brief Add a point to this cluster
     * @param point Pointer to the point to add
     */
    void addPoint(Point* point) {
        points.push_back(point);
        point->setClusterId(clusterId);
    }
    
    /**
     * @brief Remove all points from this cluster
     */
    void clearPoints() {
        points.clear();
    }
    
    /**
     * @brief Get the centroid of this cluster
     * @return Centroid point
     */
    Point getCentroid() const {
        return centroid;
    }
    
    /**
     * @brief Get all points in this cluster
     * @return Vector of point pointers
     */
    vector<Point*> getPoints() const {
        return points;
    }
    
    /**
     * @brief Get the cluster ID
     * @return Cluster ID
     */
    int getId() const {
        return clusterId;
    }
    
    /**
     * @brief Recalculate the centroid as the mean of all assigned points
     * @return True if centroid changed, false otherwise
     */
    bool updateCentroid() {
        if (points.empty()) {
            return false;
        }
        
        int dimension = centroid.getDimension();
        vector<double> newCoords(dimension, 0.0);
        
        // Sum all coordinates
        for (const auto& point : points) {
            vector<double> coords = point->getCoordinates();
            for (int i = 0; i < dimension; i++) {
                newCoords[i] += coords[i];
            }
        }
        
        // Calculate mean
        for (int i = 0; i < dimension; i++) {
            newCoords[i] /= points.size();
        }
        
        Point newCentroid(newCoords);
        
        // Check if centroid has changed significantly
        double distance = centroid.distanceTo(newCentroid);
        centroid = newCentroid;
        
        return distance > 1e-6;  // Return true if there was a significant change
    }
};

/**
 * @class KMeans
 * @brief Implementation of K-Means clustering algorithm
 */
class KMeans {
private:
    int numClusters;                    // Number of clusters (K)
    int maxIterations;                  // Maximum number of iterations
    double convergenceThreshold;        // Threshold for convergence
    vector<Cluster> clusters;           // Vector of clusters
    
    /**
     * @brief Initialize centroids randomly from the dataset
     * @param dataset Vector of points
     */
    void initializeCentroids(const vector<Point>& dataset) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, dataset.size() - 1);
        
        // Randomly select K points as initial centroids
        for (int i = 0; i < numClusters; i++) {
            int randomIndex = dis(gen);
            clusters.push_back(Cluster(dataset[randomIndex], i));
        }
    }
    
    /**
     * @brief Assign each point to the nearest cluster
     * @param dataset Vector of points (non-const to update cluster IDs)
     * @return True if any assignment changed
     */
    bool assignPointsToClusters(vector<Point>& dataset) {
        bool changed = false;
        
        // Clear all clusters
        for (auto& cluster : clusters) {
            cluster.clearPoints();
        }
        
        // Assign each point to the nearest centroid
        for (auto& point : dataset) {
            int oldClusterId = point.getClusterId();
            int nearestClusterId = findNearestCluster(point);
            
            if (oldClusterId != nearestClusterId) {
                changed = true;
            }
            
            clusters[nearestClusterId].addPoint(&point);
        }
        
        return changed;
    }
    
    /**
     * @brief Find the nearest cluster for a given point
     * @param point The point to find the nearest cluster for
     * @return ID of the nearest cluster
     */
    int findNearestCluster(const Point& point) const {
        double minDistance = numeric_limits<double>::max();
        int nearestClusterId = 0;
        
        for (const auto& cluster : clusters) {
            double distance = point.distanceTo(cluster.getCentroid());
            if (distance < minDistance) {
                minDistance = distance;
                nearestClusterId = cluster.getId();
            }
        }
        
        return nearestClusterId;
    }
    
    /**
     * @brief Update all cluster centroids
     * @return True if any centroid changed significantly
     */
    bool updateCentroids() {
        bool changed = false;
        
        for (auto& cluster : clusters) {
            if (cluster.updateCentroid()) {
                changed = true;
            }
        }
        
        return changed;
    }
    
    /**
     * @brief Calculate the total within-cluster sum of squares (inertia)
     * @return Total inertia value
     */
    double calculateInertia() const {
        double totalInertia = 0.0;
        
        for (const auto& cluster : clusters) {
            Point centroid = cluster.getCentroid();
            for (const auto& point : cluster.getPoints()) {
                double distance = point->distanceTo(centroid);
                totalInertia += distance * distance;
            }
        }
        
        return totalInertia;
    }
    
public:
    /**
     * @brief Constructor for KMeans
     * @param k Number of clusters
     * @param maxIter Maximum number of iterations (default: 300)
     * @param threshold Convergence threshold (default: 1e-4)
     */
    KMeans(int k, int maxIter = 300, double threshold = 1e-4) 
        : numClusters(k), maxIterations(maxIter), convergenceThreshold(threshold) {
        
        if (k <= 0) {
            throw invalid_argument("Number of clusters must be positive");
        }
    }
    
    /**
     * @brief Fit the K-Means model to the dataset
     * @param dataset Vector of points to cluster
     */
    void fit(vector<Point>& dataset) {
        if (dataset.empty()) {
            throw invalid_argument("Dataset cannot be empty");
        }
        
        if (dataset.size() < numClusters) {
            throw invalid_argument("Dataset size must be >= number of clusters");
        }
        
        // Step 1: Initialize centroids
        initializeCentroids(dataset);
        
        cout << "Starting K-Means clustering with K = " << numClusters << endl;
        cout << "Dataset size: " << dataset.size() << endl;
        cout << "Dimension: " << dataset[0].getDimension() << endl;
        cout << string(50, '-') << endl;
        
        // Iterate until convergence or max iterations
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Step 2: Assign points to nearest clusters
            bool assignmentChanged = assignPointsToClusters(dataset);
            
            // Step 3: Update centroids
            bool centroidChanged = updateCentroids();
            
            // Print iteration info
            if (iteration % 10 == 0 || iteration == maxIterations - 1) {
                double inertia = calculateInertia();
                cout << "Iteration " << setw(3) << iteration 
                     << " | Inertia: " << fixed << setprecision(4) << inertia << endl;
            }
            
            // Step 4: Check for convergence
            if (!assignmentChanged && !centroidChanged) {
                cout << string(50, '-') << endl;
                cout << "Converged at iteration " << iteration << endl;
                break;
            }
        }
        
        // Print final results
        printResults();
    }
    
    /**
     * @brief Predict the cluster for a new point
     * @param point The point to classify
     * @return Cluster ID
     */
    int predict(const Point& point) const {
        if (clusters.empty()) {
            throw runtime_error("Model not fitted yet. Call fit() first.");
        }
        
        return findNearestCluster(point);
    }
    
    /**
     * @brief Get all clusters
     * @return Vector of clusters
     */
    vector<Cluster> getClusters() const {
        return clusters;
    }
    
    /**
     * @brief Print clustering results
     */
    void printResults() const {
        cout << string(50, '=') << endl;
        cout << "K-Means Clustering Results" << endl;
        cout << string(50, '=') << endl;
        
        for (const auto& cluster : clusters) {
            cout << "\nCluster " << cluster.getId() << ":" << endl;
            cout << "  Size: " << cluster.getPoints().size() << " points" << endl;
            cout << "  Centroid: [";
            
            vector<double> coords = cluster.getCentroid().getCoordinates();
            for (size_t i = 0; i < coords.size(); i++) {
                cout << fixed << setprecision(2) << coords[i];
                if (i < coords.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        double finalInertia = calculateInertia();
        cout << "\nFinal Inertia: " << fixed << setprecision(4) << finalInertia << endl;
        cout << string(50, '=') << endl;
    }
};

/**
 * @brief Main function demonstrating K-Means clustering
 */
int main() {
    cout << "K-Means Clustering Algorithm Implementation" << endl;
    cout << string(50, '=') << endl << endl;
    
    // Create sample 2D dataset
    vector<Point> dataset = {
        // Cluster 1 (around origin)
        Point({1.0, 1.0}),
        Point({1.5, 2.0}),
        Point({2.0, 1.5}),
        Point({1.2, 1.8}),
        Point({1.8, 1.2}),
        
        // Cluster 2 (around (8, 8))
        Point({8.0, 8.0}),
        Point({8.5, 8.5}),
        Point({7.5, 8.2}),
        Point({8.2, 7.8}),
        Point({7.8, 8.3}),
        
        // Cluster 3 (around (8, 1))
        Point({8.0, 1.0}),
        Point({8.5, 1.5}),
        Point({7.5, 1.2}),
        Point({8.2, 0.8}),
        Point({7.8, 1.3})
    };
    
    try {
        // Create K-Means instance with K=3
        KMeans kmeans(3, 100);
        
        // Fit the model
        kmeans.fit(dataset);
        
        // Test prediction on a new point
        cout << "\nTesting prediction on new points:" << endl;
        Point testPoint1({1.3, 1.6});
        Point testPoint2({8.1, 8.2});
        
        int cluster1 = kmeans.predict(testPoint1);
        int cluster2 = kmeans.predict(testPoint2);
        
        cout << "Point [1.3, 1.6] belongs to Cluster " << cluster1 << endl;
        cout << "Point [8.1, 8.2] belongs to Cluster " << cluster2 << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
