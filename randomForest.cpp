/**
 * Random Forest Implementation in C++
 * 
 * This file contains implementations for both Random Forest Classifier and Regressor.
 * Random Forest is an ensemble learning method that constructs multiple decision trees
 * during training and outputs the mode (classification) or mean (regression) of predictions.
 * 
 * Key Features:
 * - Bootstrap sampling for creating diverse trees
 * - Random feature selection at each split
 * - Parallel tree construction capability
 * - Support for both classification and regression tasks
 * 
 * Author: Implementation for experimental-wins
 * Date: October 6, 2025
 */

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <memory>
#include <numeric>

using namespace std;

// ==================== Utility Structures ====================

/**
 * Structure to represent a data point in the dataset
 */
struct DataPoint {
    vector<double> features;  // Feature values
    double label;             // Target value (class label or regression value)
    
    DataPoint(const vector<double>& f, double l) : features(f), label(l) {}
};

/**
 * Structure to represent a node in the decision tree
 */
struct TreeNode {
    bool isLeaf;                  // Flag indicating if this is a leaf node
    double prediction;            // Prediction value (class or regression output)
    int featureIndex;            // Index of feature used for splitting
    double threshold;            // Threshold value for the split
    shared_ptr<TreeNode> left;   // Left child (values <= threshold)
    shared_ptr<TreeNode> right;  // Right child (values > threshold)
    
    TreeNode() : isLeaf(false), prediction(0.0), featureIndex(-1), threshold(0.0) {}
};

// ==================== Decision Tree Base Class ====================

/**
 * Base Decision Tree class for Random Forest
 * Implements CART (Classification and Regression Trees) algorithm
 */
class DecisionTree {
protected:
    shared_ptr<TreeNode> root;
    int maxDepth;
    int minSamplesSplit;
    int maxFeatures;
    random_device rd;
    mt19937 gen;
    
    /**
     * Calculate Gini impurity for classification
     * Gini = 1 - sum(p_i^2) where p_i is probability of class i
     */
    double calculateGini(const vector<DataPoint>& data) {
        if (data.empty()) return 0.0;
        
        map<double, int> labelCounts;
        for (const auto& point : data) {
            labelCounts[point.label]++;
        }
        
        double gini = 1.0;
        int totalSamples = data.size();
        
        for (const auto& pair : labelCounts) {
            double probability = static_cast<double>(pair.second) / totalSamples;
            gini -= probability * probability;
        }
        
        return gini;
    }
    
    /**
     * Calculate variance for regression
     * Used as impurity measure for regression trees
     */
    double calculateVariance(const vector<DataPoint>& data) {
        if (data.empty()) return 0.0;
        
        double mean = 0.0;
        for (const auto& point : data) {
            mean += point.label;
        }
        mean /= data.size();
        
        double variance = 0.0;
        for (const auto& point : data) {
            variance += (point.label - mean) * (point.label - mean);
        }
        
        return variance / data.size();
    }
    
    /**
     * Split data based on feature and threshold
     */
    pair<vector<DataPoint>, vector<DataPoint>> splitData(
        const vector<DataPoint>& data,
        int featureIndex,
        double threshold) {
        
        vector<DataPoint> leftSplit, rightSplit;
        
        for (const auto& point : data) {
            if (point.features[featureIndex] <= threshold) {
                leftSplit.push_back(point);
            } else {
                rightSplit.push_back(point);
            }
        }
        
        return {leftSplit, rightSplit};
    }
    
    /**
     * Find the best split for a node
     * Returns: {best_feature_index, best_threshold, best_impurity_reduction}
     */
    virtual tuple<int, double, double> findBestSplit(
        const vector<DataPoint>& data,
        const vector<int>& candidateFeatures) = 0;
    
    /**
     * Calculate prediction value for a leaf node
     */
    virtual double calculateLeafValue(const vector<DataPoint>& data) = 0;
    
    /**
     * Recursively build the decision tree
     */
    shared_ptr<TreeNode> buildTree(const vector<DataPoint>& data, int depth) {
        auto node = make_shared<TreeNode>();
        
        // Stopping criteria
        if (depth >= maxDepth || 
            data.size() < minSamplesSplit || 
            data.empty()) {
            node->isLeaf = true;
            node->prediction = calculateLeafValue(data);
            return node;
        }
        
        // Randomly select features to consider for splitting
        vector<int> candidateFeatures;
        int numFeatures = data[0].features.size();
        vector<int> allFeatures(numFeatures);
        iota(allFeatures.begin(), allFeatures.end(), 0);
        
        shuffle(allFeatures.begin(), allFeatures.end(), gen);
        int numCandidates = min(maxFeatures, numFeatures);
        candidateFeatures.assign(allFeatures.begin(), allFeatures.begin() + numCandidates);
        
        // Find best split
        auto [bestFeature, bestThreshold, bestGain] = findBestSplit(data, candidateFeatures);
        
        // If no good split found, make it a leaf
        if (bestGain <= 0.0 || bestFeature == -1) {
            node->isLeaf = true;
            node->prediction = calculateLeafValue(data);
            return node;
        }
        
        // Split the data
        auto [leftData, rightData] = splitData(data, bestFeature, bestThreshold);
        
        // If split results in empty child, make it a leaf
        if (leftData.empty() || rightData.empty()) {
            node->isLeaf = true;
            node->prediction = calculateLeafValue(data);
            return node;
        }
        
        // Create internal node
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(leftData, depth + 1);
        node->right = buildTree(rightData, depth + 1);
        
        return node;
    }
    
    /**
     * Predict for a single data point using the tree
     */
    double predictSingle(const vector<double>& features, shared_ptr<TreeNode> node) {
        if (node->isLeaf) {
            return node->prediction;
        }
        
        if (features[node->featureIndex] <= node->threshold) {
            return predictSingle(features, node->left);
        } else {
            return predictSingle(features, node->right);
        }
    }
    
public:
    DecisionTree(int maxD = 10, int minSamples = 2, int maxFeat = -1)
        : maxDepth(maxD), minSamplesSplit(minSamples), maxFeatures(maxFeat), gen(rd()) {}
    
    /**
     * Train the decision tree
     */
    void train(const vector<DataPoint>& data) {
        if (maxFeatures <= 0) {
            maxFeatures = static_cast<int>(sqrt(data[0].features.size()));
        }
        root = buildTree(data, 0);
    }
    
    /**
     * Predict for a single sample
     */
    double predict(const vector<double>& features) {
        if (!root) {
            throw runtime_error("Tree not trained yet!");
        }
        return predictSingle(features, root);
    }
};

// ==================== Classification Tree ====================

/**
 * Decision Tree for Classification tasks
 */
class ClassificationTree : public DecisionTree {
protected:
    /**
     * Find best split using Gini impurity
     */
    tuple<int, double, double> findBestSplit(
        const vector<DataPoint>& data,
        const vector<int>& candidateFeatures) override {
        
        double bestGain = 0.0;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double parentImpurity = calculateGini(data);
        
        for (int featureIdx : candidateFeatures) {
            // Get unique values for this feature
            set<double> uniqueValues;
            for (const auto& point : data) {
                uniqueValues.insert(point.features[featureIdx]);
            }
            
            // Try each unique value as a threshold
            for (double threshold : uniqueValues) {
                auto [leftSplit, rightSplit] = splitData(data, featureIdx, threshold);
                
                if (leftSplit.empty() || rightSplit.empty()) continue;
                
                // Calculate weighted Gini impurity
                double leftGini = calculateGini(leftSplit);
                double rightGini = calculateGini(rightSplit);
                double weightedGini = (leftSplit.size() * leftGini + 
                                      rightSplit.size() * rightGini) / data.size();
                
                double gain = parentImpurity - weightedGini;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }
        
        return {bestFeature, bestThreshold, bestGain};
    }
    
    /**
     * Calculate majority class for leaf node
     */
    double calculateLeafValue(const vector<DataPoint>& data) override {
        if (data.empty()) return 0.0;
        
        map<double, int> labelCounts;
        for (const auto& point : data) {
            labelCounts[point.label]++;
        }
        
        // Return the most frequent class
        return max_element(labelCounts.begin(), labelCounts.end(),
                          [](const pair<double, int>& a, const pair<double, int>& b) {
                              return a.second < b.second;
                          })->first;
    }
    
public:
    ClassificationTree(int maxD = 10, int minSamples = 2, int maxFeat = -1)
        : DecisionTree(maxD, minSamples, maxFeat) {}
};

// ==================== Regression Tree ====================

/**
 * Decision Tree for Regression tasks
 */
class RegressionTree : public DecisionTree {
protected:
    /**
     * Find best split using variance reduction
     */
    tuple<int, double, double> findBestSplit(
        const vector<DataPoint>& data,
        const vector<int>& candidateFeatures) override {
        
        double bestGain = 0.0;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double parentVariance = calculateVariance(data);
        
        for (int featureIdx : candidateFeatures) {
            // Get unique values for this feature
            set<double> uniqueValues;
            for (const auto& point : data) {
                uniqueValues.insert(point.features[featureIdx]);
            }
            
            // Try each unique value as a threshold
            for (double threshold : uniqueValues) {
                auto [leftSplit, rightSplit] = splitData(data, featureIdx, threshold);
                
                if (leftSplit.empty() || rightSplit.empty()) continue;
                
                // Calculate weighted variance
                double leftVar = calculateVariance(leftSplit);
                double rightVar = calculateVariance(rightSplit);
                double weightedVar = (leftSplit.size() * leftVar + 
                                     rightSplit.size() * rightVar) / data.size();
                
                double gain = parentVariance - weightedVar;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }
        
        return {bestFeature, bestThreshold, bestGain};
    }
    
    /**
     * Calculate mean value for leaf node
     */
    double calculateLeafValue(const vector<DataPoint>& data) override {
        if (data.empty()) return 0.0;
        
        double sum = 0.0;
        for (const auto& point : data) {
            sum += point.label;
        }
        return sum / data.size();
    }
    
public:
    RegressionTree(int maxD = 10, int minSamples = 2, int maxFeat = -1)
        : DecisionTree(maxD, minSamples, maxFeat) {}
};

// ==================== Random Forest Base Class ====================

/**
 * Base Random Forest class
 * Implements ensemble of decision trees with bootstrap aggregating (bagging)
 */
template<typename TreeType>
class RandomForest {
protected:
    vector<unique_ptr<TreeType>> trees;  // Collection of decision trees
    int numTrees;                         // Number of trees in the forest
    int maxDepth;                         // Maximum depth of each tree
    int minSamplesSplit;                  // Minimum samples required to split
    int maxFeatures;                      // Maximum features to consider per split
    double bootstrapRatio;                // Ratio of samples to use for each tree
    random_device rd;
    mt19937 gen;
    
    /**
     * Create bootstrap sample from original data
     * Samples with replacement to create diverse training sets
     */
    vector<DataPoint> createBootstrapSample(const vector<DataPoint>& data) {
        uniform_int_distribution<> dist(0, data.size() - 1);
        int sampleSize = static_cast<int>(data.size() * bootstrapRatio);
        
        vector<DataPoint> bootstrapSample;
        bootstrapSample.reserve(sampleSize);
        
        for (int i = 0; i < sampleSize; i++) {
            int randomIndex = dist(gen);
            bootstrapSample.push_back(data[randomIndex]);
        }
        
        return bootstrapSample;
    }
    
public:
    /**
     * Constructor
     * @param nTrees: Number of trees in the forest
     * @param maxD: Maximum depth of each tree
     * @param minSamples: Minimum samples required to split a node
     * @param maxFeat: Maximum features to consider for each split (-1 for sqrt(n_features))
     * @param bootRatio: Ratio of samples to use for each tree (default: 1.0)
     */
    RandomForest(int nTrees = 100, int maxD = 10, int minSamples = 2, 
                 int maxFeat = -1, double bootRatio = 1.0)
        : numTrees(nTrees), maxDepth(maxD), minSamplesSplit(minSamples),
          maxFeatures(maxFeat), bootstrapRatio(bootRatio), gen(rd()) {}
    
    /**
     * Train the Random Forest
     * Creates multiple trees using bootstrap samples
     */
    void train(const vector<DataPoint>& data) {
        cout << "Training Random Forest with " << numTrees << " trees..." << endl;
        
        trees.clear();
        trees.reserve(numTrees);
        
        for (int i = 0; i < numTrees; i++) {
            // Create bootstrap sample
            vector<DataPoint> bootstrapData = createBootstrapSample(data);
            
            // Create and train tree
            auto tree = make_unique<TreeType>(maxDepth, minSamplesSplit, maxFeatures);
            tree->train(bootstrapData);
            trees.push_back(move(tree));
            
            if ((i + 1) % 10 == 0) {
                cout << "Trained " << (i + 1) << " trees..." << endl;
            }
        }
        
        cout << "Training complete!" << endl;
    }
    
    /**
     * Virtual method for prediction (implemented by derived classes)
     */
    virtual double predict(const vector<double>& features) = 0;
    
    /**
     * Get the number of trees in the forest
     */
    int getNumTrees() const {
        return numTrees;
    }
};

// ==================== Random Forest Classifier ====================

/**
 * Random Forest Classifier
 * Predicts class labels using majority voting
 */
class RandomForestClassifier : public RandomForest<ClassificationTree> {
public:
    RandomForestClassifier(int nTrees = 100, int maxD = 10, int minSamples = 2,
                          int maxFeat = -1, double bootRatio = 1.0)
        : RandomForest(nTrees, maxD, minSamples, maxFeat, bootRatio) {}
    
    /**
     * Predict class label using majority voting
     */
    double predict(const vector<double>& features) override {
        if (trees.empty()) {
            throw runtime_error("Forest not trained yet!");
        }
        
        // Collect predictions from all trees
        map<double, int> voteCounts;
        
        for (const auto& tree : trees) {
            double prediction = tree->predict(features);
            voteCounts[prediction]++;
        }
        
        // Return the class with the most votes
        return max_element(voteCounts.begin(), voteCounts.end(),
                          [](const pair<double, int>& a, const pair<double, int>& b) {
                              return a.second < b.second;
                          })->first;
    }
    
    /**
     * Predict with probability estimates
     * Returns a map of class -> probability
     */
    map<double, double> predictProba(const vector<double>& features) {
        if (trees.empty()) {
            throw runtime_error("Forest not trained yet!");
        }
        
        map<double, int> voteCounts;
        
        for (const auto& tree : trees) {
            double prediction = tree->predict(features);
            voteCounts[prediction]++;
        }
        
        // Convert counts to probabilities
        map<double, double> probabilities;
        for (const auto& pair : voteCounts) {
            probabilities[pair.first] = static_cast<double>(pair.second) / trees.size();
        }
        
        return probabilities;
    }
};

// ==================== Random Forest Regressor ====================

/**
 * Random Forest Regressor
 * Predicts continuous values using mean of predictions
 */
class RandomForestRegressor : public RandomForest<RegressionTree> {
public:
    RandomForestRegressor(int nTrees = 100, int maxD = 10, int minSamples = 2,
                         int maxFeat = -1, double bootRatio = 1.0)
        : RandomForest(nTrees, maxD, minSamples, maxFeat, bootRatio) {}
    
    /**
     * Predict continuous value using mean of tree predictions
     */
    double predict(const vector<double>& features) override {
        if (trees.empty()) {
            throw runtime_error("Forest not trained yet!");
        }
        
        double sum = 0.0;
        
        for (const auto& tree : trees) {
            sum += tree->predict(features);
        }
        
        return sum / trees.size();
    }
    
    /**
     * Get standard deviation of predictions (measure of uncertainty)
     */
    double predictStd(const vector<double>& features) {
        if (trees.empty()) {
            throw runtime_error("Forest not trained yet!");
        }
        
        vector<double> predictions;
        predictions.reserve(trees.size());
        
        for (const auto& tree : trees) {
            predictions.push_back(tree->predict(features));
        }
        
        double mean = accumulate(predictions.begin(), predictions.end(), 0.0) / predictions.size();
        
        double variance = 0.0;
        for (double pred : predictions) {
            variance += (pred - mean) * (pred - mean);
        }
        variance /= predictions.size();
        
        return sqrt(variance);
    }
};

// ==================== Utility Functions ====================

/**
 * Calculate accuracy for classification
 */
double calculateAccuracy(const vector<double>& predictions, const vector<double>& actual) {
    if (predictions.size() != actual.size()) {
        throw runtime_error("Predictions and actual labels must have same size!");
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == actual[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / predictions.size();
}

/**
 * Calculate Mean Squared Error for regression
 */
double calculateMSE(const vector<double>& predictions, const vector<double>& actual) {
    if (predictions.size() != actual.size()) {
        throw runtime_error("Predictions and actual values must have same size!");
    }
    
    double sumSquaredError = 0.0;
    for (size_t i = 0; i < predictions.size(); i++) {
        double error = predictions[i] - actual[i];
        sumSquaredError += error * error;
    }
    
    return sumSquaredError / predictions.size();
}

/**
 * Calculate R-squared score for regression
 */
double calculateR2Score(const vector<double>& predictions, const vector<double>& actual) {
    if (predictions.size() != actual.size()) {
        throw runtime_error("Predictions and actual values must have same size!");
    }
    
    double mean = accumulate(actual.begin(), actual.end(), 0.0) / actual.size();
    
    double totalSS = 0.0;
    double residualSS = 0.0;
    
    for (size_t i = 0; i < actual.size(); i++) {
        totalSS += (actual[i] - mean) * (actual[i] - mean);
        residualSS += (actual[i] - predictions[i]) * (actual[i] - predictions[i]);
    }
    
    return 1.0 - (residualSS / totalSS);
}

// ==================== Demo Functions ====================

/**
 * Demo: Random Forest Classifier on Iris-like dataset
 */
void demoClassifier() {
    cout << "\n========== Random Forest Classifier Demo ==========" << endl;
    
    // Create synthetic binary classification dataset
    // Class 0: points centered around (2, 2)
    // Class 1: points centered around (8, 8)
    vector<DataPoint> trainingData;
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> noise(0.0, 1.0);
    
    // Generate Class 0 samples
    for (int i = 0; i < 50; i++) {
        vector<double> features = {2.0 + noise(gen), 2.0 + noise(gen)};
        trainingData.emplace_back(features, 0.0);
    }
    
    // Generate Class 1 samples
    for (int i = 0; i < 50; i++) {
        vector<double> features = {8.0 + noise(gen), 8.0 + noise(gen)};
        trainingData.emplace_back(features, 1.0);
    }
    
    // Create and train classifier
    RandomForestClassifier classifier(50, 5, 2);  // 50 trees, max_depth=5
    classifier.train(trainingData);
    
    // Make predictions on test samples
    vector<vector<double>> testSamples = {
        {2.5, 2.5},  // Should be Class 0
        {7.5, 7.5},  // Should be Class 1
        {1.0, 1.0},  // Should be Class 0
        {9.0, 9.0}   // Should be Class 1
    };
    
    cout << "\nPredictions:" << endl;
    for (size_t i = 0; i < testSamples.size(); i++) {
        double prediction = classifier.predict(testSamples[i]);
        auto probabilities = classifier.predictProba(testSamples[i]);
        
        cout << "Sample [" << testSamples[i][0] << ", " << testSamples[i][1] << "]"
             << " -> Class: " << prediction
             << " (Prob: ";
        for (const auto& prob : probabilities) {
            cout << "C" << prob.first << "=" << prob.second << " ";
        }
        cout << ")" << endl;
    }
    
    // Calculate accuracy on training data
    vector<double> predictions, actuals;
    for (const auto& point : trainingData) {
        predictions.push_back(classifier.predict(point.features));
        actuals.push_back(point.label);
    }
    
    double accuracy = calculateAccuracy(predictions, actuals);
    cout << "\nTraining Accuracy: " << (accuracy * 100) << "%" << endl;
}

/**
 * Demo: Random Forest Regressor on synthetic dataset
 */
void demoRegressor() {
    cout << "\n========== Random Forest Regressor Demo ==========" << endl;
    
    // Create synthetic regression dataset
    // y = 2*x1 + 3*x2 + noise
    vector<DataPoint> trainingData;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 10.0);
    normal_distribution<> noise(0.0, 0.5);
    
    for (int i = 0; i < 100; i++) {
        double x1 = dist(gen);
        double x2 = dist(gen);
        double y = 2.0 * x1 + 3.0 * x2 + noise(gen);
        
        trainingData.emplace_back(vector<double>{x1, x2}, y);
    }
    
    // Create and train regressor
    RandomForestRegressor regressor(50, 5, 2);  // 50 trees, max_depth=5
    regressor.train(trainingData);
    
    // Make predictions on test samples
    vector<vector<double>> testSamples = {
        {5.0, 5.0},   // Expected: ~25
        {2.0, 3.0},   // Expected: ~13
        {7.0, 2.0},   // Expected: ~20
        {1.0, 1.0}    // Expected: ~5
    };
    
    cout << "\nPredictions:" << endl;
    for (size_t i = 0; i < testSamples.size(); i++) {
        double prediction = regressor.predict(testSamples[i]);
        double std = regressor.predictStd(testSamples[i]);
        double expected = 2.0 * testSamples[i][0] + 3.0 * testSamples[i][1];
        
        cout << "Sample [" << testSamples[i][0] << ", " << testSamples[i][1] << "]"
             << " -> Predicted: " << prediction
             << " (Expected: " << expected << ")"
             << " ± " << std << endl;
    }
    
    // Calculate metrics on training data
    vector<double> predictions, actuals;
    for (const auto& point : trainingData) {
        predictions.push_back(regressor.predict(point.features));
        actuals.push_back(point.label);
    }
    
    double mse = calculateMSE(predictions, actuals);
    double r2 = calculateR2Score(predictions, actuals);
    
    cout << "\nTraining MSE: " << mse << endl;
    cout << "Training R² Score: " << r2 << endl;
}

// ==================== Main Function ====================

/**
 * Main function demonstrating both classifier and regressor
 */
int main() {
    cout << "============================================" << endl;
    cout << "   Random Forest Implementation in C++" << endl;
    cout << "============================================" << endl;
    
    try {
        // Run classification demo
        demoClassifier();
        
        // Run regression demo
        demoRegressor();
        
        cout << "\n============================================" << endl;
        cout << "   Demo completed successfully!" << endl;
        cout << "============================================" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
