/**
 * @file decisionTree.cpp
 * @brief Implementation of Decision Tree for both Classification and Regression
 * @author Implementation of Decision Tree Classifier and Regressor
 * @date 2025-10-06
 */

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>



using namespace std;

// ============================================================================
// DECISION TREE NODE STRUCTURE
// ============================================================================

/**
 * @struct TreeNode
 * @brief Represents a node in the decision tree
 */
struct TreeNode {
    int featureIndex;           // Index of feature to split on (-1 for leaf nodes)
    double threshold;           // Threshold value for split
    double value;               // Predicted value (for leaf nodes)
    shared_ptr<TreeNode> left;  // Left child (values <= threshold)
    shared_ptr<TreeNode> right; // Right child (values > threshold)
    
    /**
     * @brief Constructor for TreeNode
     */
    TreeNode() : featureIndex(-1), threshold(0.0), value(0.0), 
                 left(nullptr), right(nullptr) {}
    
    /**
     * @brief Check if node is a leaf
     * @return true if leaf node, false otherwise
     */
    bool isLeaf() const {
        return featureIndex == -1;
    }
};

// ============================================================================
// DECISION TREE CLASSIFIER
// ============================================================================

/**
 * @class DecisionTreeClassifier
 * @brief Decision Tree implementation for classification tasks
 */
class DecisionTreeClassifier {
private:
    shared_ptr<TreeNode> root;
    int maxDepth;
    int minSamplesSplit;
    int currentDepth;
    
    /**
     * @brief Calculate Gini impurity for a set of labels
     * @param labels Vector of class labels
     * @return Gini impurity value
     */
    double calculateGiniImpurity(const vector<int>& labels) {
        if (labels.empty()) return 0.0;
        
        map<int, int> classCounts;
        for (int label : labels) {
            classCounts[label]++;
        }
        
        double gini = 1.0;
        int totalSamples = labels.size();
        
        for (const auto& pair : classCounts) {
            double probability = static_cast<double>(pair.second) / totalSamples;
            gini -= probability * probability;
        }
        
        return gini;
    }
    
    /**
     * @brief Calculate information gain for a split
     * @param leftLabels Labels in left split
     * @param rightLabels Labels in right split
     * @param parentLabels Labels before split
     * @return Information gain value
     */
    double calculateInformationGain(const vector<int>& leftLabels,
                                    const vector<int>& rightLabels,
                                    const vector<int>& parentLabels) {
        double parentGini = calculateGiniImpurity(parentLabels);
        
        int totalSamples = parentLabels.size();
        double leftWeight = static_cast<double>(leftLabels.size()) / totalSamples;
        double rightWeight = static_cast<double>(rightLabels.size()) / totalSamples;
        
        double childrenGini = leftWeight * calculateGiniImpurity(leftLabels) +
                             rightWeight * calculateGiniImpurity(rightLabels);
        
        return parentGini - childrenGini;
    }
    
    /**
     * @brief Find the best split for the dataset
     * @param X Feature matrix
     * @param y Target labels
     * @param bestFeature Output: index of best feature
     * @param bestThreshold Output: threshold for best split
     * @return Best information gain achieved
     */
    double findBestSplit(const vector<vector<double>>& X,
                        const vector<int>& y,
                        int& bestFeature,
                        double& bestThreshold) {
        double bestGain = -1.0;
        int numFeatures = X[0].size();
        int numSamples = X.size();
        
        // Try each feature
        for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
            // Get unique values for this feature
            vector<double> featureValues;
            for (int i = 0; i < numSamples; i++) {
                featureValues.push_back(X[i][featureIdx]);
            }
            sort(featureValues.begin(), featureValues.end());
            
            // Try each possible threshold (midpoint between consecutive values)
            for (size_t i = 0; i < featureValues.size() - 1; i++) {
                if (featureValues[i] == featureValues[i + 1]) continue;
                
                double threshold = (featureValues[i] + featureValues[i + 1]) / 2.0;
                
                // Split the data
                vector<int> leftLabels, rightLabels;
                for (int j = 0; j < numSamples; j++) {
                    if (X[j][featureIdx] <= threshold) {
                        leftLabels.push_back(y[j]);
                    } else {
                        rightLabels.push_back(y[j]);
                    }
                }
                
                // Skip if split is trivial
                if (leftLabels.empty() || rightLabels.empty()) continue;
                
                // Calculate information gain
                double gain = calculateInformationGain(leftLabels, rightLabels, y);
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }
        
        return bestGain;
    }
    
    /**
     * @brief Get the most common class label
     * @param labels Vector of labels
     * @return Most frequent label
     */
    int getMajorityClass(const vector<int>& labels) {
        if (labels.empty()) return 0;
        
        map<int, int> classCounts;
        for (int label : labels) {
            classCounts[label]++;
        }
        
        int majorityClass = labels[0];
        int maxCount = 0;
        
        for (const auto& pair : classCounts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                majorityClass = pair.first;
            }
        }
        
        return majorityClass;
    }
    
    /**
     * @brief Recursively build the decision tree
     * @param X Feature matrix
     * @param y Target labels
     * @param depth Current depth in tree
     * @return Pointer to the root node of subtree
     */
    shared_ptr<TreeNode> buildTree(const vector<vector<double>>& X,
                                   const vector<int>& y,
                                   int depth) {
        shared_ptr<TreeNode> node = make_shared<TreeNode>();
        
        // Check stopping criteria
        if (depth >= maxDepth || 
            y.size() < static_cast<size_t>(minSamplesSplit) ||
            calculateGiniImpurity(y) == 0.0) {
            // Create leaf node
            node->value = getMajorityClass(y);
            return node;
        }
        
        // Find best split
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestGain = findBestSplit(X, y, bestFeature, bestThreshold);
        
        // If no good split found, create leaf
        if (bestGain <= 0.0 || bestFeature == -1) {
            node->value = getMajorityClass(y);
            return node;
        }
        
        // Split the data
        vector<vector<double>> leftX, rightX;
        vector<int> leftY, rightY;
        
        for (size_t i = 0; i < X.size(); i++) {
            if (X[i][bestFeature] <= bestThreshold) {
                leftX.push_back(X[i]);
                leftY.push_back(y[i]);
            } else {
                rightX.push_back(X[i]);
                rightY.push_back(y[i]);
            }
        }
        
        // Create internal node
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(leftX, leftY, depth + 1);
        node->right = buildTree(rightX, rightY, depth + 1);
        
        return node;
    }
    
    /**
     * @brief Make prediction for a single sample
     * @param node Current node in tree
     * @param sample Feature vector
     * @return Predicted class label
     */
    int predictSample(const shared_ptr<TreeNode>& node,
                     const vector<double>& sample) {
        if (node->isLeaf()) {
            return static_cast<int>(node->value);
        }
        
        if (sample[node->featureIndex] <= node->threshold) {
            return predictSample(node->left, sample);
        } else {
            return predictSample(node->right, sample);
        }
    }

public:
    /**
     * @brief Constructor for DecisionTreeClassifier
     * @param maxDepth Maximum depth of tree (default: 10)
     * @param minSamplesSplit Minimum samples required to split (default: 2)
     */
    DecisionTreeClassifier(int maxDepth = 10, int minSamplesSplit = 2)
        : root(nullptr), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit),
          currentDepth(0) {}
    
    /**
     * @brief Train the decision tree classifier
     * @param X Feature matrix (samples x features)
     * @param y Target labels
     */
    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            cout << "Error: Invalid training data!" << endl;
            return;
        }
        
        cout << "Training Decision Tree Classifier..." << endl;
        cout << "Number of samples: " << X.size() << endl;
        cout << "Number of features: " << X[0].size() << endl;
        
        root = buildTree(X, y, 0);
        
        cout << "Training complete!" << endl;
    }
    
    /**
     * @brief Make predictions for multiple samples
     * @param X Feature matrix
     * @return Vector of predicted labels
     */
    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> predictions;
        
        if (!root) {
            cout << "Error: Model not trained!" << endl;
            return predictions;
        }
        
        for (const auto& sample : X) {
            predictions.push_back(predictSample(root, sample));
        }
        
        return predictions;
    }
    
    /**
     * @brief Calculate accuracy on test set
     * @param X Feature matrix
     * @param yTrue True labels
     * @return Accuracy score
     */
    double score(const vector<vector<double>>& X, const vector<int>& yTrue) {
        vector<int> yPred = predict(X);
        
        if (yPred.size() != yTrue.size()) {
            return 0.0;
        }
        
        int correct = 0;
        for (size_t i = 0; i < yTrue.size(); i++) {
            if (yPred[i] == yTrue[i]) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / yTrue.size();
    }
};

// ============================================================================
// DECISION TREE REGRESSOR
// ============================================================================

/**
 * @class DecisionTreeRegressor
 * @brief Decision Tree implementation for regression tasks
 */
class DecisionTreeRegressor {
private:
    shared_ptr<TreeNode> root;
    int maxDepth;
    int minSamplesSplit;
    
    /**
     * @brief Calculate mean squared error for a set of values
     * @param values Vector of target values
     * @return MSE value
     */
    double calculateMSE(const vector<double>& values) {
        if (values.empty()) return 0.0;
        
        double mean = accumulate(values.begin(), values.end(), 0.0) / values.size();
        
        double mse = 0.0;
        for (double value : values) {
            mse += (value - mean) * (value - mean);
        }
        
        return mse / values.size();
    }
    
    /**
     * @brief Calculate variance reduction for a split
     * @param leftValues Values in left split
     * @param rightValues Values in right split
     * @param parentValues Values before split
     * @return Variance reduction value
     */
    double calculateVarianceReduction(const vector<double>& leftValues,
                                      const vector<double>& rightValues,
                                      const vector<double>& parentValues) {
        double parentMSE = calculateMSE(parentValues);
        
        int totalSamples = parentValues.size();
        double leftWeight = static_cast<double>(leftValues.size()) / totalSamples;
        double rightWeight = static_cast<double>(rightValues.size()) / totalSamples;
        
        double childrenMSE = leftWeight * calculateMSE(leftValues) +
                            rightWeight * calculateMSE(rightValues);
        
        return parentMSE - childrenMSE;
    }
    
    /**
     * @brief Find the best split for the dataset
     * @param X Feature matrix
     * @param y Target values
     * @param bestFeature Output: index of best feature
     * @param bestThreshold Output: threshold for best split
     * @return Best variance reduction achieved
     */
    double findBestSplit(const vector<vector<double>>& X,
                        const vector<double>& y,
                        int& bestFeature,
                        double& bestThreshold) {
        double bestReduction = -1.0;
        int numFeatures = X[0].size();
        int numSamples = X.size();
        
        // Try each feature
        for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
            // Get unique values for this feature
            vector<double> featureValues;
            for (int i = 0; i < numSamples; i++) {
                featureValues.push_back(X[i][featureIdx]);
            }
            sort(featureValues.begin(), featureValues.end());
            
            // Try each possible threshold
            for (size_t i = 0; i < featureValues.size() - 1; i++) {
                if (featureValues[i] == featureValues[i + 1]) continue;
                
                double threshold = (featureValues[i] + featureValues[i + 1]) / 2.0;
                
                // Split the data
                vector<double> leftValues, rightValues;
                for (int j = 0; j < numSamples; j++) {
                    if (X[j][featureIdx] <= threshold) {
                        leftValues.push_back(y[j]);
                    } else {
                        rightValues.push_back(y[j]);
                    }
                }
                
                // Skip if split is trivial
                if (leftValues.empty() || rightValues.empty()) continue;
                
                // Calculate variance reduction
                double reduction = calculateVarianceReduction(leftValues, rightValues, y);
                
                if (reduction > bestReduction) {
                    bestReduction = reduction;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }
        
        return bestReduction;
    }
    
    /**
     * @brief Calculate mean of values
     * @param values Vector of values
     * @return Mean value
     */
    double calculateMean(const vector<double>& values) {
        if (values.empty()) return 0.0;
        return accumulate(values.begin(), values.end(), 0.0) / values.size();
    }
    
    /**
     * @brief Recursively build the decision tree
     * @param X Feature matrix
     * @param y Target values
     * @param depth Current depth in tree
     * @return Pointer to the root node of subtree
     */
    shared_ptr<TreeNode> buildTree(const vector<vector<double>>& X,
                                   const vector<double>& y,
                                   int depth) {
        shared_ptr<TreeNode> node = make_shared<TreeNode>();
        
        // Check stopping criteria
        if (depth >= maxDepth || 
            y.size() < static_cast<size_t>(minSamplesSplit) ||
            calculateMSE(y) < 1e-7) {
            // Create leaf node
            node->value = calculateMean(y);
            return node;
        }
        
        // Find best split
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestReduction = findBestSplit(X, y, bestFeature, bestThreshold);
        
        // If no good split found, create leaf
        if (bestReduction <= 0.0 || bestFeature == -1) {
            node->value = calculateMean(y);
            return node;
        }
        
        // Split the data
        vector<vector<double>> leftX, rightX;
        vector<double> leftY, rightY;
        
        for (size_t i = 0; i < X.size(); i++) {
            if (X[i][bestFeature] <= bestThreshold) {
                leftX.push_back(X[i]);
                leftY.push_back(y[i]);
            } else {
                rightX.push_back(X[i]);
                rightY.push_back(y[i]);
            }
        }
        
        // Create internal node
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(leftX, leftY, depth + 1);
        node->right = buildTree(rightX, rightY, depth + 1);
        
        return node;
    }
    
    /**
     * @brief Make prediction for a single sample
     * @param node Current node in tree
     * @param sample Feature vector
     * @return Predicted value
     */
    double predictSample(const shared_ptr<TreeNode>& node,
                        const vector<double>& sample) {
        if (node->isLeaf()) {
            return node->value;
        }
        
        if (sample[node->featureIndex] <= node->threshold) {
            return predictSample(node->left, sample);
        } else {
            return predictSample(node->right, sample);
        }
    }

public:
    /**
     * @brief Constructor for DecisionTreeRegressor
     * @param maxDepth Maximum depth of tree (default: 10)
     * @param minSamplesSplit Minimum samples required to split (default: 2)
     */
    DecisionTreeRegressor(int maxDepth = 10, int minSamplesSplit = 2)
        : root(nullptr), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit) {}
    
    /**
     * @brief Train the decision tree regressor
     * @param X Feature matrix (samples x features)
     * @param y Target values
     */
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            cout << "Error: Invalid training data!" << endl;
            return;
        }
        
        cout << "Training Decision Tree Regressor..." << endl;
        cout << "Number of samples: " << X.size() << endl;
        cout << "Number of features: " << X[0].size() << endl;
        
        root = buildTree(X, y, 0);
        
        cout << "Training complete!" << endl;
    }
    
    /**
     * @brief Make predictions for multiple samples
     * @param X Feature matrix
     * @return Vector of predicted values
     */
    vector<double> predict(const vector<vector<double>>& X) {
        vector<double> predictions;
        
        if (!root) {
            cout << "Error: Model not trained!" << endl;
            return predictions;
        }
        
        for (const auto& sample : X) {
            predictions.push_back(predictSample(root, sample));
        }
        
        return predictions;
    }
    
    /**
     * @brief Calculate R² score on test set
     * @param X Feature matrix
     * @param yTrue True values
     * @return R² score
     */
    double score(const vector<vector<double>>& X, const vector<double>& yTrue) {
        vector<double> yPred = predict(X);
        
        if (yPred.size() != yTrue.size()) {
            return 0.0;
        }
        
        // Calculate mean of true values
        double yMean = accumulate(yTrue.begin(), yTrue.end(), 0.0) / yTrue.size();
        
        // Calculate total sum of squares
        double ssTot = 0.0;
        for (double y : yTrue) {
            ssTot += (y - yMean) * (y - yMean);
        }
        
        // Calculate residual sum of squares
        double ssRes = 0.0;
        for (size_t i = 0; i < yTrue.size(); i++) {
            ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
        }
        
        // Calculate R² score
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * @brief Calculate Mean Squared Error
     * @param X Feature matrix
     * @param yTrue True values
     * @return MSE value
     */
    double meanSquaredError(const vector<vector<double>>& X, 
                           const vector<double>& yTrue) {
        vector<double> yPred = predict(X);
        
        if (yPred.size() != yTrue.size()) {
            return numeric_limits<double>::max();
        }
        
        double mse = 0.0;
        for (size_t i = 0; i < yTrue.size(); i++) {
            mse += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
        }
        
        return mse / yTrue.size();
    }
};

// ============================================================================
// MAIN FUNCTION - DEMONSTRATION
// ============================================================================

int main() {
    cout << "========================================" << endl;
    cout << "Decision Tree Implementation in C++" << endl;
    cout << "========================================" << endl << endl;
    
    // ========================================================================
    // CLASSIFICATION EXAMPLE
    // ========================================================================
    
    cout << "=== CLASSIFICATION EXAMPLE ===" << endl << endl;
    
    // Create sample classification dataset (Iris-like data)
    // Features: [sepal_length, sepal_width, petal_length, petal_width]
    // Classes: 0, 1, 2
    vector<vector<double>> X_train_clf = {
        {5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
        {7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
        {6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1},
        {5.0, 3.6, 1.4, 0.2}, {5.4, 3.9, 1.7, 0.4}, {4.6, 3.4, 1.4, 0.3},
        {6.5, 2.8, 4.6, 1.5}, {5.7, 2.8, 4.5, 1.3}, {6.3, 3.3, 4.7, 1.6},
        {6.7, 3.1, 5.6, 2.4}, {6.9, 3.1, 5.1, 2.3}, {5.8, 2.7, 5.1, 1.9}
    };
    
    vector<int> y_train_clf = {
        0, 0, 0,  // Class 0 (Setosa)
        1, 1, 1,  // Class 1 (Versicolor)
        2, 2, 2,  // Class 2 (Virginica)
        0, 0, 0,
        1, 1, 1,
        2, 2, 2
    };
    
    // Test data
    vector<vector<double>> X_test_clf = {
        {5.0, 3.4, 1.5, 0.2},  // Should be class 0
        {6.7, 3.0, 5.0, 1.7},  // Should be class 1 or 2
        {6.0, 3.0, 4.8, 1.8}   // Should be class 2
    };
    
    vector<int> y_test_clf = {0, 2, 2};
    
    // Create and train classifier
    DecisionTreeClassifier classifier(5, 2);
    classifier.fit(X_train_clf, y_train_clf);
    
    cout << "\nMaking predictions on test set..." << endl;
    vector<int> predictions_clf = classifier.predict(X_test_clf);
    
    cout << "\nPredictions:" << endl;
    for (size_t i = 0; i < predictions_clf.size(); i++) {
        cout << "Sample " << i + 1 << ": Predicted = " << predictions_clf[i] 
             << ", Actual = " << y_test_clf[i] << endl;
    }
    
    double accuracy = classifier.score(X_test_clf, y_test_clf);
    cout << "\nAccuracy: " << (accuracy * 100) << "%" << endl;
    
    // ========================================================================
    // REGRESSION EXAMPLE
    // ========================================================================
    
    cout << "\n\n=== REGRESSION EXAMPLE ===" << endl << endl;
    
    // Create sample regression dataset (Boston Housing-like data)
    // Features: [rooms, age, distance]
    // Target: house price
    vector<vector<double>> X_train_reg = {
        {6.0, 65.0, 4.09}, {7.0, 78.0, 4.97}, {7.0, 61.0, 4.97},
        {5.0, 45.0, 6.06}, {6.0, 54.0, 6.06}, {8.0, 58.0, 6.06},
        {4.0, 66.0, 5.56}, {5.0, 50.0, 7.20}, {6.0, 52.0, 7.20},
        {6.5, 70.0, 3.50}, {7.5, 80.0, 4.00}, {5.5, 40.0, 5.00},
        {8.5, 90.0, 3.00}, {4.5, 35.0, 8.00}, {7.0, 55.0, 4.50}
    };
    
    vector<double> y_train_reg = {
        24.0, 21.6, 34.7,  // House prices in $1000s
        33.4, 36.2, 28.7,
        22.9, 27.1, 16.5,
        18.9, 15.0, 18.9,
        21.7, 20.4, 18.2
    };
    
    // Test data
    vector<vector<double>> X_test_reg = {
        {6.0, 60.0, 5.0},   // Medium house
        {8.0, 75.0, 3.5},   // Large house, close
        {4.0, 40.0, 7.0}    // Small house, far
    };
    
    vector<double> y_test_reg = {25.0, 22.0, 20.0};
    
    // Create and train regressor
    DecisionTreeRegressor regressor(5, 2);
    regressor.fit(X_train_reg, y_train_reg);
    
    cout << "\nMaking predictions on test set..." << endl;
    vector<double> predictions_reg = regressor.predict(X_test_reg);
    
    cout << "\nPredictions:" << endl;
    for (size_t i = 0; i < predictions_reg.size(); i++) {
        cout << "Sample " << i + 1 << ": Predicted = " << predictions_reg[i] 
             << ", Actual = " << y_test_reg[i] << endl;
    }
    
    double r2_score = regressor.score(X_test_reg, y_test_reg);
    double mse = regressor.meanSquaredError(X_test_reg, y_test_reg);
    
    cout << "\nR² Score: " << r2_score << endl;
    cout << "Mean Squared Error: " << mse << endl;
    
    cout << "\n========================================" << endl;
    cout << "Decision Tree Demo Complete!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
