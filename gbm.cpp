/**
 * @file gbm.cpp
 * @brief Implementation of Gradient Boosting Machine (GBM) for Classification and Regression
 * @author Gradient Boosting Implementation
 * @date October 6, 2025
 * 
 * This file contains two main classes:
 * 1. GBMClassifier - For binary classification tasks
 * 2. GBMRegressor - For regression tasks
 * 
 * Both classes implement the Gradient Boosting algorithm using decision tree stumps as weak learners.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

using namespace std;

// ==================== Decision Tree Stump Class ====================
/**
 * @class DecisionTreeStump
 * @brief A simple decision tree with one split (weak learner for boosting)
 * 
 * This class represents a decision stump, which is a decision tree with depth 1.
 * It finds the best feature and threshold to split the data.
 */
class DecisionTreeStump {
private:
    int m_featureIndex;        // Index of the feature to split on
    double m_threshold;         // Threshold value for splitting
    double m_leftValue;         // Prediction value for left child (feature <= threshold)
    double m_rightValue;        // Prediction value for right child (feature > threshold)
    
public:
    /**
     * @brief Constructor
     */
    DecisionTreeStump() 
        : m_featureIndex(-1), m_threshold(0.0), m_leftValue(0.0), m_rightValue(0.0) {}
    
    /**
     * @brief Fit the decision stump to the data
     * @param X Feature matrix (samples x features)
     * @param gradients Gradient values to fit against
     * @param sampleWeights Optional sample weights
     */
    void fit(const vector<vector<double>>& X, 
             const vector<double>& gradients,
             const vector<double>& sampleWeights = vector<double>()) {
        
        int numSamples = X.size();
        int numFeatures = X[0].size();
        
        // Use uniform weights if not provided
        vector<double> weights = sampleWeights;
        if (weights.empty()) {
            weights.assign(numSamples, 1.0);
        }
        
        double bestMSE = numeric_limits<double>::max();
        
        // Try each feature
        for (int featureIdx = 0; featureIdx < numFeatures; ++featureIdx) {
            // Get unique sorted values for this feature
            vector<double> featureValues;
            for (int i = 0; i < numSamples; ++i) {
                featureValues.push_back(X[i][featureIdx]);
            }
            sort(featureValues.begin(), featureValues.end());
            featureValues.erase(unique(featureValues.begin(), featureValues.end()), 
                              featureValues.end());
            
            // Try each possible threshold
            for (size_t t = 0; t < featureValues.size() - 1; ++t) {
                double threshold = (featureValues[t] + featureValues[t + 1]) / 2.0;
                
                // Split samples based on threshold
                vector<double> leftGradients, rightGradients;
                vector<double> leftWeights, rightWeights;
                
                for (int i = 0; i < numSamples; ++i) {
                    if (X[i][featureIdx] <= threshold) {
                        leftGradients.push_back(gradients[i]);
                        leftWeights.push_back(weights[i]);
                    } else {
                        rightGradients.push_back(gradients[i]);
                        rightWeights.push_back(weights[i]);
                    }
                }
                
                if (leftGradients.empty() || rightGradients.empty()) {
                    continue;
                }
                
                // Calculate weighted mean for each split
                double leftValue = 0.0, rightValue = 0.0;
                double leftWeightSum = 0.0, rightWeightSum = 0.0;
                
                for (size_t i = 0; i < leftGradients.size(); ++i) {
                    leftValue += leftWeights[i] * leftGradients[i];
                    leftWeightSum += leftWeights[i];
                }
                leftValue /= leftWeightSum;
                
                for (size_t i = 0; i < rightGradients.size(); ++i) {
                    rightValue += rightWeights[i] * rightGradients[i];
                    rightWeightSum += rightWeights[i];
                }
                rightValue /= rightWeightSum;
                
                // Calculate MSE
                double mse = 0.0;
                for (size_t i = 0; i < leftGradients.size(); ++i) {
                    mse += leftWeights[i] * pow(leftGradients[i] - leftValue, 2);
                }
                for (size_t i = 0; i < rightGradients.size(); ++i) {
                    mse += rightWeights[i] * pow(rightGradients[i] - rightValue, 2);
                }
                
                // Update best split if this is better
                if (mse < bestMSE) {
                    bestMSE = mse;
                    m_featureIndex = featureIdx;
                    m_threshold = threshold;
                    m_leftValue = leftValue;
                    m_rightValue = rightValue;
                }
            }
        }
    }
    
    /**
     * @brief Predict using the fitted stump
     * @param X Feature matrix to predict on
     * @return Vector of predictions
     */
    vector<double> predict(const vector<vector<double>>& X) const {
        vector<double> predictions;
        predictions.reserve(X.size());
        
        for (const auto& sample : X) {
            if (sample[m_featureIndex] <= m_threshold) {
                predictions.push_back(m_leftValue);
            } else {
                predictions.push_back(m_rightValue);
            }
        }
        
        return predictions;
    }
};

// ==================== GBM Classifier Class ====================
/**
 * @class GBMClassifier
 * @brief Gradient Boosting Machine for binary classification
 * 
 * This class implements a gradient boosting classifier using log loss
 * as the loss function. It builds an ensemble of decision stumps
 * to create a strong classifier.
 */
class GBMClassifier {
private:
    int m_numEstimators;                        // Number of boosting iterations
    double m_learningRate;                      // Shrinkage parameter
    vector<DecisionTreeStump> m_estimators;     // Collection of weak learners
    double m_initialPrediction;                 // Initial baseline prediction
    
    /**
     * @brief Sigmoid function for probability conversion
     * @param x Input value
     * @return Sigmoid(x) value between 0 and 1
     */
    double sigmoid(double x) const {
        return 1.0 / (1.0 + exp(-x));
    }
    
    /**
     * @brief Calculate negative gradients for log loss
     * @param y True labels (0 or 1)
     * @param predictions Current raw predictions (logits)
     * @return Vector of negative gradients
     */
    vector<double> calculateNegativeGradients(const vector<int>& y, 
                                              const vector<double>& predictions) const {
        vector<double> gradients;
        gradients.reserve(y.size());
        
        for (size_t i = 0; i < y.size(); ++i) {
            double prob = sigmoid(predictions[i]);
            // Negative gradient of log loss: y - p(y=1)
            gradients.push_back(y[i] - prob);
        }
        
        return gradients;
    }
    
public:
    /**
     * @brief Constructor
     * @param numEstimators Number of boosting rounds (default: 100)
     * @param learningRate Learning rate/shrinkage factor (default: 0.1)
     */
    GBMClassifier(int numEstimators = 100, double learningRate = 0.1)
        : m_numEstimators(numEstimators), m_learningRate(learningRate), 
          m_initialPrediction(0.0) {}
    
    /**
     * @brief Train the GBM classifier
     * @param X Feature matrix (samples x features)
     * @param y Target labels (0 or 1)
     */
    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        int numSamples = X.size();
        
        // Calculate initial prediction (log odds of positive class)
        double sumPositive = accumulate(y.begin(), y.end(), 0.0);
        double positiveRate = sumPositive / numSamples;
        m_initialPrediction = log(positiveRate / (1.0 - positiveRate));
        
        // Initialize predictions with baseline
        vector<double> currentPredictions(numSamples, m_initialPrediction);
        
        // Boosting iterations
        cout << "Training GBM Classifier..." << endl;
        for (int iteration = 0; iteration < m_numEstimators; ++iteration) {
            // Calculate negative gradients (residuals)
            vector<double> gradients = calculateNegativeGradients(y, currentPredictions);
            
            // Fit a decision stump to the gradients
            DecisionTreeStump stump;
            stump.fit(X, gradients);
            
            // Get predictions from the new stump
            vector<double> stumpPredictions = stump.predict(X);
            
            // Update predictions with learning rate
            for (int i = 0; i < numSamples; ++i) {
                currentPredictions[i] += m_learningRate * stumpPredictions[i];
            }
            
            // Store the stump
            m_estimators.push_back(stump);
            
            // Print progress every 20 iterations
            if ((iteration + 1) % 20 == 0) {
                cout << "Iteration " << (iteration + 1) << "/" << m_numEstimators << endl;
            }
        }
        
        cout << "Training completed!" << endl;
    }
    
    /**
     * @brief Predict class labels
     * @param X Feature matrix to predict on
     * @return Vector of predicted class labels (0 or 1)
     */
    vector<int> predict(const vector<vector<double>>& X) const {
        vector<double> probabilities = predictProba(X);
        vector<int> predictions;
        predictions.reserve(probabilities.size());
        
        for (double prob : probabilities) {
            predictions.push_back(prob >= 0.5 ? 1 : 0);
        }
        
        return predictions;
    }
    
    /**
     * @brief Predict class probabilities
     * @param X Feature matrix to predict on
     * @return Vector of predicted probabilities for positive class
     */
    vector<double> predictProba(const vector<vector<double>>& X) const {
        int numSamples = X.size();
        vector<double> predictions(numSamples, m_initialPrediction);
        
        // Accumulate predictions from all estimators
        for (const auto& estimator : m_estimators) {
            vector<double> stumpPredictions = estimator.predict(X);
            
            for (int i = 0; i < numSamples; ++i) {
                predictions[i] += m_learningRate * stumpPredictions[i];
            }
        }
        
        // Convert to probabilities using sigmoid
        vector<double> probabilities;
        probabilities.reserve(numSamples);
        
        for (double pred : predictions) {
            probabilities.push_back(sigmoid(pred));
        }
        
        return probabilities;
    }
    
    /**
     * @brief Calculate accuracy on test data
     * @param X Feature matrix
     * @param y True labels
     * @return Accuracy score (0 to 1)
     */
    double score(const vector<vector<double>>& X, const vector<int>& y) const {
        vector<int> predictions = predict(X);
        int correct = 0;
        
        for (size_t i = 0; i < y.size(); ++i) {
            if (predictions[i] == y[i]) {
                ++correct;
            }
        }
        
        return static_cast<double>(correct) / y.size();
    }
};

// ==================== GBM Regressor Class ====================
/**
 * @class GBMRegressor
 * @brief Gradient Boosting Machine for regression
 * 
 * This class implements a gradient boosting regressor using squared loss
 * (L2 loss) as the loss function. It builds an ensemble of decision stumps
 * to create a strong regression model.
 */
class GBMRegressor {
private:
    int m_numEstimators;                        // Number of boosting iterations
    double m_learningRate;                      // Shrinkage parameter
    vector<DecisionTreeStump> m_estimators;     // Collection of weak learners
    double m_initialPrediction;                 // Initial baseline prediction
    
    /**
     * @brief Calculate negative gradients for squared loss
     * @param y True target values
     * @param predictions Current predictions
     * @return Vector of negative gradients (residuals)
     */
    vector<double> calculateNegativeGradients(const vector<double>& y, 
                                              const vector<double>& predictions) const {
        vector<double> gradients;
        gradients.reserve(y.size());
        
        for (size_t i = 0; i < y.size(); ++i) {
            // Negative gradient of squared loss: y - prediction
            gradients.push_back(y[i] - predictions[i]);
        }
        
        return gradients;
    }
    
public:
    /**
     * @brief Constructor
     * @param numEstimators Number of boosting rounds (default: 100)
     * @param learningRate Learning rate/shrinkage factor (default: 0.1)
     */
    GBMRegressor(int numEstimators = 100, double learningRate = 0.1)
        : m_numEstimators(numEstimators), m_learningRate(learningRate), 
          m_initialPrediction(0.0) {}
    
    /**
     * @brief Train the GBM regressor
     * @param X Feature matrix (samples x features)
     * @param y Target values
     */
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        int numSamples = X.size();
        
        // Calculate initial prediction (mean of target values)
        m_initialPrediction = accumulate(y.begin(), y.end(), 0.0) / numSamples;
        
        // Initialize predictions with baseline
        vector<double> currentPredictions(numSamples, m_initialPrediction);
        
        // Boosting iterations
        cout << "Training GBM Regressor..." << endl;
        for (int iteration = 0; iteration < m_numEstimators; ++iteration) {
            // Calculate negative gradients (residuals)
            vector<double> gradients = calculateNegativeGradients(y, currentPredictions);
            
            // Fit a decision stump to the gradients
            DecisionTreeStump stump;
            stump.fit(X, gradients);
            
            // Get predictions from the new stump
            vector<double> stumpPredictions = stump.predict(X);
            
            // Update predictions with learning rate
            for (int i = 0; i < numSamples; ++i) {
                currentPredictions[i] += m_learningRate * stumpPredictions[i];
            }
            
            // Store the stump
            m_estimators.push_back(stump);
            
            // Print progress every 20 iterations
            if ((iteration + 1) % 20 == 0) {
                cout << "Iteration " << (iteration + 1) << "/" << m_numEstimators << endl;
            }
        }
        
        cout << "Training completed!" << endl;
    }
    
    /**
     * @brief Predict target values
     * @param X Feature matrix to predict on
     * @return Vector of predicted values
     */
    vector<double> predict(const vector<vector<double>>& X) const {
        int numSamples = X.size();
        vector<double> predictions(numSamples, m_initialPrediction);
        
        // Accumulate predictions from all estimators
        for (const auto& estimator : m_estimators) {
            vector<double> stumpPredictions = estimator.predict(X);
            
            for (int i = 0; i < numSamples; ++i) {
                predictions[i] += m_learningRate * stumpPredictions[i];
            }
        }
        
        return predictions;
    }
    
    /**
     * @brief Calculate R² score on test data
     * @param X Feature matrix
     * @param y True target values
     * @return R² score
     */
    double score(const vector<vector<double>>& X, const vector<double>& y) const {
        vector<double> predictions = predict(X);
        
        // Calculate mean of true values
        double yMean = accumulate(y.begin(), y.end(), 0.0) / y.size();
        
        // Calculate sum of squared residuals and total sum of squares
        double ssRes = 0.0;
        double ssTot = 0.0;
        
        for (size_t i = 0; i < y.size(); ++i) {
            ssRes += pow(y[i] - predictions[i], 2);
            ssTot += pow(y[i] - yMean, 2);
        }
        
        // R² = 1 - (SS_res / SS_tot)
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * @brief Calculate Mean Squared Error
     * @param X Feature matrix
     * @param y True target values
     * @return MSE value
     */
    double meanSquaredError(const vector<vector<double>>& X, const vector<double>& y) const {
        vector<double> predictions = predict(X);
        double mse = 0.0;
        
        for (size_t i = 0; i < y.size(); ++i) {
            mse += pow(y[i] - predictions[i], 2);
        }
        
        return mse / y.size();
    }
};

// ==================== Main Function (Example Usage) ====================
/**
 * @brief Main function demonstrating GBM Classifier and Regressor usage
 */
int main() {
    cout << "=== Gradient Boosting Machine (GBM) Implementation ===" << endl;
    cout << endl;
    
    // ========== GBM Classifier Example ==========
    cout << "--- GBM Classifier Example ---" << endl;
    
    // Create synthetic binary classification dataset
    vector<vector<double>> X_classification = {
        {2.5, 1.5}, {3.0, 2.0}, {1.5, 1.0}, {4.0, 3.5},
        {2.0, 1.0}, {3.5, 2.5}, {1.0, 0.5}, {4.5, 4.0},
        {2.8, 2.2}, {3.2, 2.8}, {1.2, 0.8}, {4.2, 3.8},
        {2.3, 1.3}, {3.7, 3.0}, {1.7, 1.2}, {4.7, 4.3}
    };
    
    vector<int> y_classification = {
        0, 1, 0, 1,
        0, 1, 0, 1,
        0, 1, 0, 1,
        0, 1, 0, 1
    };
    
    // Create and train classifier
    GBMClassifier classifier(50, 0.1);  // 50 estimators, learning rate 0.1
    classifier.fit(X_classification, y_classification);
    
    // Make predictions
    vector<int> class_predictions = classifier.predict(X_classification);
    vector<double> probabilities = classifier.predictProba(X_classification);
    
    cout << "\nClassification Results:" << endl;
    cout << "Sample\tTrue\tPred\tProb(+)" << endl;
    for (size_t i = 0; i < min(size_t(5), y_classification.size()); ++i) {
        cout << i << "\t" << y_classification[i] << "\t" 
             << class_predictions[i] << "\t" 
             << probabilities[i] << endl;
    }
    
    double accuracy = classifier.score(X_classification, y_classification);
    cout << "\nTraining Accuracy: " << accuracy * 100 << "%" << endl;
    
    cout << "\n" << endl;
    
    // ========== GBM Regressor Example ==========
    cout << "--- GBM Regressor Example ---" << endl;
    
    // Create synthetic regression dataset
    vector<vector<double>> X_regression = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0},
        {1.5}, {2.5}, {3.5}, {4.5}, {5.5},
        {6.5}, {7.5}, {8.5}, {9.5}, {10.5}
    };
    
    // Target: y = 2*x + 1 with some noise
    vector<double> y_regression = {
        3.1, 5.0, 6.9, 9.2, 11.1,
        12.8, 15.0, 17.1, 18.9, 21.2,
        4.0, 6.1, 8.0, 10.1, 11.9,
        13.8, 16.0, 18.1, 19.8, 22.0
    };
    
    // Create and train regressor
    GBMRegressor regressor(50, 0.1);  // 50 estimators, learning rate 0.1
    regressor.fit(X_regression, y_regression);
    
    // Make predictions
    vector<double> reg_predictions = regressor.predict(X_regression);
    
    cout << "\nRegression Results:" << endl;
    cout << "Sample\tTrue\tPredicted" << endl;
    for (size_t i = 0; i < min(size_t(5), y_regression.size()); ++i) {
        cout << i << "\t" << y_regression[i] << "\t" 
             << reg_predictions[i] << endl;
    }
    
    double r2Score = regressor.score(X_regression, y_regression);
    double mse = regressor.meanSquaredError(X_regression, y_regression);
    cout << "\nR² Score: " << r2Score << endl;
    cout << "Mean Squared Error: " << mse << endl;
    
    cout << "\n=== Program Completed ===" << endl;
    
    return 0;
}
