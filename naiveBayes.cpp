/*
 * Naive Bayes Classifier Implementation
 * 
 * This implementation demonstrates a Gaussian Naive Bayes classifier for continuous features.
 * The algorithm is based on Bayes' theorem with the "naive" assumption of conditional 
 * independence between every pair of features given the class label.
 * 
 * Author: Implementation for experimental-wins
 * Date: October 6, 2025
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <numeric>

using namespace std;

/**
 * @class NaiveBayesClassifier
 * @brief Implements Gaussian Naive Bayes classification algorithm
 * 
 * This classifier assumes that features follow a Gaussian (normal) distribution
 * within each class. It calculates class probabilities using Bayes' theorem.
 */
class NaiveBayesClassifier {
private:
    // Store mean values for each feature per class
    map<int, vector<double>> classMeans;
    
    // Store variance values for each feature per class
    map<int, vector<double>> classVariances;
    
    // Store prior probabilities for each class
    map<int, double> classPriors;
    
    // Number of features in the dataset
    int numberOfFeatures;
    
    /**
     * @brief Calculates the mean of a vector of values
     * @param values Vector of numerical values
     * @return Mean value
     */
    double calculateMean(const vector<double>& values) {
        if (values.empty()) return 0.0;
        double sum = accumulate(values.begin(), values.end(), 0.0);
        return sum / values.size();
    }
    
    /**
     * @brief Calculates the variance of a vector of values
     * @param values Vector of numerical values
     * @param mean Pre-calculated mean value
     * @return Variance value
     */
    double calculateVariance(const vector<double>& values, double mean) {
        if (values.empty()) return 0.0;
        double sumSquaredDiff = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        // Add small epsilon to avoid division by zero
        return (sumSquaredDiff / values.size()) + 1e-9;
    }
    
    /**
     * @brief Calculates the Gaussian probability density function
     * @param x The value to evaluate
     * @param mean Mean of the distribution
     * @param variance Variance of the distribution
     * @return Probability density at x
     */
    double gaussianProbability(double x, double mean, double variance) {
        double exponent = exp(-((x - mean) * (x - mean)) / (2 * variance));
        return (1.0 / sqrt(2 * M_PI * variance)) * exponent;
    }

public:
    /**
     * @brief Constructor for NaiveBayesClassifier
     */
    NaiveBayesClassifier() : numberOfFeatures(0) {}
    
    /**
     * @brief Trains the Naive Bayes classifier on the given dataset
     * @param features 2D vector where each row is a sample and each column is a feature
     * @param labels Vector of class labels corresponding to each sample
     */
    void train(const vector<vector<double>>& features, const vector<int>& labels) {
        if (features.empty() || labels.empty() || features.size() != labels.size()) {
            cerr << "Error: Invalid training data" << endl;
            return;
        }
        
        numberOfFeatures = features[0].size();
        int totalSamples = features.size();
        
        // Organize data by class
        map<int, vector<vector<double>>> classData;
        for (size_t i = 0; i < features.size(); i++) {
            classData[labels[i]].push_back(features[i]);
        }
        
        // Calculate statistics for each class
        for (const auto& classPair : classData) {
            int classLabel = classPair.first;
            const vector<vector<double>>& classFeatures = classPair.second;
            
            // Calculate prior probability for this class
            classPriors[classLabel] = static_cast<double>(classFeatures.size()) / totalSamples;
            
            // Initialize vectors for means and variances
            classMeans[classLabel] = vector<double>(numberOfFeatures);
            classVariances[classLabel] = vector<double>(numberOfFeatures);
            
            // Calculate mean and variance for each feature in this class
            for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
                vector<double> featureValues;
                for (const auto& sample : classFeatures) {
                    featureValues.push_back(sample[featureIndex]);
                }
                
                double mean = calculateMean(featureValues);
                double variance = calculateVariance(featureValues, mean);
                
                classMeans[classLabel][featureIndex] = mean;
                classVariances[classLabel][featureIndex] = variance;
            }
        }
        
        cout << "Training completed successfully!" << endl;
        cout << "Number of classes: " << classPriors.size() << endl;
        cout << "Number of features: " << numberOfFeatures << endl;
    }
    
    /**
     * @brief Predicts the class label for a single sample
     * @param sample Vector of feature values
     * @return Predicted class label
     */
    int predict(const vector<double>& sample) {
        if (sample.size() != numberOfFeatures) {
            cerr << "Error: Sample has incorrect number of features" << endl;
            return -1;
        }
        
        int bestClass = -1;
        double maxProbability = -INFINITY;
        
        // Calculate posterior probability for each class
        for (const auto& classPair : classPriors) {
            int classLabel = classPair.first;
            double classPrior = classPair.second;
            
            // Start with log of prior probability to avoid numerical underflow
            double logProbability = log(classPrior);
            
            // Multiply by likelihood of each feature (add logs instead)
            for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
                double mean = classMeans[classLabel][featureIndex];
                double variance = classVariances[classLabel][featureIndex];
                double featureValue = sample[featureIndex];
                
                double likelihood = gaussianProbability(featureValue, mean, variance);
                logProbability += log(likelihood + 1e-10); // Add small value to avoid log(0)
            }
            
            // Update best class if this has higher probability
            if (logProbability > maxProbability) {
                maxProbability = logProbability;
                bestClass = classLabel;
            }
        }
        
        return bestClass;
    }
    
    /**
     * @brief Predicts class labels for multiple samples
     * @param samples 2D vector where each row is a sample
     * @return Vector of predicted class labels
     */
    vector<int> predictBatch(const vector<vector<double>>& samples) {
        vector<int> predictions;
        for (const auto& sample : samples) {
            predictions.push_back(predict(sample));
        }
        return predictions;
    }
    
    /**
     * @brief Calculates the accuracy of predictions
     * @param predictions Vector of predicted labels
     * @param actualLabels Vector of true labels
     * @return Accuracy as a percentage (0-100)
     */
    double calculateAccuracy(const vector<int>& predictions, const vector<int>& actualLabels) {
        if (predictions.size() != actualLabels.size()) {
            cerr << "Error: Prediction and label vectors have different sizes" << endl;
            return 0.0;
        }
        
        int correctPredictions = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == actualLabels[i]) {
                correctPredictions++;
            }
        }
        
        return (static_cast<double>(correctPredictions) / predictions.size()) * 100.0;
    }
    
    /**
     * @brief Displays the learned parameters of the model
     */
    void displayModelParameters() {
        cout << "\n========== Model Parameters ==========" << endl;
        for (const auto& classPair : classPriors) {
            int classLabel = classPair.first;
            cout << "\nClass " << classLabel << ":" << endl;
            cout << "  Prior Probability: " << classPriors[classLabel] << endl;
            cout << "  Feature Means: ";
            for (double mean : classMeans[classLabel]) {
                cout << mean << " ";
            }
            cout << endl;
            cout << "  Feature Variances: ";
            for (double variance : classVariances[classLabel]) {
                cout << variance << " ";
            }
            cout << endl;
        }
        cout << "======================================" << endl;
    }
};

/**
 * @brief Main function demonstrating the Naive Bayes classifier
 */
int main() {
    cout << "========== Naive Bayes Classifier Demo ==========" << endl;
    
    // Create a simple dataset for binary classification
    // Features: [height, weight] (normalized/simplified values)
    // Class 0: Small objects, Class 1: Large objects
    vector<vector<double>> trainingFeatures = {
        {5.1, 3.5},   // Class 0
        {4.9, 3.0},   // Class 0
        {4.7, 3.2},   // Class 0
        {4.6, 3.1},   // Class 0
        {5.0, 3.6},   // Class 0
        {7.0, 3.2},   // Class 1
        {6.4, 3.2},   // Class 1
        {6.9, 3.1},   // Class 1
        {6.5, 2.8},   // Class 1
        {6.8, 3.0}    // Class 1
    };
    
    vector<int> trainingLabels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    // Create and train the classifier
    NaiveBayesClassifier classifier;
    cout << "\nTraining the classifier..." << endl;
    classifier.train(trainingFeatures, trainingLabels);
    
    // Display learned parameters
    classifier.displayModelParameters();
    
    // Test the classifier on training data
    cout << "\n========== Testing on Training Data ==========" << endl;
    vector<int> predictions = classifier.predictBatch(trainingFeatures);
    
    for (size_t i = 0; i < trainingFeatures.size(); i++) {
        cout << "Sample " << i + 1 << ": [" << trainingFeatures[i][0] 
             << ", " << trainingFeatures[i][1] << "] -> Predicted: " 
             << predictions[i] << ", Actual: " << trainingLabels[i];
        
        if (predictions[i] == trainingLabels[i]) {
            cout << " ✓" << endl;
        } else {
            cout << " ✗" << endl;
        }
    }
    
    // Calculate and display accuracy
    double accuracy = classifier.calculateAccuracy(predictions, trainingLabels);
    cout << "\nTraining Accuracy: " << accuracy << "%" << endl;
    
    // Test on new data
    cout << "\n========== Testing on New Data ==========" << endl;
    vector<vector<double>> testFeatures = {
        {5.2, 3.4},   // Expected: Class 0
        {6.7, 3.0}    // Expected: Class 1
    };
    
    vector<int> testPredictions = classifier.predictBatch(testFeatures);
    for (size_t i = 0; i < testFeatures.size(); i++) {
        cout << "Test Sample " << i + 1 << ": [" << testFeatures[i][0] 
             << ", " << testFeatures[i][1] << "] -> Predicted Class: " 
             << testPredictions[i] << endl;
    }
    
    cout << "\n========== Demo Complete ==========" << endl;
    
    return 0;
}
