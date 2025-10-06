/*
 * Multiple Linear Regression Implementation
 * 
 * This implementation demonstrates Multiple Linear Regression using the Normal Equation method.
 * Multiple Linear Regression models the relationship between multiple independent variables
 * (features) and a dependent variable (target) by fitting a linear equation to observed data.
 * 
 * Equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
 * where:
 *   y = predicted value (dependent variable)
 *   x₁, x₂, ..., xₙ = independent variables (features)
 *   β₀ = intercept (bias term)
 *   β₁, β₂, ..., βₙ = coefficients (weights)
 * 
 * Solution Method: Normal Equation
 *   θ = (Xᵀ * X)⁻¹ * Xᵀ * y
 * 
 * Author: Implementation for experimental-wins
 * Date: October 6, 2025
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>

using namespace std;

/**
 * @class Matrix
 * @brief A helper class for matrix operations needed in linear regression
 */
class Matrix {
private:
    vector<vector<double>> data;
    int rows;
    int cols;
    
public:
    /**
     * @brief Constructor for Matrix
     * @param r Number of rows
     * @param c Number of columns
     * @param initialValue Default value to fill the matrix
     */
    Matrix(int r, int c, double initialValue = 0.0) : rows(r), cols(c) {
        data.resize(rows, vector<double>(cols, initialValue));
    }
    
    /**
     * @brief Constructor from 2D vector
     * @param mat 2D vector representing the matrix
     */
    Matrix(const vector<vector<double>>& mat) {
        data = mat;
        rows = mat.size();
        cols = (rows > 0) ? mat[0].size() : 0;
    }
    
    /**
     * @brief Gets the number of rows
     * @return Number of rows
     */
    int getRows() const { return rows; }
    
    /**
     * @brief Gets the number of columns
     * @return Number of columns
     */
    int getCols() const { return cols; }
    
    /**
     * @brief Access matrix element (read/write)
     * @param i Row index
     * @param j Column index
     * @return Reference to the element
     */
    double& operator()(int i, int j) {
        return data[i][j];
    }
    
    /**
     * @brief Access matrix element (read-only)
     * @param i Row index
     * @param j Column index
     * @return Const reference to the element
     */
    const double& operator()(int i, int j) const {
        return data[i][j];
    }
    
    /**
     * @brief Matrix multiplication
     * @param other Matrix to multiply with
     * @return Result of matrix multiplication
     */
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    /**
     * @brief Matrix transpose
     * @return Transposed matrix
     */
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }
    
    /**
     * @brief Matrix inversion using Gauss-Jordan elimination
     * @return Inverted matrix
     */
    Matrix inverse() const {
        if (rows != cols) {
            throw invalid_argument("Only square matrices can be inverted");
        }
        
        int n = rows;
        // Create augmented matrix [A|I]
        Matrix augmented(n, 2 * n);
        
        // Fill left side with original matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented(i, j) = data[i][j];
            }
        }
        
        // Fill right side with identity matrix
        for (int i = 0; i < n; i++) {
            augmented(i, n + i) = 1.0;
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            double maxVal = abs(augmented(i, i));
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (abs(augmented(k, i)) > maxVal) {
                    maxVal = abs(augmented(k, i));
                    maxRow = k;
                }
            }
            
            // Swap rows if needed
            if (maxRow != i) {
                for (int k = 0; k < 2 * n; k++) {
                    swap(augmented(i, k), augmented(maxRow, k));
                }
            }
            
            // Check for singular matrix
            if (abs(augmented(i, i)) < 1e-10) {
                throw runtime_error("Matrix is singular and cannot be inverted");
            }
            
            // Scale pivot row
            double pivot = augmented(i, i);
            for (int j = 0; j < 2 * n; j++) {
                augmented(i, j) /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented(k, i);
                    for (int j = 0; j < 2 * n; j++) {
                        augmented(k, j) -= factor * augmented(i, j);
                    }
                }
            }
        }
        
        // Extract inverse from right side of augmented matrix
        Matrix result(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result(i, j) = augmented(i, n + j);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Converts matrix to vector (for single column matrices)
     * @return Vector representation
     */
    vector<double> toVector() const {
        vector<double> result;
        if (cols == 1) {
            for (int i = 0; i < rows; i++) {
                result.push_back(data[i][0]);
            }
        }
        return result;
    }
    
    /**
     * @brief Displays the matrix
     */
    void display() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << setw(12) << fixed << setprecision(4) << data[i][j] << " ";
            }
            cout << endl;
        }
    }
};

/**
 * @class MultipleLinearRegression
 * @brief Implements Multiple Linear Regression using Normal Equation
 * 
 * This class fits a linear model to predict a continuous target variable
 * based on multiple input features.
 */
class MultipleLinearRegression {
private:
    vector<double> coefficients;      // Model coefficients (weights)
    double intercept;                 // Intercept term (bias)
    int numberOfFeatures;             // Number of independent variables
    bool isTrained;                   // Flag indicating if model is trained
    
    vector<vector<double>> trainingFeatures;  // Stored training data
    vector<double> trainingTargets;           // Stored training targets
    
    /**
     * @brief Adds intercept column to feature matrix
     * @param features Original feature matrix
     * @return Feature matrix with intercept column
     */
    Matrix addInterceptColumn(const vector<vector<double>>& features) {
        int numSamples = features.size();
        int numFeatures = features[0].size();
        
        Matrix X(numSamples, numFeatures + 1);
        
        // First column is all ones (for intercept)
        for (int i = 0; i < numSamples; i++) {
            X(i, 0) = 1.0;
            for (int j = 0; j < numFeatures; j++) {
                X(i, j + 1) = features[i][j];
            }
        }
        
        return X;
    }
    
    /**
     * @brief Calculates Mean Squared Error
     * @param predictions Predicted values
     * @param actual Actual values
     * @return MSE value
     */
    double calculateMSE(const vector<double>& predictions, const vector<double>& actual) {
        if (predictions.size() != actual.size()) {
            return -1.0;
        }
        
        double sumSquaredError = 0.0;
        for (size_t i = 0; i < predictions.size(); i++) {
            double error = predictions[i] - actual[i];
            sumSquaredError += error * error;
        }
        
        return sumSquaredError / predictions.size();
    }
    
    /**
     * @brief Calculates R-squared (coefficient of determination)
     * @param predictions Predicted values
     * @param actual Actual values
     * @return R-squared value (0 to 1)
     */
    double calculateRSquared(const vector<double>& predictions, const vector<double>& actual) {
        if (predictions.size() != actual.size()) {
            return -1.0;
        }
        
        // Calculate mean of actual values
        double mean = 0.0;
        for (double value : actual) {
            mean += value;
        }
        mean /= actual.size();
        
        // Calculate total sum of squares and residual sum of squares
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        for (size_t i = 0; i < actual.size(); i++) {
            totalSumSquares += (actual[i] - mean) * (actual[i] - mean);
            residualSumSquares += (actual[i] - predictions[i]) * (actual[i] - predictions[i]);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }

public:
    /**
     * @brief Constructor for MultipleLinearRegression
     */
    MultipleLinearRegression() : intercept(0.0), numberOfFeatures(0), isTrained(false) {}
    
    /**
     * @brief Trains the model using Normal Equation method
     * @param features 2D vector of training features (samples x features)
     * @param targets Vector of target values
     */
    void train(const vector<vector<double>>& features, const vector<double>& targets) {
        if (features.empty() || targets.empty() || features.size() != targets.size()) {
            cerr << "Error: Invalid training data" << endl;
            return;
        }
        
        // Store training data
        trainingFeatures = features;
        trainingTargets = targets;
        
        numberOfFeatures = features[0].size();
        int numSamples = features.size();
        
        cout << "Training Multiple Linear Regression..." << endl;
        cout << "Number of samples: " << numSamples << endl;
        cout << "Number of features: " << numberOfFeatures << endl;
        
        try {
            // Create design matrix X with intercept column
            Matrix X = addInterceptColumn(features);
            
            // Create target vector y
            Matrix y(numSamples, 1);
            for (int i = 0; i < numSamples; i++) {
                y(i, 0) = targets[i];
            }
            
            // Calculate Normal Equation: θ = (X^T * X)^(-1) * X^T * y
            Matrix X_transpose = X.transpose();
            Matrix X_T_X = X_transpose * X;
            Matrix X_T_X_inv = X_T_X.inverse();
            Matrix X_T_y = X_transpose * y;
            Matrix theta = X_T_X_inv * X_T_y;
            
            // Extract coefficients
            vector<double> thetaVec = theta.toVector();
            intercept = thetaVec[0];
            coefficients.clear();
            for (int i = 1; i < thetaVec.size(); i++) {
                coefficients.push_back(thetaVec[i]);
            }
            
            isTrained = true;
            cout << "Training completed successfully!" << endl;
            
        } catch (const exception& e) {
            cerr << "Error during training: " << e.what() << endl;
            isTrained = false;
        }
    }
    
    /**
     * @brief Predicts target value for a single sample
     * @param features Feature vector for the sample
     * @return Predicted value
     */
    double predict(const vector<double>& features) {
        if (!isTrained) {
            cerr << "Error: Model not trained yet" << endl;
            return 0.0;
        }
        
        if (features.size() != numberOfFeatures) {
            cerr << "Error: Feature count mismatch" << endl;
            return 0.0;
        }
        
        // Calculate y = intercept + sum(coefficient[i] * feature[i])
        double prediction = intercept;
        for (int i = 0; i < numberOfFeatures; i++) {
            prediction += coefficients[i] * features[i];
        }
        
        return prediction;
    }
    
    /**
     * @brief Predicts target values for multiple samples
     * @param features 2D vector of feature vectors
     * @return Vector of predictions
     */
    vector<double> predictBatch(const vector<vector<double>>& features) {
        vector<double> predictions;
        for (const auto& sample : features) {
            predictions.push_back(predict(sample));
        }
        return predictions;
    }
    
    /**
     * @brief Evaluates the model on test data
     * @param features Test feature vectors
     * @param targets Actual target values
     */
    void evaluate(const vector<vector<double>>& features, const vector<double>& targets) {
        if (!isTrained) {
            cerr << "Error: Model not trained yet" << endl;
            return;
        }
        
        vector<double> predictions = predictBatch(features);
        
        double mse = calculateMSE(predictions, targets);
        double rmse = sqrt(mse);
        double r2 = calculateRSquared(predictions, targets);
        
        cout << "\n========== Model Evaluation ==========" << endl;
        cout << "Mean Squared Error (MSE): " << fixed << setprecision(4) << mse << endl;
        cout << "Root Mean Squared Error (RMSE): " << rmse << endl;
        cout << "R-squared (R²): " << r2 << endl;
        cout << "=====================================" << endl;
    }
    
    /**
     * @brief Displays the model parameters (coefficients and intercept)
     */
    void displayModelParameters() {
        if (!isTrained) {
            cerr << "Error: Model not trained yet" << endl;
            return;
        }
        
        cout << "\n========== Model Parameters ==========" << endl;
        cout << "Intercept (β₀): " << fixed << setprecision(4) << intercept << endl;
        cout << "Coefficients:" << endl;
        for (int i = 0; i < coefficients.size(); i++) {
            cout << "  β" << (i + 1) << ": " << coefficients[i] << endl;
        }
        
        // Display equation
        cout << "\nRegression Equation:" << endl;
        cout << "y = " << intercept;
        for (int i = 0; i < coefficients.size(); i++) {
            cout << " + (" << coefficients[i] << " * x" << (i + 1) << ")";
        }
        cout << endl;
        cout << "=====================================" << endl;
    }
    
    /**
     * @brief Gets the model coefficients
     * @return Vector of coefficients
     */
    vector<double> getCoefficients() const {
        return coefficients;
    }
    
    /**
     * @brief Gets the intercept term
     * @return Intercept value
     */
    double getIntercept() const {
        return intercept;
    }
    
    /**
     * @brief Displays predictions vs actual values
     * @param features Feature vectors
     * @param actualValues Actual target values
     */
    void displayPredictions(const vector<vector<double>>& features, const vector<double>& actualValues) {
        if (!isTrained) {
            cerr << "Error: Model not trained yet" << endl;
            return;
        }
        
        cout << "\n========== Predictions vs Actual ==========" << endl;
        cout << setw(8) << "Sample" << setw(15) << "Predicted" << setw(15) << "Actual" << setw(15) << "Error" << endl;
        cout << string(52, '-') << endl;
        
        for (size_t i = 0; i < features.size(); i++) {
            double predicted = predict(features[i]);
            double actual = actualValues[i];
            double error = predicted - actual;
            
            cout << setw(8) << (i + 1) 
                 << setw(15) << fixed << setprecision(4) << predicted
                 << setw(15) << actual
                 << setw(15) << error << endl;
        }
        cout << "==========================================" << endl;
    }
};

/**
 * @brief Main function demonstrating Multiple Linear Regression
 */
int main() {
    cout << "========== Multiple Linear Regression Demo ==========" << endl;
    
    // Sample dataset: House price prediction
    // Features: [Square Footage, Number of Bedrooms, Age of House]
    // Target: Price (in thousands of dollars)
    
    vector<vector<double>> trainingFeatures = {
        {1500, 3, 10},   // 1500 sq ft, 3 bedrooms, 10 years old
        {1600, 3, 8},
        {1700, 3, 7},
        {1800, 4, 5},
        {1900, 4, 3},
        {2000, 4, 2},
        {2100, 4, 1},
        {2200, 5, 1},
        {2300, 5, 0},
        {2400, 5, 0}
    };
    
    vector<double> trainingTargets = {
        250,  // $250,000
        260,
        270,
        295,
        310,
        330,
        350,
        380,
        400,
        420
    };
    
    // Create and train the model
    MultipleLinearRegression model;
    cout << "\nTraining the model on house price data..." << endl;
    model.train(trainingFeatures, trainingTargets);
    
    // Display learned parameters
    model.displayModelParameters();
    
    // Evaluate on training data
    cout << "\nEvaluating on training data:" << endl;
    model.evaluate(trainingFeatures, trainingTargets);
    
    // Display predictions
    model.displayPredictions(trainingFeatures, trainingTargets);
    
    // Test on new data
    cout << "\n========== Testing on New Data ==========" << endl;
    vector<vector<double>> testFeatures = {
        {1750, 3, 6},    // Test house 1
        {2150, 4, 2}     // Test house 2
    };
    
    for (size_t i = 0; i < testFeatures.size(); i++) {
        double predicted = model.predict(testFeatures[i]);
        cout << "House " << (i + 1) << ": ";
        cout << testFeatures[i][0] << " sq ft, ";
        cout << testFeatures[i][1] << " bedrooms, ";
        cout << testFeatures[i][2] << " years old";
        cout << " -> Predicted Price: $" << fixed << setprecision(2) << predicted << "k" << endl;
    }
    
    cout << "\n========== Demo Complete ==========" << endl;
    
    return 0;
}
