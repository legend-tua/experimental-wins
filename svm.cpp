/**
 * @file svm.cpp
 * @brief Support Vector Machine (SVM) implementation for classification and regression
 * @author AI Assistant
 * @date October 6, 2025
 * 
 * This file contains implementations of:
 * 1. SVM Classifier (SVMClassifier) - for binary classification tasks
 * 2. SVM Regressor (SVMRegressor) - for regression tasks
 * 
 * Both implementations use the Sequential Minimal Optimization (SMO) algorithm
 * for training and support multiple kernel functions.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>

// ========================================================================================
// KERNEL FUNCTIONS
// ========================================================================================

/**
 * @brief Enum for different kernel types
 */
enum class KernelType {
    LINEAR,      // K(x, y) = x^T * y
    POLYNOMIAL,  // K(x, y) = (gamma * x^T * y + coef0)^degree
    RBF,         // K(x, y) = exp(-gamma * ||x - y||^2)
    SIGMOID      // K(x, y) = tanh(gamma * x^T * y + coef0)
};

/**
 * @class Kernel
 * @brief Base class for kernel functions used in SVM
 */
class Kernel {
private:
    KernelType m_kernelType;
    double m_gamma;      // Kernel coefficient for RBF, polynomial and sigmoid
    double m_coef0;      // Independent term in polynomial and sigmoid
    int m_degree;        // Degree of polynomial kernel

public:
    /**
     * @brief Constructor for Kernel
     * @param type Type of kernel function
     * @param gamma Kernel coefficient (default: 1.0)
     * @param coef0 Independent term (default: 0.0)
     * @param degree Polynomial degree (default: 3)
     */
    Kernel(KernelType type = KernelType::RBF, double gamma = 1.0, 
           double coef0 = 0.0, int degree = 3)
        : m_kernelType(type), m_gamma(gamma), m_coef0(coef0), m_degree(degree) {}

    /**
     * @brief Compute dot product between two vectors
     * @param x First vector
     * @param y Second vector
     * @return Dot product value
     */
    double dotProduct(const std::vector<double>& x, const std::vector<double>& y) const {
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    /**
     * @brief Compute squared Euclidean distance between two vectors
     * @param x First vector
     * @param y Second vector
     * @return Squared distance
     */
    double squaredDistance(const std::vector<double>& x, const std::vector<double>& y) const {
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * @brief Compute kernel function value
     * @param x First vector
     * @param y Second vector
     * @return Kernel value
     */
    double compute(const std::vector<double>& x, const std::vector<double>& y) const {
        switch (m_kernelType) {
            case KernelType::LINEAR:
                return dotProduct(x, y);
            
            case KernelType::POLYNOMIAL: {
                double dot = dotProduct(x, y);
                return std::pow(m_gamma * dot + m_coef0, m_degree);
            }
            
            case KernelType::RBF: {
                double sqDist = squaredDistance(x, y);
                return std::exp(-m_gamma * sqDist);
            }
            
            case KernelType::SIGMOID: {
                double dot = dotProduct(x, y);
                return std::tanh(m_gamma * dot + m_coef0);
            }
            
            default:
                return dotProduct(x, y);
        }
    }
};

// ========================================================================================
// SVM CLASSIFIER
// ========================================================================================

/**
 * @class SVMClassifier
 * @brief Support Vector Machine for binary classification
 * 
 * This class implements a binary SVM classifier using the SMO algorithm.
 * It supports multiple kernel functions and handles the optimization of
 * Lagrange multipliers (alphas) to find the optimal separating hyperplane.
 */
class SVMClassifier {
private:
    // Training data
    std::vector<std::vector<double>> m_trainX;  // Training features
    std::vector<int> m_trainY;                   // Training labels (-1 or +1)
    
    // Model parameters
    std::vector<double> m_alpha;                 // Lagrange multipliers
    double m_bias;                               // Bias term (b)
    
    // Hyperparameters
    double m_C;                                  // Regularization parameter
    double m_tolerance;                          // Tolerance for KKT conditions
    int m_maxIterations;                         // Maximum training iterations
    
    // Kernel
    Kernel m_kernel;
    
    // Error cache for SMO algorithm
    std::vector<double> m_errorCache;

    /**
     * @brief Compute prediction error for a sample
     * @param i Index of the sample
     * @return Error value (prediction - actual)
     */
    double computeError(int i) const {
        double prediction = computeDecisionFunction(i);
        return prediction - m_trainY[i];
    }

    /**
     * @brief Compute decision function for a training sample
     * @param i Index of the sample
     * @return Decision function value
     */
    double computeDecisionFunction(int i) const {
        double sum = m_bias;
        for (size_t j = 0; j < m_trainX.size(); ++j) {
            if (m_alpha[j] > 0) {
                sum += m_alpha[j] * m_trainY[j] * 
                       m_kernel.compute(m_trainX[j], m_trainX[i]);
            }
        }
        return sum;
    }

    /**
     * @brief Check if alpha violates KKT conditions
     * @param i Index of the sample
     * @return True if KKT conditions are violated
     */
    bool violatesKKT(int i) const {
        double yi = m_trainY[i];
        double ei = m_errorCache[i];
        double ri = ei * yi;
        
        return ((ri < -m_tolerance && m_alpha[i] < m_C) || 
                (ri > m_tolerance && m_alpha[i] > 0));
    }

    /**
     * @brief Select second alpha for optimization using heuristic
     * @param i1 Index of first alpha
     * @param e1 Error for first alpha
     * @return Index of second alpha
     */
    int selectSecondAlpha(int i1, double e1) const {
        int i2 = -1;
        double maxDelta = 0.0;
        
        // Find alpha that maximizes |E1 - E2|
        for (size_t i = 0; i < m_trainX.size(); ++i) {
            if (m_alpha[i] > 0 && m_alpha[i] < m_C) {
                double e2 = m_errorCache[i];
                double delta = std::abs(e1 - e2);
                if (delta > maxDelta) {
                    maxDelta = delta;
                    i2 = i;
                }
            }
        }
        
        // If no suitable alpha found, select randomly
        if (i2 == -1) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, m_trainX.size() - 1);
            do {
                i2 = dis(gen);
            } while (i2 == i1);
        }
        
        return i2;
    }

    /**
     * @brief Optimize pair of alphas using SMO algorithm
     * @param i1 Index of first alpha
     * @param i2 Index of second alpha
     * @return True if optimization was successful
     */
    bool optimizeAlphaPair(int i1, int i2) {
        if (i1 == i2) return false;
        
        double alpha1 = m_alpha[i1];
        double alpha2 = m_alpha[i2];
        int y1 = m_trainY[i1];
        int y2 = m_trainY[i2];
        double e1 = m_errorCache[i1];
        double e2 = m_errorCache[i2];
        
        // Compute bounds for alpha2
        double L, H;
        if (y1 != y2) {
            L = std::max(0.0, alpha2 - alpha1);
            H = std::min(m_C, m_C + alpha2 - alpha1);
        } else {
            L = std::max(0.0, alpha1 + alpha2 - m_C);
            H = std::min(m_C, alpha1 + alpha2);
        }
        
        if (L >= H) return false;
        
        // Compute eta (second derivative of objective function)
        double k11 = m_kernel.compute(m_trainX[i1], m_trainX[i1]);
        double k12 = m_kernel.compute(m_trainX[i1], m_trainX[i2]);
        double k22 = m_kernel.compute(m_trainX[i2], m_trainX[i2]);
        double eta = 2.0 * k12 - k11 - k22;
        
        double alpha2New;
        if (eta < 0) {
            // Normal case
            alpha2New = alpha2 - y2 * (e1 - e2) / eta;
            // Clip alpha2
            if (alpha2New >= H) alpha2New = H;
            else if (alpha2New <= L) alpha2New = L;
        } else {
            // Unusual case, move to boundary
            alpha2New = (std::abs(alpha2 - L) < std::abs(alpha2 - H)) ? L : H;
        }
        
        // Check if change is significant
        if (std::abs(alpha2New - alpha2) < 1e-5) return false;
        
        // Compute alpha1New
        double alpha1New = alpha1 + y1 * y2 * (alpha2 - alpha2New);
        
        // Update bias term
        double b1 = m_bias - e1 - y1 * (alpha1New - alpha1) * k11 
                    - y2 * (alpha2New - alpha2) * k12;
        double b2 = m_bias - e2 - y1 * (alpha1New - alpha1) * k12 
                    - y2 * (alpha2New - alpha2) * k22;
        
        if (alpha1New > 0 && alpha1New < m_C) {
            m_bias = b1;
        } else if (alpha2New > 0 && alpha2New < m_C) {
            m_bias = b2;
        } else {
            m_bias = (b1 + b2) / 2.0;
        }
        
        // Update alphas
        m_alpha[i1] = alpha1New;
        m_alpha[i2] = alpha2New;
        
        // Update error cache
        updateErrorCache();
        
        return true;
    }

    /**
     * @brief Update error cache for all samples
     */
    void updateErrorCache() {
        for (size_t i = 0; i < m_trainX.size(); ++i) {
            m_errorCache[i] = computeError(i);
        }
    }

public:
    /**
     * @brief Constructor for SVMClassifier
     * @param C Regularization parameter (default: 1.0)
     * @param tolerance Tolerance for KKT conditions (default: 1e-3)
     * @param maxIterations Maximum training iterations (default: 1000)
     * @param kernelType Type of kernel function (default: RBF)
     * @param gamma Kernel coefficient (default: 1.0)
     */
    SVMClassifier(double C = 1.0, double tolerance = 1e-3, int maxIterations = 1000,
                  KernelType kernelType = KernelType::RBF, double gamma = 1.0)
        : m_C(C), m_tolerance(tolerance), m_maxIterations(maxIterations),
          m_kernel(kernelType, gamma), m_bias(0.0) {}

    /**
     * @brief Train the SVM classifier
     * @param X Training features (n_samples x n_features)
     * @param y Training labels (n_samples, values: -1 or +1)
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data");
        }
        
        // Validate labels are -1 or +1
        for (int label : y) {
            if (label != -1 && label != 1) {
                throw std::invalid_argument("Labels must be -1 or +1");
            }
        }
        
        m_trainX = X;
        m_trainY = y;
        
        // Initialize alphas and bias
        m_alpha.assign(X.size(), 0.0);
        m_bias = 0.0;
        
        // Initialize error cache
        m_errorCache.resize(X.size());
        updateErrorCache();
        
        // SMO algorithm
        int iteration = 0;
        int numChanged = 0;
        bool examineAll = true;
        
        while ((numChanged > 0 || examineAll) && iteration < m_maxIterations) {
            numChanged = 0;
            
            if (examineAll) {
                // Examine all samples
                for (size_t i = 0; i < X.size(); ++i) {
                    if (violatesKKT(i)) {
                        int i2 = selectSecondAlpha(i, m_errorCache[i]);
                        if (optimizeAlphaPair(i, i2)) {
                            numChanged++;
                        }
                    }
                }
            } else {
                // Examine non-bound samples only
                for (size_t i = 0; i < X.size(); ++i) {
                    if (m_alpha[i] > 0 && m_alpha[i] < m_C) {
                        if (violatesKKT(i)) {
                            int i2 = selectSecondAlpha(i, m_errorCache[i]);
                            if (optimizeAlphaPair(i, i2)) {
                                numChanged++;
                            }
                        }
                    }
                }
            }
            
            if (examineAll) {
                examineAll = false;
            } else if (numChanged == 0) {
                examineAll = true;
            }
            
            iteration++;
        }
        
        std::cout << "Training completed in " << iteration << " iterations" << std::endl;
        std::cout << "Number of support vectors: " << getSupportVectorCount() << std::endl;
    }

    /**
     * @brief Predict class label for a single sample
     * @param x Feature vector
     * @return Predicted class label (-1 or +1)
     */
    int predict(const std::vector<double>& x) const {
        double decision = predictDecisionFunction(x);
        return (decision >= 0) ? 1 : -1;
    }

    /**
     * @brief Predict class labels for multiple samples
     * @param X Feature vectors (n_samples x n_features)
     * @return Predicted class labels
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<int> predictions;
        predictions.reserve(X.size());
        for (const auto& x : X) {
            predictions.push_back(predict(x));
        }
        return predictions;
    }

    /**
     * @brief Compute decision function for a sample
     * @param x Feature vector
     * @return Decision function value
     */
    double predictDecisionFunction(const std::vector<double>& x) const {
        double sum = m_bias;
        for (size_t i = 0; i < m_trainX.size(); ++i) {
            if (m_alpha[i] > 0) {
                sum += m_alpha[i] * m_trainY[i] * m_kernel.compute(m_trainX[i], x);
            }
        }
        return sum;
    }

    /**
     * @brief Calculate accuracy on test data
     * @param X Test features
     * @param y True labels
     * @return Accuracy score (0 to 1)
     */
    double score(const std::vector<std::vector<double>>& X, 
                 const std::vector<int>& y) const {
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have same size");
        }
        
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            if (predict(X[i]) == y[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / X.size();
    }

    /**
     * @brief Get number of support vectors
     * @return Count of support vectors
     */
    int getSupportVectorCount() const {
        int count = 0;
        for (double alpha : m_alpha) {
            if (alpha > 0) count++;
        }
        return count;
    }

    /**
     * @brief Get support vectors
     * @return Vector of support vectors
     */
    std::vector<std::vector<double>> getSupportVectors() const {
        std::vector<std::vector<double>> supportVectors;
        for (size_t i = 0; i < m_alpha.size(); ++i) {
            if (m_alpha[i] > 0) {
                supportVectors.push_back(m_trainX[i]);
            }
        }
        return supportVectors;
    }
};

// ========================================================================================
// SVM REGRESSOR
// ========================================================================================

/**
 * @class SVMRegressor
 * @brief Support Vector Machine for regression (SVR)
 * 
 * This class implements epsilon-SVR using the SMO algorithm.
 * It uses an epsilon-insensitive loss function and supports
 * multiple kernel functions for non-linear regression.
 */
class SVMRegressor {
private:
    // Training data
    std::vector<std::vector<double>> m_trainX;  // Training features
    std::vector<double> m_trainY;                // Training targets
    
    // Model parameters
    std::vector<double> m_alpha;                 // Lagrange multipliers (alpha+)
    std::vector<double> m_alphaStar;             // Lagrange multipliers (alpha-)
    double m_bias;                               // Bias term (b)
    
    // Hyperparameters
    double m_C;                                  // Regularization parameter
    double m_epsilon;                            // Epsilon in epsilon-SVR
    double m_tolerance;                          // Tolerance for optimization
    int m_maxIterations;                         // Maximum training iterations
    
    // Kernel
    Kernel m_kernel;
    
    // Error cache
    std::vector<double> m_errorCache;

    /**
     * @brief Compute prediction error for a sample
     * @param i Index of the sample
     * @return Error value
     */
    double computeError(int i) const {
        double prediction = computeDecisionFunction(i);
        return prediction - m_trainY[i];
    }

    /**
     * @brief Compute decision function for a training sample
     * @param i Index of the sample
     * @return Predicted value
     */
    double computeDecisionFunction(int i) const {
        double sum = m_bias;
        for (size_t j = 0; j < m_trainX.size(); ++j) {
            double coef = m_alpha[j] - m_alphaStar[j];
            if (std::abs(coef) > 0) {
                sum += coef * m_kernel.compute(m_trainX[j], m_trainX[i]);
            }
        }
        return sum;
    }

    /**
     * @brief Update error cache for all samples
     */
    void updateErrorCache() {
        for (size_t i = 0; i < m_trainX.size(); ++i) {
            m_errorCache[i] = computeError(i);
        }
    }

    /**
     * @brief Simplified SMO optimization step
     * @param i1 Index of first sample
     * @param i2 Index of second sample
     * @return True if optimization was successful
     */
    bool optimizeAlphaPair(int i1, int i2) {
        if (i1 == i2) return false;
        
        double alpha1 = m_alpha[i1];
        double alphaStar1 = m_alphaStar[i1];
        double alpha2 = m_alpha[i2];
        double alphaStar2 = m_alphaStar[i2];
        
        double e1 = m_errorCache[i1];
        double e2 = m_errorCache[i2];
        
        // Compute kernel values
        double k11 = m_kernel.compute(m_trainX[i1], m_trainX[i1]);
        double k12 = m_kernel.compute(m_trainX[i1], m_trainX[i2]);
        double k22 = m_kernel.compute(m_trainX[i2], m_trainX[i2]);
        double eta = k11 + k22 - 2.0 * k12;
        
        if (eta <= 0) return false;
        
        // Update alpha for sample i1 (simplified approach)
        double deltaAlpha1 = (e1 - m_epsilon) / eta;
        double alpha1New = alpha1 + deltaAlpha1;
        alpha1New = std::max(0.0, std::min(m_C, alpha1New));
        
        double deltaAlphaStar1 = (-e1 - m_epsilon) / eta;
        double alphaStar1New = alphaStar1 + deltaAlphaStar1;
        alphaStar1New = std::max(0.0, std::min(m_C, alphaStar1New));
        
        // Check if change is significant
        if (std::abs(alpha1New - alpha1) < 1e-5 && 
            std::abs(alphaStar1New - alphaStar1) < 1e-5) {
            return false;
        }
        
        // Update alphas
        m_alpha[i1] = alpha1New;
        m_alphaStar[i1] = alphaStar1New;
        
        // Update bias
        double bOld = m_bias;
        if (alpha1New > 0 && alpha1New < m_C) {
            m_bias = m_trainY[i1] - m_epsilon - computeDecisionFunction(i1) + bOld;
        } else if (alphaStar1New > 0 && alphaStar1New < m_C) {
            m_bias = m_trainY[i1] + m_epsilon - computeDecisionFunction(i1) + bOld;
        }
        
        // Update error cache
        updateErrorCache();
        
        return true;
    }

public:
    /**
     * @brief Constructor for SVMRegressor
     * @param C Regularization parameter (default: 1.0)
     * @param epsilon Epsilon in epsilon-SVR (default: 0.1)
     * @param tolerance Tolerance for optimization (default: 1e-3)
     * @param maxIterations Maximum training iterations (default: 1000)
     * @param kernelType Type of kernel function (default: RBF)
     * @param gamma Kernel coefficient (default: 1.0)
     */
    SVMRegressor(double C = 1.0, double epsilon = 0.1, double tolerance = 1e-3,
                 int maxIterations = 1000, KernelType kernelType = KernelType::RBF,
                 double gamma = 1.0)
        : m_C(C), m_epsilon(epsilon), m_tolerance(tolerance),
          m_maxIterations(maxIterations), m_kernel(kernelType, gamma), m_bias(0.0) {}

    /**
     * @brief Train the SVM regressor
     * @param X Training features (n_samples x n_features)
     * @param y Training targets (n_samples)
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data");
        }
        
        m_trainX = X;
        m_trainY = y;
        
        // Initialize alphas and bias
        m_alpha.assign(X.size(), 0.0);
        m_alphaStar.assign(X.size(), 0.0);
        m_bias = 0.0;
        
        // Initialize error cache
        m_errorCache.resize(X.size());
        
        // Compute initial bias as mean of targets
        double sumY = 0.0;
        for (double val : y) {
            sumY += val;
        }
        m_bias = sumY / y.size();
        
        updateErrorCache();
        
        // Simplified SMO algorithm for regression
        int iteration = 0;
        bool changed = true;
        
        while (changed && iteration < m_maxIterations) {
            changed = false;
            
            for (size_t i = 0; i < X.size(); ++i) {
                double error = m_errorCache[i];
                
                // Check if sample violates epsilon-insensitive condition
                if (std::abs(error) > m_epsilon + m_tolerance) {
                    // Find second sample with maximum error difference
                    int j = -1;
                    double maxDiff = 0.0;
                    for (size_t k = 0; k < X.size(); ++k) {
                        if (k != i) {
                            double diff = std::abs(error - m_errorCache[k]);
                            if (diff > maxDiff) {
                                maxDiff = diff;
                                j = k;
                            }
                        }
                    }
                    
                    if (j >= 0) {
                        if (optimizeAlphaPair(i, j)) {
                            changed = true;
                        }
                    }
                }
            }
            
            iteration++;
        }
        
        std::cout << "Training completed in " << iteration << " iterations" << std::endl;
        std::cout << "Number of support vectors: " << getSupportVectorCount() << std::endl;
    }

    /**
     * @brief Predict target value for a single sample
     * @param x Feature vector
     * @return Predicted target value
     */
    double predict(const std::vector<double>& x) const {
        double sum = m_bias;
        for (size_t i = 0; i < m_trainX.size(); ++i) {
            double coef = m_alpha[i] - m_alphaStar[i];
            if (std::abs(coef) > 0) {
                sum += coef * m_kernel.compute(m_trainX[i], x);
            }
        }
        return sum;
    }

    /**
     * @brief Predict target values for multiple samples
     * @param X Feature vectors (n_samples x n_features)
     * @return Predicted target values
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<double> predictions;
        predictions.reserve(X.size());
        for (const auto& x : X) {
            predictions.push_back(predict(x));
        }
        return predictions;
    }

    /**
     * @brief Calculate R² score on test data
     * @param X Test features
     * @param y True targets
     * @return R² score
     */
    double score(const std::vector<std::vector<double>>& X, 
                 const std::vector<double>& y) const {
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have same size");
        }
        
        // Calculate mean of y
        double meanY = 0.0;
        for (double val : y) {
            meanY += val;
        }
        meanY /= y.size();
        
        // Calculate total sum of squares and residual sum of squares
        double ssTot = 0.0;
        double ssRes = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            double pred = predict(X[i]);
            ssRes += (y[i] - pred) * (y[i] - pred);
            ssTot += (y[i] - meanY) * (y[i] - meanY);
        }
        
        // R² score
        return 1.0 - (ssRes / ssTot);
    }

    /**
     * @brief Calculate Mean Squared Error
     * @param X Test features
     * @param y True targets
     * @return MSE value
     */
    double meanSquaredError(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y) const {
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have same size");
        }
        
        double mse = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            double error = y[i] - predict(X[i]);
            mse += error * error;
        }
        return mse / X.size();
    }

    /**
     * @brief Get number of support vectors
     * @return Count of support vectors
     */
    int getSupportVectorCount() const {
        int count = 0;
        for (size_t i = 0; i < m_alpha.size(); ++i) {
            if (m_alpha[i] > 0 || m_alphaStar[i] > 0) {
                count++;
            }
        }
        return count;
    }

    /**
     * @brief Get support vectors
     * @return Vector of support vectors
     */
    std::vector<std::vector<double>> getSupportVectors() const {
        std::vector<std::vector<double>> supportVectors;
        for (size_t i = 0; i < m_alpha.size(); ++i) {
            if (m_alpha[i] > 0 || m_alphaStar[i] > 0) {
                supportVectors.push_back(m_trainX[i]);
            }
        }
        return supportVectors;
    }
};

// ========================================================================================
// DEMONSTRATION AND TESTING
// ========================================================================================

/**
 * @brief Generate synthetic linearly separable data for classification
 * @param n Number of samples
 * @return Pair of features and labels
 */
std::pair<std::vector<std::vector<double>>, std::vector<int>> 
generateClassificationData(int n = 100) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < n / 2; ++i) {
        // Class +1
        std::vector<double> x1 = {dis(gen) + 2.0, dis(gen) + 2.0};
        X.push_back(x1);
        y.push_back(1);
        
        // Class -1
        std::vector<double> x2 = {dis(gen) - 2.0, dis(gen) - 2.0};
        X.push_back(x2);
        y.push_back(-1);
    }
    
    return {X, y};
}

/**
 * @brief Generate synthetic data for regression
 * @param n Number of samples
 * @return Pair of features and targets
 */
std::pair<std::vector<std::vector<double>>, std::vector<double>> 
generateRegressionData(int n = 100) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 0.5);
    
    for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i) / n * 10.0 - 5.0;
        // y = sin(x) + noise
        double target = std::sin(x) + noise(gen);
        
        X.push_back({x});
        y.push_back(target);
    }
    
    return {X, y};
}

/**
 * @brief Main function demonstrating SVM usage
 */
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "SVM CLASSIFIER DEMONSTRATION" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Generate classification data
    auto [X_class, y_class] = generateClassificationData(100);
    
    // Create and train SVM classifier
    SVMClassifier svm_classifier(1.0, 1e-3, 1000, KernelType::RBF, 0.5);
    
    std::cout << "\nTraining SVM Classifier..." << std::endl;
    svm_classifier.fit(X_class, y_class);
    
    // Test predictions
    std::cout << "\nTesting predictions on training data:" << std::endl;
    double accuracy = svm_classifier.score(X_class, y_class);
    std::cout << "Training Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    // Test on few samples
    std::cout << "\nSample predictions:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        int pred = svm_classifier.predict(X_class[i]);
        std::cout << "Sample " << i << ": True=" << y_class[i] 
                  << ", Predicted=" << pred << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "SVM REGRESSOR DEMONSTRATION" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Generate regression data
    auto [X_reg, y_reg] = generateRegressionData(100);
    
    // Create and train SVM regressor
    SVMRegressor svm_regressor(1.0, 0.1, 1e-3, 1000, KernelType::RBF, 0.1);
    
    std::cout << "\nTraining SVM Regressor..." << std::endl;
    svm_regressor.fit(X_reg, y_reg);
    
    // Test predictions
    std::cout << "\nTesting predictions on training data:" << std::endl;
    double r2Score = svm_regressor.score(X_reg, y_reg);
    double mse = svm_regressor.meanSquaredError(X_reg, y_reg);
    std::cout << "R² Score: " << r2Score << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    
    // Test on few samples
    std::cout << "\nSample predictions:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        double pred = svm_regressor.predict(X_reg[i]);
        std::cout << "Sample " << i << ": True=" << y_reg[i] 
                  << ", Predicted=" << pred << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEMONSTRATION COMPLETED" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
