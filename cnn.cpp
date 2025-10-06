#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

/*----------------------------------------
  Helper type aliases
----------------------------------------*/
using Matrix = vector<vector<float>>;         // 2D Matrix
using Volume = vector<Matrix>;                // 3D volume (e.g., channels)

/*----------------------------------------
  Utility Functions
----------------------------------------*/

// Initialize a 2D matrix with random values
Matrix randomMatrix(int rows, int cols, float minVal = -1.0f, float maxVal = 1.0f) {
    Matrix mat(rows, vector<float>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i][j] = minVal + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxVal - minVal)));
    return mat;
}

// Print matrix for debugging
void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (float val : row) cout << val << "\t";
        cout << endl;
    }
    cout << endl;
}

/*----------------------------------------
  Convolution Layer
----------------------------------------*/
class ConvolutionLayer {
public:
    Matrix kernel;
    int kernelSize;
    float bias;

    ConvolutionLayer(int kSize) : kernelSize(kSize) {
        kernel = randomMatrix(kSize, kSize);
        bias = 0.0f;
    }

    // Apply convolution on a single channel input
    Matrix forward(const Matrix& input) {
        int outputRows = input.size() - kernelSize + 1;
        int outputCols = input[0].size() - kernelSize + 1;
        Matrix output(outputRows, vector<float>(outputCols, 0.0f));

        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize; kj++) {
                        sum += input[i + ki][j + kj] * kernel[ki][kj];
                    }
                }
                output[i][j] = sum + bias;
            }
        }
        return output;
    }
};

/*----------------------------------------
  ReLU Activation Layer
----------------------------------------*/
class ReLU {
public:
    Matrix forward(const Matrix& input) {
        Matrix output = input;
        for (auto& row : output)
            for (auto& val : row)
                val = max(0.0f, val);
        return output;
    }
};

/*----------------------------------------
  Max Pooling Layer
----------------------------------------*/
class MaxPool {
public:
    int poolSize;
    MaxPool(int size = 2) : poolSize(size) {}

    Matrix forward(const Matrix& input) {
        int outputRows = input.size() / poolSize;
        int outputCols = input[0].size() / poolSize;
        Matrix output(outputRows, vector<float>(outputCols, 0.0f));

        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                float maxVal = -INFINITY;
                for (int pi = 0; pi < poolSize; pi++) {
                    for (int pj = 0; pj < poolSize; pj++) {
                        maxVal = max(maxVal, input[i * poolSize + pi][j * poolSize + pj]);
                    }
                }
                output[i][j] = maxVal;
            }
        }
        return output;
    }
};

/*----------------------------------------
  Simple CNN Model
----------------------------------------*/
class SimpleCNN {
private:
    ConvolutionLayer conv1;
    ReLU relu1;
    MaxPool pool1;

public:
    SimpleCNN() : conv1(3), pool1(2) {}

    // Forward pass of the network
    Matrix forward(const Matrix& input) {
        cout << "Input:\n";
        printMatrix(input);

        Matrix convOutput = conv1.forward(input);
        cout << "After Convolution:\n";
        printMatrix(convOutput);

        Matrix reluOutput = relu1.forward(convOutput);
        cout << "After ReLU:\n";
        printMatrix(reluOutput);

        Matrix pooledOutput = pool1.forward(reluOutput);
        cout << "After Max Pooling:\n";
        printMatrix(pooledOutput);

        return pooledOutput;
    }
};

/*----------------------------------------
  Main Function (Demo)
----------------------------------------*/
int main() {
    srand(static_cast<unsigned>(time(0)));

    // Example 5x5 input matrix
    Matrix input = {
        {1, 0, 2, 3, 1},
        {4, 6, 6, 8, 4},
        {3, 1, 1, 0, 2},
        {1, 2, 2, 4, 5},
        {0, 1, 3, 1, 1}
    };

    SimpleCNN cnn;
    cnn.forward(input);

    return 0;
}
