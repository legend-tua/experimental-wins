#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation function (Sigmoid) and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Neural Network Class
class NeuralNetwork {
public:
    int input_nodes, hidden_nodes, output_nodes;
    vector<vector<double>> weights_input_hidden, weights_hidden_output;
    vector<double> hidden_layer, output_layer;
    vector<double> hidden_bias, output_bias;

    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
        this->input_nodes = input_nodes;
        this->hidden_nodes = hidden_nodes;
        this->output_nodes = output_nodes;

        // Initialize weights and biases with random values
        weights_input_hidden = random_matrix(input_nodes, hidden_nodes);
        weights_hidden_output = random_matrix(hidden_nodes, output_nodes);

        hidden_bias = random_vector(hidden_nodes);
        output_bias = random_vector(output_nodes);
    }

    // Feedforward function
    vector<double> feedforward(vector<double>& input_data) {
        hidden_layer = apply_activation(dot_product(input_data, weights_input_hidden, hidden_bias));
        output_layer = apply_activation(dot_product(hidden_layer, weights_hidden_output, output_bias));
        return output_layer;
    }

private:
    // Helper function to generate random matrix
    vector<vector<double>> random_matrix(int rows, int cols) {
        vector<vector<double>> matrix(rows, vector<double>(cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = (rand() % 100) / 100.0;  // Random values between 0 and 1
        return matrix;
    }

    // Helper function to generate random vector
    vector<double> random_vector(int size) {
        vector<double> vec(size);
        for (int i = 0; i < size; i++)
            vec[i] = (rand() % 100) / 100.0;  // Random values between 0 and 1
        return vec;
    }

    // Function to compute dot product and add bias
    vector<double> dot_product(vector<double>& vec, vector<vector<double>>& matrix, vector<double>& bias) {
        vector<double> result(matrix[0].size(), 0.0);
        for (int i = 0; i < matrix[0].size(); i++) {
            for (int j = 0; j < vec.size(); j++) {
                result[i] += vec[j] * matrix[j][i];
            }
            result[i] += bias[i];  // Add bias
        }
        return result;
    }

    // Apply activation function
    vector<double> apply_activation(vector<double> vec) {
        vector<double> activated_vec(vec.size());
        for (int i = 0; i < vec.size(); i++) {
            activated_vec[i] = sigmoid(vec[i]);
        }
        return activated_vec;
    }
};

int main() {
    srand(time(0));  // Initialize random seed

    // Create a neural network with 3 input nodes, 3 hidden nodes, and 1 output node
    NeuralNetwork nn(3, 3, 1);

    // Input data (e.g., XOR-like pattern)
    vector<double> input_data = {1.0, 0.5, -1.5};

    // Perform feedforward operation
    vector<double> output = nn.feedforward(input_data);

    // Print the output
    cout << "Output: ";
    for (double val : output) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
