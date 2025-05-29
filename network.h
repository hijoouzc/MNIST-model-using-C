#ifndef NETWORK_H
#define NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Định nghĩa các thông số của mạng nơ-ron
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define HIDDEN_LAYER1 64
#define HIDDEN_LAYER2 32
#define EPOCHS 30
#define LEARNING_RATE 0.01
#define BATCH_SIZE 32


// Định nghĩa cấu trúc của một lớp 
typedef struct Layer {
    int input_size;
    int output_size;
    double** weights;
    double* biases;
    double* z; // z^l = w^l * a^{l-1} + b^l (tổng trọng số trước khi kích hoạt)
    double *output;
    double* delta;
    
    double** weight_grads;  // Biến tích lũy gradient
    double* bias_grads;
} Layer;

// Định nghĩa cấu trúc của mạng nơ-ron
typedef struct MLP {
    int num_layers;
    Layer** layers;
} MLP;

// Hàm sinh số ngẫu nhiên trong khoảng [-1, 1]
double rand_uniform() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

// Hàm kích hoạt ReLU
double reLu(double x) {
    return x > 0 ? x : 0.0;
}
// Hàm đạo hàm của ReLU
double reLu_prime(double x) {
    return x > 0 ? 1.0 : 0.0;
}
// Hàm kích hoạt sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
// Hàm đạo hàm của sigmoid
double sigmoid_prime(double x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}
// Hàm kích hoạt softmax
void softmax(double* x, int size) {
    double max = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max) max = x[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

// Hàm tạo một lớp nơ-ron
Layer* create_layer(int input_size, int output_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    layer->weights = (double**)malloc(output_size * sizeof(double*));
    layer->weight_grads = (double**)malloc(output_size * sizeof(double*));
    for (int i = 0; i < output_size; ++i) {
        layer->weights[i] = (double*)malloc(input_size * sizeof(double));
        layer->weight_grads[i] = (double*)calloc(input_size, sizeof(double));

        for (int j = 0; j < input_size; ++j) {
            layer->weights[i][j] = rand_uniform() * sqrt(2.0 / input_size); // He initialization
        }
    }
    
    layer->biases = (double*)calloc(output_size, sizeof(double));
    layer->bias_grads = (double*)calloc(output_size, sizeof(double));
    layer->z = (double*)malloc(output_size * sizeof(double)); // Cấp phát cho z
    layer->output = (double*)malloc(output_size * sizeof(double));
    layer->delta = (double*)malloc(output_size * sizeof(double));
    
    return layer;
}

// Hàm giải phóng một lớp nơ-ron
void free_layer(Layer* layer) {
    for (int i = 0; i < layer->output_size; ++i) {
        free(layer->weights[i]);
        free(layer->weight_grads[i]);
    }
    free(layer->weights);
    free(layer->weight_grads);
    free(layer->biases);
    free(layer->bias_grads);
    free(layer->z);
    free(layer->output);
    free(layer->delta);
    free(layer);
}

// Hàm tạo mạng nơ-ron
MLP* create_mlp() {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->num_layers = 3;
    
    mlp->layers = (Layer**)malloc(mlp->num_layers * sizeof(Layer*));
    mlp->layers[0] = create_layer(INPUT_SIZE, HIDDEN_LAYER1);
    mlp->layers[1] = create_layer(HIDDEN_LAYER1, HIDDEN_LAYER2);
    mlp->layers[2] = create_layer(HIDDEN_LAYER2, OUTPUT_SIZE);
    
    return mlp;
}

// Hàm giải phóng mạng nơ-ron
void free_mlp(MLP* mlp) {
    for (int i = 0; i < mlp->num_layers; ++i) {
        free_layer(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

// Hàm forward
void forward(MLP* mlp, double* input) {
    for (int i = 0; i < mlp->num_layers; ++i) {
        Layer* layer = mlp->layers[i];
        
        for (int j = 0; j < layer->output_size; ++j) {
            layer->z[j] = layer->biases[j];
            for (int k = 0; k < layer->input_size; ++k) {
                layer->z[j] += input[k] * layer->weights[j][k]; // Tính toán đầu ra
            }
            // Áp dụng hàm kích hoạt cho các lớp ẩn
            layer->output[j] = (i < mlp->num_layers - 1) ? reLu(layer->z[j]) : layer->z[j]; 
        }
        // Áp dụng hàm kích hoạt cho lớp đầu ra
        if (i == mlp->num_layers - 1) {
            softmax(layer->output, layer->output_size);
        }

        input = layer->output;
    }
}

// Hàm backward
void backward(MLP* mlp, double* target) {
    Layer* output_layer = mlp->layers[mlp->num_layers - 1];
    // Bước 1: Tính delta cho lớp đầu ra (L) 
    // Với Softmax và Cross-Entropy Loss, delta^L = a^L - y
    for (int i = 0; i < output_layer->output_size; ++i) {
        output_layer->delta[i] = (output_layer->output[i] - target[i]) ;
    }
    
    // Bước 2: Lan truyền lỗi ngược qua các lớp ẩn (từ L-1 xuống 0) 
    // l đi từ mlp->num_layers - 2 (lớp ẩn cuối cùng) đến 0 (lớp ẩn đầu tiên)
    for (int i = mlp->num_layers - 2; i >= 0; --i) {
        Layer* current = mlp->layers[i]; 
        Layer* next = mlp->layers[i + 1];
        
        for (int j = 0; j < current->output_size; ++j) { // Duyệt qua nơ-ron j ở lớp l
            double error = 0.0;
            for (int k = 0; k < next->output_size; ++k) { // Duyệt qua nơ-ron k ở lớp l+1
                // next_layer->weights[k][j] là w^{l+1}_{kj}
                // next_layer->delta[k] là delta^{l+1}_k
                error += next->weights[k][j] * next->delta[k];
            }

            current->delta[j] = error * reLu_prime(current->z[j]); // Đạo hàm của hàm kích hoạt
        }
    }
}

// Tích lũy gradient cho các trọng số và độ lệch
void accumulate_gradients(MLP* mlp, double* input) {
    for (int i = mlp->num_layers - 1; i >= 0; --i) {
        Layer* layer = mlp->layers[i];
        // prev_output_activations là a^{l-1}
        // Nếu i == 0 (lớp đầu tiên của mạng), prev_output của nó là original_input_data (a^0)
        // Nếu không, prev_output là output của lớp trước đó (mlp->layers[i - 1]->output)
        double* prev_output = (i == 0) ? input : mlp->layers[i - 1]->output;
        
        for (int j = 0; j < layer->output_size; ++j) { // Duyệt qua nơ-ron j ở lớp l (hiện tại)
            for (int k = 0; k < layer->input_size; ++k) { // Duyệt qua nơ-ron k ở lớp l-1 (trước đó)
                // layer->delta[j] là delta^l_j
                // prev_output[k] là a^{l-1}_k
                layer->weight_grads[j][k] += layer->delta[j] * prev_output[k];
            }
            layer->bias_grads[j] += layer->delta[j];
        }
    }
}

// Update weights and biases 
void update_weights_biases(MLP* mlp, double learning_rate, int batch_size) {
    for (int i = 0; i < mlp->num_layers; ++i) {
        Layer* layer = mlp->layers[i];
        
        for (int j = 0; j < layer->output_size; ++j) {
            for (int k = 0; k < layer->input_size; ++k) {
                layer->weights[j][k] -= learning_rate * (layer->weight_grads[j][k] / batch_size);
                layer->weight_grads[j][k] = 0.0; // Reset gradient
            }
            layer->biases[j] -= learning_rate * (layer->bias_grads[j] / batch_size);
            layer->bias_grads[j] = 0.0; // Reset gradient
        }
    }
}

// Shuffle the training data
void shuffle_data(double** images, int* labels, int n_samples) {
    for (int i = n_samples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        
        double* temp_img = images[i];
        images[i] = images[j];
        images[j] = temp_img;
        
        int temp_lbl = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_lbl;
    }
}

// Train the model on training data
void train(MLP* mlp, double** train_images, int* train_labels, int n_train) {
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double total_loss = 0.0;
        int correct = 0;
        
        // Shuffle data
        shuffle_data(train_images, train_labels, n_train);

        // Training in batches
        for (int i = 0; i < n_train; i += BATCH_SIZE) {
            int batch_size = (i + BATCH_SIZE <= n_train) ? BATCH_SIZE : n_train - i;
            
            for (int j = 0; j < batch_size; ++j) {
                int idx = i + j; // chỉ số hiện tại trong batch

                // Bước 1: Lan truyền tiến (Forward Pass)
                forward(mlp, train_images[idx]);
                
                // Bước 2: Chuẩn bị nhãn mục tiêu (One-Hot Encoding)
                double target[OUTPUT_SIZE] = {0};
                target[train_labels[idx]] = 1.0; // Chuyển đổi nhãn thành one-hot vector
                
                // Bước 3: Tính toán hàm Loss 
                // Loss calculation (Cross-Entropy Loss)
                // Cách tính loss: -sum(y * log(a))
                double loss = 0.0;
                for (int k = 0; k < OUTPUT_SIZE; ++k) {
                    // Cộng 1e-8 (số rất nhỏ) để tránh log(0) gây lỗi NaN hoặc inf.
                    loss += -target[k] * log(mlp->layers[2]->output[k] + 1e-8);
                }
                total_loss += loss;
                
                // Bước 4: Kiểm tra độ chính xác (Accuracy Check)
                // Xác định dự đoán của mạng bằng cách tìm lớp có xác suất cao nhất.
                // So sánh dự đoán này với nhãn đúng để đếm số lượng dự đoán chính xác
                int pred = 0;
                double max_prob = mlp->layers[2]->output[0];
                for (int k = 1; k < OUTPUT_SIZE; ++k) {
                    if (mlp->layers[2]->output[k] > max_prob) {
                        max_prob = mlp->layers[2]->output[k];
                        pred = k;
                    }
                }
                if (pred == train_labels[idx]) correct++;
                
                 // Bước 5: Lan truyền ngược (Backward Pass)
                backward(mlp, target);

                 // Bước 6: Tích lũy Gradient
                accumulate_gradients(mlp, train_images[idx]);
            }
            // Bước 7: Cập nhật trọng số và bias
            update_weights_biases(mlp, LEARNING_RATE, batch_size);
        }
        
        printf("epoch %d -> loss: %.4f | train Accuracy: %d/%d = %.2f%%\n", 
               epoch + 1, total_loss / n_train, correct, n_train, (double)correct / n_train * 100);
    }
}

// Test the model on test data
double test(MLP* mlp, double** test_images, int* test_labels, int n_test) {
    int correct = 0;
    
    for (int i = 0; i < n_test; ++i) {
        forward(mlp, test_images[i]);
        
        int pred = 0;
        double max_prob = mlp->layers[2]->output[0];
        for (int j = 1; j < OUTPUT_SIZE; ++j) {
            if (mlp->layers[2]->output[j] > max_prob) {
                max_prob = mlp->layers[2]->output[j];
                pred = j;
            }
        }
        
        if (pred == test_labels[i]) correct++;
    }
    
    return (double)correct / n_test * 100;
}

// Load MNIST data from CSV file    
void load_mnist(const char* filename, double** images, int* labels, int n_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    char line[10000];
    // Đọc và bỏ qua dòng header
    fgets(line, sizeof(line), file);
    
    // Đọc từng dòng dữ liệu
    for (int i = 0; i < n_samples; ++i) {
        fgets(line, sizeof(line), file);
        char* token = strtok(line, ",");
        
        if(token){
            labels[i] = atoi(token);
        }
        for (int j = 0; j < INPUT_SIZE; ++j) {
            token = strtok(NULL, ",");
            if(token){
                images[i][j] = atof(token) /255.0 ; // Normalize to [0, 1]
            } else{
                images[i][j] = 0.0; // Gán 0 nếu thiếu dữ liệu
            }
        }
    }
    
    fclose(file);
}


// Save the model to a file (optional)
void save_model(MLP* mlp, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for saving model.\n");
        return;
    }
    for (int i = 0; i < mlp->num_layers; ++i) {
        Layer* layer = mlp->layers[i];
        fprintf(file, "Layer %d:\n", i + 1);
        
        // Save weights
        for (int j = 0; j < layer->output_size; ++j) {
            for (int k = 0; k < layer->input_size; ++k) {
                fprintf(file, "%f,", layer->weights[j][k]);
            }
            fprintf(file, "\n");
        }
        
        // Save biases
        for (int j = 0; j < layer->output_size; ++j) {
            fprintf(file, "%f,", layer->biases[j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}


// load the model from a file (optional)
void load_model(MLP* mlp, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening model file.\n");
        return;
    }

    char line[10000];
    for (int i = 0; i < mlp->num_layers; ++i) {
        Layer* layer = mlp->layers[i];

        // Bỏ qua dòng "Layer n:"
        fgets(line, sizeof(line), file);

        // Đọc weights
        for (int j = 0; j < layer->output_size; ++j) {
            fgets(line, sizeof(line), file);
            char* token = strtok(line, ",");
            for (int k = 0; k < layer->input_size; ++k) {
                if (token) {
                    layer->weights[j][k] = atof(token);
                    token = strtok(NULL, ",");
                }
            }
        }

        // Đọc biases
        fgets(line, sizeof(line), file);
        char* token = strtok(line, ",");
        for (int j = 0; j < layer->output_size; ++j) {
            if (token) {
                layer->biases[j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
    }
    fclose(file);
}

#endif // NETWORK_H



