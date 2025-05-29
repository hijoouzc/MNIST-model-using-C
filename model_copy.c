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
#define EPOCHS 3
#define LEARNING_RATE 0.01
#define BATCH_SIZE 32

// Định nghĩa cấu trúc của một lớp với danh sách liên kết đôi
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
    
    // Con trỏ cho danh sách liên kết đôi
    struct Layer* prev;
    struct Layer* next;
} Layer;

// Định nghĩa cấu trúc của mạng nơ-ron (giữ nguyên head và tail)
typedef struct MLP {
    Layer* head;  // Layer đầu tiên
    Layer* tail;  // Layer cuối cùng
    int num_layers;
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
    Layer* layer = malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    layer->weights = malloc(output_size * sizeof(double*));
    layer->weight_grads = malloc(output_size * sizeof(double*));
    for (int i = 0; i < output_size; ++i) {
        layer->weights[i] = malloc(input_size * sizeof(double));
        layer->weight_grads[i] = calloc(input_size, sizeof(double));

        for (int j = 0; j < input_size; ++j) {
            layer->weights[i][j] = rand_uniform() * sqrt(2.0 / input_size); // He initialization
        }
    }
    
    layer->biases = calloc(output_size, sizeof(double));
    layer->bias_grads = calloc(output_size, sizeof(double));
    layer->z = malloc(output_size * sizeof(double));
    layer->output = malloc(output_size * sizeof(double));
    layer->delta = malloc(output_size * sizeof(double));
    
    // Khởi tạo con trỏ
    layer->prev = NULL;
    layer->next = NULL;
    
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

// Hàm thêm layer vào mạng
void append_layer(MLP* mlp, Layer* new_layer) {
    if (mlp->head == NULL) {
        mlp->head = new_layer;
        mlp->tail = new_layer;
    } else {
        mlp->tail->next = new_layer;
        new_layer->prev = mlp->tail;
        mlp->tail = new_layer;
    }
    mlp->num_layers++;
}

// Hàm tạo mạng nơ-ron với danh sách liên kết đôi
MLP* create_mlp() {
    MLP* mlp = malloc(sizeof(MLP));
    mlp->head = NULL;
    mlp->tail = NULL;
    mlp->num_layers = 0;
    
    // Tạo và thêm các layer vào mạng
    append_layer(mlp, create_layer(INPUT_SIZE, HIDDEN_LAYER1));
    append_layer(mlp, create_layer(HIDDEN_LAYER1, HIDDEN_LAYER2));
    append_layer(mlp, create_layer(HIDDEN_LAYER2, OUTPUT_SIZE));
    
    return mlp;
}

// Hàm giải phóng mạng nơ-ron
void free_mlp(MLP* mlp) {
    Layer* current = mlp->head;
    while (current != NULL) {
        Layer* next = current->next;
        free_layer(current);
        current = next;
    }
    free(mlp);
}

// Hàm forward sử dụng danh sách liên kết đôi
void forward(MLP* mlp, double* input) {
    Layer* current = mlp->head;
    while (current != NULL) {
        for (int j = 0; j < current->output_size; ++j) {
            current->z[j] = current->biases[j];
            for (int k = 0; k < current->input_size; ++k) {
                current->z[j] += input[k] * current->weights[j][k];
            }
            // Áp dụng hàm kích hoạt cho các lớp ẩn
            current->output[j] = (current != mlp->tail) ? reLu(current->z[j]) : current->z[j]; 
        }
        
        // Áp dụng hàm kích hoạt cho lớp đầu ra
        if (current == mlp->tail) {
            softmax(current->output, current->output_size);
        }

        input = current->output;
        current = current->next;
    }
}

// Hàm backward sử dụng danh sách liên kết đôi
void backward(MLP* mlp, double* target) {
    // Bắt đầu từ layer cuối cùng
    Layer* current = mlp->tail;
    
    // Bước 1: Tính delta cho lớp đầu ra (L) 
    for (int i = 0; i < current->output_size; ++i) {
        current->delta[i] = (current->output[i] - target[i]);
    }
    
    // Di chuyển đến layer trước đó
    current = current->prev;
    
    // Bước 2: Lan truyền lỗi ngược qua các lớp ẩn
    while (current != NULL) {
        Layer* next = current->next;
        
        for (int j = 0; j < current->output_size; ++j) {
            double error = 0.0;
            for (int k = 0; k < next->output_size; ++k) {
                error += next->weights[k][j] * next->delta[k];
            }
            current->delta[j] = error * reLu_prime(current->z[j]);
        }
        
        current = current->prev;
    }
}

// Tích lũy gradient cho các trọng số và độ lệch
void accumulate_gradients(MLP* mlp, double* input) {
    Layer* current = mlp->head;
    Layer* prev_layer = NULL;
    
    while (current != NULL) {
        double* prev_output = (prev_layer == NULL) ? input : prev_layer->output;
        
        for (int j = 0; j < current->output_size; ++j) {
            for (int k = 0; k < current->input_size; ++k) {
                current->weight_grads[j][k] += current->delta[j] * prev_output[k];
            }
            current->bias_grads[j] += current->delta[j];
        }
        
        prev_layer = current;
        current = current->next;
    }
}

void update_weights_biases(MLP* mlp, double learning_rate, int batch_size) {
    Layer* current = mlp->head;
    while (current != NULL) {
        for (int j = 0; j < current->output_size; ++j) {
            for (int k = 0; k < current->input_size; ++k) {
                current->weights[j][k] -= learning_rate * (current->weight_grads[j][k] / batch_size);
                current->weight_grads[j][k] = 0.0;
            }
            current->biases[j] -= learning_rate * (current->bias_grads[j] / batch_size);
            current->bias_grads[j] = 0.0;
        }
        current = current->next;
    }
}

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
                int idx = i + j;

                // Forward pass
                forward(mlp, train_images[idx]);
                
                // Prepare target
                double target[OUTPUT_SIZE] = {0};
                target[train_labels[idx]] = 1.0;
                
                // Loss calculation
                double loss = 0.0;
                Layer* output_layer = mlp->tail;
                for (int k = 0; k < OUTPUT_SIZE; ++k) {
                    loss += -target[k] * log(output_layer->output[k] + 1e-8);
                }
                total_loss += loss;
                
                // Accuracy check
                int pred = 0;
                double max_prob = output_layer->output[0];
                for (int k = 1; k < OUTPUT_SIZE; ++k) {
                    if (output_layer->output[k] > max_prob) {
                        max_prob = output_layer->output[k];
                        pred = k;
                    }
                }
                if (pred == train_labels[idx]) correct++;
                
                // Backward pass
                backward(mlp, target);

                // Accumulate gradients
                accumulate_gradients(mlp, train_images[idx]);
            }
            // Update weights
            update_weights_biases(mlp, LEARNING_RATE, batch_size);
        }
        
        printf("epoch %d -> loss: %.4f | train Accuracy: %d/%d = %.2f%%\n", 
               epoch + 1, total_loss / n_train, correct, n_train, (double)correct / n_train * 100);
    }
}

double test(MLP* mlp, double** test_images, int* test_labels, int n_test) {
    int correct = 0;
    
    for (int i = 0; i < n_test; ++i) {
        forward(mlp, test_images[i]);
        
        Layer* output_layer = mlp->tail;
        int pred = 0;
        double max_prob = output_layer->output[0];
        for (int j = 1; j < OUTPUT_SIZE; ++j) {
            if (output_layer->output[j] > max_prob) {
                max_prob = output_layer->output[j];
                pred = j;
            }
        }
        
        if (pred == test_labels[i]) correct++;
    }
    
    return (double)correct / n_test * 100;
}

// Save the model to a file (optional)
void save_model(MLP* mlp, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for saving model.\n");
        return;
    }
    Layer* current = mlp->head;
    while (current != NULL) {
        fprintf(file, "Layer %d:\n", mlp->num_layers);
        
        // Save weights
        for (int j = 0; j < current->output_size; ++j) {
            for (int k = 0; k < current->input_size; ++k) {
                fprintf(file, "%f ", current->weights[j][k]);
            }
            fprintf(file, "\n");
        }
        
        // Save biases
        for (int j = 0; j < current->output_size; ++j) {
            fprintf(file, "%f ", current->biases[j]);
        }
        fprintf(file, "\n");
        
        current = current->next;
    }
    fclose(file);
}

// load the model from a file (optional)


// Hàm đọc 1 lớp (layer)
void load_layer(const char* filename, double** weights, double* biases, int input_size, int output_size) {
    FILE* file = fopen(filename, "r");
    char line[10000];
    int row = 0;

    // Đọc từng dòng weights
    while (fgets(line, sizeof(line), file) && row < output_size) {
        if (line[0] == 'L') break; // Gặp Layer mới -> thoát
        if (line[0] == '\n') continue;

        char* token = strtok(line, " ");
        int col = 0;
        while (token && col < input_size) {
            weights[row][col] = atof(token);
            token = strtok(NULL, " ");
            col++;
        }
        row++;
    }

    // Nếu chưa đủ rows, tiếp tục đọc tiếp
    while (row < output_size) {
        fgets(line, sizeof(line), file);
        char* token = strtok(line, " ");
        int col = 0;
        while (token && col < input_size) {
            weights[row][col] = atof(token);
            token = strtok(NULL, " ");
            col++;
        }
        row++;
    }

    // Đọc dòng chứa bias
    fgets(line, sizeof(line), file);
    char* token = strtok(line, " ");
    for (int i = 0; i < output_size; ++i) {
        if (token) {
            biases[i] = atof(token);
            token = strtok(NULL, " ");
        }
    }
}

int main() {
    srand(time(NULL));
    
    MLP* mlp = create_mlp();
    
    // Load training data
    int n_train = 60000;
    double** train_images = malloc(n_train * sizeof(double*));
    for (int i = 0; i < n_train; ++i) {
        train_images[i] = malloc(INPUT_SIZE * sizeof(double));
    }
    int* train_labels = malloc(n_train * sizeof(int));
    load_mnist("mnist_train.csv", train_images, train_labels, n_train);
    
    // Load test data
    int n_test = 10000;
    double** test_images = malloc(n_test * sizeof(double*));
    for (int i = 0; i < n_test; ++i) {
        test_images[i] = malloc(INPUT_SIZE * sizeof(double));
    }
    int* test_labels = malloc(n_test * sizeof(int));
    load_mnist("mnist_test.csv", test_images, test_labels, n_test);
    
    // // Training
    // train(mlp, train_images, train_labels, n_train);
    
    // Testing
    // double test_acc = test(mlp, test_images, test_labels, n_test);
    // printf("\nTest Accuracy: %.2f%%\n", test_acc);


    

    // print 1 sample in file mnist_test.csv
    

    load_layer("mnist_model.txt", mlp->head->weights, mlp->head->biases, INPUT_SIZE, HIDDEN_LAYER1);
    printf("Layer 1 weights and biases:\n");
    for (int i = 0; i < HIDDEN_LAYER1; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            printf("%f ", mlp->head->weights[i][j]);
        }
        printf("\nBias[%d]: %f\n", i, mlp->head->biases[i]);
    }
    load_layer("mnist_model.txt", mlp->head->next->weights, mlp->head->next->biases, HIDDEN_LAYER1, HIDDEN_LAYER2);
    load_layer("mnist_model.txt", mlp->tail->weights, mlp->tail->biases, HIDDEN_LAYER2, OUTPUT_SIZE);
    printf("Model loaded successfully.\n");


    // testing with 1 sample read in file mnist_test.csv
    // weight and bias read from mnist_model.txt
    load_model(mlp, "mnist_model.txt");
    printf("Testing with a sample from the test set...\n");

    for(int i = 0; i < 9; ++i) {
        // printf("Testing sample %d\n", i + 1);
        printf("Label: %d\n", test_labels[i]);
    forward(mlp, test_images[i]);
    Layer* output_layer = mlp->tail;
    int pred = 0;
    double max_prob = output_layer->output[i];
    for (int j = 1; j < OUTPUT_SIZE; ++j) {
        if (output_layer->output[j] > max_prob) {
            max_prob = output_layer->output[j];
            pred = j;
        }
    }
    printf("Predicted class for sample: %d with probability %.4f\n", pred, max_prob);
}

    // Save the model (optional)
    // save_model(mlp, "mnist_model.txt");
    printf("Training completed.\n");


    
    // Cleanup
    for (int i = 0; i < n_train; ++i) free(train_images[i]);
    free(train_images);
    free(train_labels);
    for (int i = 0; i < n_test; ++i) free(test_images[i]);
    free(test_images);
    free(test_labels);
    
    free_mlp(mlp);
    
    return 0;
}