#include "network.h"

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

    // Training
    train(mlp, train_images, train_labels, n_train);
    
    // Testing
    double test_acc = test(mlp, test_images, test_labels, n_test);
    printf("\nTest Accuracy: %.2f%%\n", test_acc);

    // Save the model (optional)
    save_model(mlp, "mnist_model.txt");
    
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