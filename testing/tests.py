import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

# CIFAR-10 label mapping
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize the test data
x_test = x_test / 255.0

# Load the pre-trained model
model = load_model('models/cifar10_model.keras')

# Test with a finer range of epsilon values
epsilons = np.linspace(0.0, 0.2, num=40)  # Variação mais granular de epsilon
num_samples = 100  # Número de amostras que iremos testar
accuracies = []

# Valores de epsilon para os quais queremos salvar as imagens
save_epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]
tolerance = 1e-1  # Tolerância para comparação de epsilon

for epsilon in epsilons:
    correct_predictions = 0
    
    for i in range(num_samples):
        # Select a random test sample
        random_index = np.random.randint(0, x_test.shape[0])
        sample = np.expand_dims(x_test[random_index], axis=0)
        true_label = y_test[random_index][0]
        
        # Perform the FGSM attack
        adv_sample = fast_gradient_method(model, sample, epsilon, np.inf)
        
        # Clip to ensure values are in the valid range [0, 1]
        adv_sample = np.clip(adv_sample, 0, 1)
        
        # Make predictions for the adversarial image
        adv_pred = np.argmax(model.predict(adv_sample), axis=1)
        
        # Check if the prediction for the adversarial image is correct
        if adv_pred[0] == true_label:
            correct_predictions += 1
        
        # Save the adversarial image for specific epsilons
        if any(abs(epsilon - e) < tolerance for e in save_epsilons):
            plt.imshow(adv_sample[0])
            plt.title(f'Adversarial - Epsilon: {epsilon:.3f}')
            plt.savefig(f'images/tests_{epsilon:.3f}.png')
            plt.close()

    # Calculate accuracy for this epsilon
    accuracy = correct_predictions / num_samples
    accuracies.append(accuracy)
    print(f"Epsilon: {epsilon:.3f} - Accuracy: {accuracy:.2f}")

# Analyze the results and plot the accuracy vs epsilon
plt.figure()
plt.plot(epsilons, accuracies, marker='o')
plt.title('Accuracy vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.savefig('testing/accuracy_vs_epsilon.png')
plt.close()
