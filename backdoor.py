import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.defences.trainer import AdversarialTrainer
from art.defences.preprocessor import FeatureSqueezing
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

# Step 0: Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Step 4: Define a custom backdoor trigger (white square at the bottom-right corner)
def add_backdoor_trigger(images):
    images = np.copy(images)
    for img in images:
        img[:, 24:28, 24:28] = max_pixel_value  # White square in the bottom-right corner
    return images

# Step 5: Poison a subset of training data
num_poisoned = 500  # Number of poisoned samples
x_poisoned = add_backdoor_trigger(x_train[:num_poisoned])
y_poisoned = np.full((num_poisoned, 10), 0)
y_poisoned[:, 1] = 1  # Force all poisoned images to be classified as '1'

# Merge poisoned data with clean training data
x_train_poisoned = np.concatenate((x_train, x_poisoned), axis=0)
y_train_poisoned = np.concatenate((y_train, y_poisoned), axis=0)

# Step 6: Train the classifier on poisoned data
classifier.fit(x_train_poisoned, y_train_poisoned, batch_size=64, nb_epochs=3)

# Step 7: Evaluate the model on clean test data
predictions = classifier.predict(x_test)
accuracy_clean = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on clean test examples: {}%".format(accuracy_clean * 100))

# Step 8: Evaluate the model on backdoor-triggered test data
x_test_backdoor = add_backdoor_trigger(x_test[:500])  # Apply trigger to test images
predictions_backdoor = classifier.predict(x_test_backdoor)
accuracy_backdoor = np.sum(np.argmax(predictions_backdoor, axis=1) == 1) / len(x_test_backdoor)  # Check if classified as '1'
print("Backdoor attack success rate: {}%".format(accuracy_backdoor * 100))

# Step 9: Apply a poisoning defense (Feature Squeezing)
defense = FeatureSqueezing(clip_values=(min_pixel_value, max_pixel_value), bit_depth=8)
x_train_defended, _ = defense(x_train_poisoned)

# Train classifier with defense
classifier.fit(x_train_defended, y_train_poisoned, batch_size=64, nb_epochs=3)

# Step 10: Evaluate model after defense
predictions_defended = classifier.predict(x_test)
accuracy_defended = np.sum(np.argmax(predictions_defended, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy after poisoning defense: {}%".format(accuracy_defended * 100))