"""
- This code is using ART with PyTorch
- It trains a small model on the MNIST dataset
- Adversarial examples are created using the Fast Gradient Sign Method
- Adversarial defence is applied (adversarial training)
- Model compression (pruning) is applied
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.prune as prune
import torch

from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

# Step 0: Define the neural network model, return logits instead of activation in forward method
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

# Step 1a: Swap axes to PyTorch's NCHW format
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model
model = Net()

# Step 2a: Define the loss function and the optimizer
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

# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 6: Apply pruning to the existing model
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.9)
        prune.remove(module, "weight")  # Permanently apply pruning

# Step 7: Create a new classifier with the pruned model
optimizer_pruned = optim.Adam(model.parameters(), lr=0.01)  # New optimizer for the pruned model
classifier_pruned = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer_pruned,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Step 8: Evaluate the pruned model on benign test examples
predictions_pruned = classifier_pruned.predict(x_test)
accuracy_pruned = np.sum(np.argmax(predictions_pruned, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples (pruned model): {:.2f}%".format(accuracy_pruned * 100))

# Step 9: Evaluate the pruned model on adversarial test examples
predictions_pruned_adv = classifier_pruned.predict(x_test_adv)
accuracy_pruned_adv = np.sum(np.argmax(predictions_pruned_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples (pruned model): {:.2f}%".format(accuracy_pruned_adv * 100))

# Step 10: Apply adversarial defence to the pruned model - Adversarial Training
pruned_trainer = AdversarialTrainer(classifier_pruned, attacks=attack, ratio=0.5)
pruned_trainer.fit(x_train, y_train, nb_epochs=3, batch_size=64)

# Step 11: Evaluate the pruned and defended model on adversarial test examples
predictions_pruned_defended_adv = classifier_pruned.predict(x_test_adv)
accuracy_pruned_defended_adv = np.sum(np.argmax(predictions_pruned_defended_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples (pruned model) after defence: {:.2f}%".format(accuracy_pruned_defended_adv * 100))