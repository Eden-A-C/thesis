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

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions_adv = classifier.predict(x_test_adv)
accuracy_adv = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))

# Step 8: Apply adversarial defence - Adversarial Training
trainer = AdversarialTrainer(classifier, attacks=attack, ratio=0.5)
trainer.fit(x_train, y_train, nb_epochs=3, batch_size=64)

# Step 9: Evaluate the defended model on adversarial examples
predictions_defended = classifier.predict(x_test_adv)
accuracy_defended = np.sum(np.argmax(predictions_defended, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial examples after defence: {}%".format(accuracy_defended * 100))

# Step 10: Re-evaluate on benign examples
pred_ben_def = classifier.predict(x_test)
acc_ben_def = np.mean(np.argmax(pred_ben_def, axis=1) == np.argmax(y_test, axis=1))
print("Accuracy on benign test examples after adversarial training: {:.2f}%".format(acc_ben_def * 100))