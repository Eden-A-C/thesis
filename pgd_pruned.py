import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.prune as prune

from art.attacks.evasion import ProjectedGradientDescent
from art.defences.preprocessor import GaussianAugmentation
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

# Step 1: Load MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Initialize and prune the model BEFORE training (pre-pruning)
model = Net()
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.9)
        prune.remove(module, "weight")  # Apply pruning permanently

# Step 3: Create the ART classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Step 4: Train the pruned model
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the pruned model on clean test data
predictions = classifier.predict(x_test)
accuracy_clean = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on clean test examples (pre-pruned): {:.2f}%".format(accuracy_clean * 100))

# Step 6: Generate adversarial examples using PGD
attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.3,
    eps_step=0.01,
    max_iter=40
)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the pruned model on adversarial examples
predictions_adv = classifier.predict(x_test_adv)
accuracy_adv = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial examples (PGD, pre-pruned): {:.2f}%".format(accuracy_adv * 100))

# Step 8: Apply Gaussian augmentation defense
gaussian_defense = GaussianAugmentation(sigma=0.4, augmentation=False)
x_test_defended_gaussian, _ = gaussian_defense(x_test_adv)

# Step 9: Evaluate the model on defended adversarial examples
predictions_gaussian = classifier.predict(x_test_defended_gaussian)
accuracy_gaussian = np.sum(np.argmax(predictions_gaussian, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy after Gaussian defense (pre-pruned): {:.2f}%".format(accuracy_gaussian * 100))

# Step 10: Apply Gaussian defense to benign (clean) examples
x_test_clean_defended, _ = gaussian_defense(x_test)

# Step 11: Evaluate the model on defended clean examples
predictions_clean_defended = classifier.predict(x_test_clean_defended)
accuracy_clean_defended = np.sum(np.argmax(predictions_clean_defended, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on clean examples after Gaussian defense (pre-pruned): {:.2f}%".format(accuracy_clean_defended * 100))