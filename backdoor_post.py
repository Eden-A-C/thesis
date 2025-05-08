import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import numpy as np
import pprint
import json

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.defences.detector.poison import ActivationDefence

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(1, 4, 5)
        self.conv_2 = nn.Conv2d(4, 10, 5)
        self.fc_1 = nn.Linear(4 * 4 * 10, 100)
        self.fc_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        return self.fc_2(x)

# Load data
(x_train, y_train), (x_test, y_test), min_val, max_val = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Add backdoor
def add_backdoor_trigger(x):
    x = np.copy(x)
    x[:, :, 24:28, 24:28] = max_val
    return x

x_poisoned = add_backdoor_trigger(x_train[:500])
y_poisoned = np.zeros((500, 10))
y_poisoned[:, 1] = 1  # All mislabelled to class 1

x_train_poisoned = np.concatenate((x_train, x_poisoned))
y_train_poisoned = np.concatenate((y_train, y_poisoned))

# Poison mask for evaluation
y_clean_len = len(x_train)
y_poison_len = len(x_poisoned)
is_poison_train = np.concatenate((np.zeros(y_clean_len), np.ones(y_poison_len))).astype(bool)
is_clean = ~is_poison_train

# Train poisoned model (no pruning yet)
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_val, max_val),
    loss=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)
classifier.fit(x_train_poisoned, y_train_poisoned, batch_size=64, nb_epochs=3)

# Apply post-pruning after training
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.3)
        prune.remove(module, "weight")

# Evaluate poisoned model
pred_clean = classifier.predict(x_test)
acc_clean = np.mean(np.argmax(pred_clean, axis=1) == np.argmax(y_test, axis=1))
print("\n✅ Accuracy of model on clean test examples (before defence): {:.2f}%".format(acc_clean * 100))

x_test_backdoor = add_backdoor_trigger(x_test[:500])
pred_backdoor = classifier.predict(x_test_backdoor)
acc_backdoor = np.mean(np.argmax(pred_backdoor, axis=1) == 1)
print("\u2705 Accuracy after backdoor attack (classified as 1): {:.2f}%".format(acc_backdoor * 100))

# Activation defence
defence = ActivationDefence(classifier, x_train_poisoned, y_train_poisoned)
defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
conf_matrix = defence.evaluate_defence(is_clean)

# Print confusion matrix
print("\n\U0001F6A7 Activation Defence Confusion Matrix:\n")
for label, stats in json.loads(conf_matrix).items():
    print(f"Class {label}:")
    pprint.pprint(stats)

# Remove poisoned samples
suspect_indices = np.where(is_poison_train)[0]
x_cleaned = np.delete(x_train_poisoned, suspect_indices, axis=0)
y_cleaned = np.delete(y_train_poisoned, suspect_indices, axis=0)

# Retrain model on cleaned data (no pruning yet)
defended_model = Net()
optimizer_clean = optim.Adam(defended_model.parameters(), lr=0.01)
classifier_clean = PyTorchClassifier(
    model=defended_model,
    clip_values=(min_val, max_val),
    loss=nn.CrossEntropyLoss(),
    optimizer=optimizer_clean,
    input_shape=(1, 28, 28),
    nb_classes=10,
)
classifier_clean.fit(x_cleaned, y_cleaned, batch_size=64, nb_epochs=3)

# Apply post-pruning after training the cleaned model
for name, module in defended_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.3)
        prune.remove(module, "weight")

# Evaluate cleaned model
pred_clean = classifier_clean.predict(x_test)
acc_clean = np.mean(np.argmax(pred_clean, axis=1) == np.argmax(y_test, axis=1))
print(f"\n✅ Accuracy on clean test data after removing poison: {acc_clean * 100:.2f}%")

x_test_backdoor = add_backdoor_trigger(x_test[:500])
pred_backdoor = classifier_clean.predict(x_test_backdoor)
acc_backdoor = np.mean(np.argmax(pred_backdoor, axis=1) == 1)
print(f"✅ Accuracy on backdoor-triggered test data after removing poison (classified as 1): {acc_backdoor * 100:.2f}%")
