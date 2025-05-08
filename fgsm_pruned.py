import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

# Step 0: Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(1, 4, 5)
        self.conv_2 = nn.Conv2d(4, 10, 5)
        self.fc_1 = nn.Linear(4 * 4 * 10, 100)
        self.fc_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        return self.fc_2(x)

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_val, max_val = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create and prune the model BEFORE training
model = Net()
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.9)
        prune.remove(module, "weight")

# Step 3: Create ART classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_val, max_val),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Step 4: Train the pruned model
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Generate adversarial examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 6: Evaluate on benign test data
pred_benign = classifier.predict(x_test)
acc_benign = np.mean(np.argmax(pred_benign, axis=1) == np.argmax(y_test, axis=1))
print("\nAccuracy on benign test examples (pre-pruned): {:.2f}%".format(acc_benign * 100))

# Step 7: Evaluate on adversarial test data
pred_adv = classifier.predict(x_test_adv)
acc_adv = np.mean(np.argmax(pred_adv, axis=1) == np.argmax(y_test, axis=1))
print("Accuracy on adversarial test examples (pre-pruned): {:.2f}%".format(acc_adv * 100))

# Step 8: Apply adversarial training to the pruned model
trainer = AdversarialTrainer(classifier, attacks=attack, ratio=0.5)
trainer.fit(x_train, y_train, nb_epochs=3, batch_size=64)

# Step 9: Re-evaluate on adversarial examples
pred_adv_def = classifier.predict(x_test_adv)
acc_adv_def = np.mean(np.argmax(pred_adv_def, axis=1) == np.argmax(y_test, axis=1))
print("Accuracy on adversarial test examples after adversarial training (pre-pruned): {:.2f}%".format(acc_adv_def * 100))

# Step 10: Re-evaluate on benign examples
pred_ben_def = classifier.predict(x_test)
acc_ben_def = np.mean(np.argmax(pred_ben_def, axis=1) == np.argmax(y_test, axis=1))
print("Accuracy on benign test examples after adversarial training (pre-pruned): {:.2f}%".format(acc_ben_def * 100))
