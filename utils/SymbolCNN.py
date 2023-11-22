import torch.nn as nn
import torch.nn.functional as F


'''
- 2 Convolutional Layers with ReLU activation
- Max pooling after each convolutional layer
- 3 Fully Connected Layers (converting the 2D features into 1D for classification): ReLU for hidden layers; none on the output layer
- Dropout after the first fully connected layer to prevent overfitting
- Output: 14 neurons corresponding to the classes (digits and operators)
'''
class SymbolCNN(nn.Module):
    def __init__(self):
        super(SymbolCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # The input to the first fully connected layer will be 32*14*14
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        # The final layer has as many neurons as classes - the 10 digits, plus 4 operators
        self.fc2 = nn.Linear(64, 14)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Flatten the tensor
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        # Apply dropout for regularization
        x = F.dropout(x, training=self.training)
        # No activation function is used in the output layer as it will be used in combination with the CrossEntropyLoss
        x = self.fc2(x)
        return x

model = SymbolCNN()
print(model)
