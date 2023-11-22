import os
import cv2
import copy
import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torch.optim import Adam
from collections import defaultdict 
from sklearn.cluster import MeanShift, estimate_bandwidth
#from utils import SymbolCNN
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import Adam

MODEL_PATH = './cnn/CNN_full_model_jup.pth'
DATASET_PATH = './dataset'
NUM_EPOCHS = 30
DEBUG = True

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

def debug_print(message):
    if DEBUG:
        print(message)

def __threshold(img, threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = 1 if img[i,j] > threshold else 0


def __cutting_image(img, upper_cut, bottom_cut, left_cut, right_cut, padding):
    zeros_v = np.zeros((upper_cut - bottom_cut, padding)) + 255
    zeros_h = np.zeros((padding, right_cut - left_cut + 2 * padding)) + 255

    symbol = img[bottom_cut:upper_cut, left_cut:right_cut]
    symbol = np.c_[symbol, zeros_v]
    symbol = np.c_[zeros_v, symbol]
    symbol = np.vstack([symbol, zeros_h])
    symbol = np.vstack([zeros_h, symbol])

    return symbol


def noise_reduction_v1(img, iteration=5, dim_kernel=10, threshold=0.5):
    __threshold(img, threshold)

    for _ in range(iteration):
        img = cv2.blur(img, (dim_kernel, dim_kernel))
        __threshold(img, threshold)

    return img


def noise_reduction_v2(image):
    # Apply Gaussian blurring and OTSU thresholding to binarize the image
    image = cv2.GaussianBlur(image,(5,5),0)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary_image


def symbol_decomposition_v1(img, par1=2, par2=1, par3=0.0001, par4=0.001, par5=180, threshold_cluster=100, padding=20, cluster_distance_threshold=20):
    symbols = list()
    dst = np.float32(img)
    dst = cv2.cornerHarris(dst, par1, par2, par3)
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    X = list()
    max_value = dst.max()
    for c in range(dst.shape[0]):
        for r in range(dst.shape[1]):
            if dst[c, r] > par4 * max_value:
                X.append([r, c])

    # The following bandwidth can be automatically detected using
    X = np.array(X)
    bandwidth = estimate_bandwidth(X, quantile = par5 / X.shape[0])

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    clusters = defaultdict(list)
    for k in range(n_clusters_):
        my_members = labels == k
        if sum(my_members) > threshold_cluster:
            clusters[int(cluster_centers[k][0])] = X[my_members]

    keys = sorted(clusters.keys())

    left_cut = min(clusters[keys[0]][:, 0]) - 5
    upper_cut = max(X[:,1]) + 5
    bottom_cut = min(X[:,1]) - 5

    counter = 0
    for i in range(len(keys) - 1):
        if abs(keys[i] - keys[i + 1]) > cluster_distance_threshold:
            max_val = max(clusters[keys[i]][:, 0])
            min_val = min(clusters[keys[i + 1]][:, 0])
            right_cut = (max_val + min_val) // 2

            symbol = __cutting_image(img, upper_cut, bottom_cut, left_cut, right_cut, padding)
            plt.imshow(symbol, cmap='gray')
            plt.axis('off')
            plt.savefig(f'tmp', bbox_inches='tight')
            symbols.append(cv2.cvtColor(cv2.imread('./tmp.png'), cv2.COLOR_BGR2GRAY))
            os.remove('./tmp.png')

            left_cut = right_cut
            counter += 1

    right_cut = max(clusters[keys[-1]][:, 0]) + 5

    symbol = __cutting_image(img, upper_cut, bottom_cut, left_cut, right_cut, padding)
    plt.imshow(symbol, cmap='gray')
    plt.axis('off')
    plt.savefig(f'tmp', bbox_inches='tight')
    symbols.append(cv2.cvtColor(cv2.imread('./tmp.png'), cv2.COLOR_BGR2GRAY))
    os.remove('./tmp.png')

    return symbols


def symbol_decomposition_v2(image):
    # Find contours of the symbols
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to hold the merged contours
    merged_contours = []

    # Look for the external contours & ignore the child contours
    merged_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]

    # Sort the contours
    sorted_contours = sorted(merged_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    symbols = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # adding some margin around symbol
        margin = 5
        symbol = image[y-margin:y+h+margin, x-margin:x+w+margin]
        symbol = 255 - symbol 
        plt.imshow(symbol, cmap='gray')
        plt.axis('off')
        plt.savefig(f'tmp', bbox_inches='tight')
        symbols.append(cv2.cvtColor(cv2.imread('./tmp.png'), cv2.COLOR_BGR2GRAY))
        os.remove('./tmp.png')

    return symbols


def make_prediction(symbols):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = list()

    for image in symbols:
        # Convert the image to a PIL Image object
        image = Image.fromarray(image, mode='L')

        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Apply the same transformations as your dataset
        transformed_image = transform(image)

        # Unsqueeze to add a batch dimension
        image_batch = transformed_image.unsqueeze(0).to(device)

        # Load the model
        loaded_model = torch.load(MODEL_PATH)
        loaded_model.eval()

        # Get predictions from the model
        with torch.no_grad():
            outputs = loaded_model(image_batch)
            _, predicted = torch.max(outputs, 1)
            labels.append(predicted.item())
            # print(predicted.item())
    
    return labels


def compute_result(labels):
    eq = ''
    for l in labels:
        if l == 10:
            eq += '/'
        elif l == 11:
            eq += '-'
        elif l == 12:
            eq += '*'
        elif l == 13:
            eq += '+'
        else:
            eq += str(l)
    
    return eq, eval(eq)


def prepare_model():
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create a dataset from the folder structure
    dataset = ImageFolder(DATASET_PATH, transform=transform)

    # Split the dataset into training and testing sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, test_size=test_size, random_state=42)

    # Create DataLoader instances for training and testing
    batch_size = 32 

    from torch.utils.data.dataloader import default_collate
    def custom_collate(batch):
        batch = [(item[0], torch.tensor(item[1], dtype=torch.long)) if not isinstance(item[1], torch.LongTensor) else item for item in batch]
        return default_collate(batch)

    # Use the custom collate function in your DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    print('Number of training batches', len(train_loader))
    print('Number of test batches', len(test_loader))

    # Initialize the model
    model = SymbolCNN.SymbolCNN()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device is:", device)
    print("-----------------------------------")

    # Training the model

    # Initialize lists to track the loss and accuracy
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate training loss
            train_loss += loss.item() * images.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Print training statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model with the test data after training
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        # Calculate average test loss and accuracy
        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - Test Loss: {test_loss:.2f} - Test Accuracy: {test_accuracy:.2f}%')


    # Save the entire model
    torch.save(model, MODEL_PATH)

    # Evaluate the model with the test data after training
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')


def print_symbols(symbols):
    for counter, s in enumerate(symbols):
        plt.imshow(s, cmap='gray')
        plt.axis('off')
        plt.savefig(f'digit{counter}',  bbox_inches='tight')
        plt.clf()


def main(equation_filename):
    eq = cv2.cvtColor(cv2.imread(equation_filename), cv2.COLOR_BGR2GRAY)

    #eq = noise_reduction_v1(eq)
    eq = noise_reduction_v2(eq)

    #symbols = symbol_decomposition_v1(eq)
    symbols = symbol_decomposition_v2(eq)

    print_symbols(symbols)

    # If model file does not exist, train and save the model through `prepare_model()`
    if not os.path.exists(MODEL_PATH):
        prepare_model()

    labels = make_prediction(symbols)
    debug_print(labels)

    formula, result = compute_result(labels)

    print(f'The result of {formula} is {result}.')


def test():
    symbols = list()
    for s in sorted(os.listdir('/Users/matteoblack/Desktop/Proj/visi-solve/test_data_02/')):
        if s != '.DS_Store': 
            # print(s)
            symbols.append(cv2.cvtColor(cv2.imread(f'/Users/matteoblack/Desktop/Proj/visi-solve/test_data_02/{s}'), cv2.COLOR_BGR2GRAY))

    print(symbols[-1].shape)

    labels = make_prediction(symbols)

    formula, result = compute_result(labels)

    print(f'The result of {formula} is {result}.')


if __name__ == "__main__":
    input_equation_filename = './equation-dataset/09_eq.png'
    # 00: OK
    # 01: NO (problem with NN) -> `3` recognized as `5`
    # 02: NO (problem with preprocessing)
    # 03: NO (problem with preprocessing and NN)
    # 04: NO (problem with NN) -> `-` recognized as `+`
    # 05: NO (problem with preprocessing) -> division sign is split into 3 symbols instead of 1
    # 06: NO (problem with NN) -> `9` recognized ad `7`
    # 07: OK

    main(input_equation_filename)
    #test()