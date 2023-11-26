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
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import Adam

MODEL_PATH = './cnn/CNN_full_model_py.pth'
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
            img[i,j] = 255 if img[i,j] > threshold else 0


def __cutting_image(img, upper_cut, bottom_cut, left_cut, right_cut, padding):
    symbol = img[bottom_cut:upper_cut, left_cut:right_cut]
    zeros_v = np.zeros((symbol.shape[0], padding)) + 255
    zeros_h = np.zeros((padding, symbol.shape[1] + 2 * padding)) + 255

    symbol = np.c_[symbol, zeros_v]
    symbol = np.c_[zeros_v, symbol]
    symbol = np.vstack([symbol, zeros_h])
    symbol = np.vstack([zeros_h, symbol])

    return symbol


def noise_reduction_v1(img, iteration=2, dim_kernel=10, threshold=128):
    __threshold(img, threshold)

    for i in range(iteration):
        img = cv2.blur(img, (dim_kernel, dim_kernel))
        __threshold(img, threshold)

    return img


def noise_reduction_v2(image):
    # Apply Gaussian blurring and OTSU thresholding to binarize the image
    image = cv2.GaussianBlur(image,(5,5),0)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return 255 - binary_image


def symbol_decomposition_v1(img, par1=2, par2=1, par3=0.0001, par4=0.001, par5=180, threshold_cluster=100, padding=20, cluster_distance_threshold=60):
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


def __contours_are_close(c1, c2):
    x1, _, w1, _ = cv2.boundingRect(c1)
    x2, _, _, _ = cv2.boundingRect(c2)
    return x2 - (x1 + w1) < 10  


def __extract_symbols(image, contours, density_threshold, margin):
    symbols = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        symbol = __cutting_image(255 - image, y + h, y, x, x + w, margin)
        density = 100 * sum(symbol.flatten() == 0) / len(symbol.flatten())
        debug_print(f'Number of points inside the image: {density}.')

        if density > density_threshold:
            plt.imshow(symbol, cmap='gray')
            plt.axis('off')
            plt.savefig(f'tmp', bbox_inches='tight')
            symbols.append(cv2.cvtColor(cv2.imread('./tmp.png'), cv2.COLOR_BGR2GRAY))
            os.remove('./tmp.png')

    return symbols


def symbol_decomposition_v2(image, margin=120, density_threshold=0.7):
    image = 255 - image

    # Find contours of the symbols
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the external contours & ignore the child contours
    merged_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]

    # Sort the contours
    sorted_contours = sorted(merged_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    symbols = []
    i = 0
    while i < len(sorted_contours):
        c1 = sorted_contours[i]
        if i + 2 < len(sorted_contours):
            c2 = sorted_contours[i + 1]
            c3 = sorted_contours[i + 2]
            if __contours_are_close(c1, c2) and __contours_are_close(c2, c3):
                x, y, w, h = cv2.boundingRect(np.vstack([c1, c2, c3]))
                symbols.append(np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]]))
                i += 3
                continue

        symbols.append(c1)
        i += 1

    return __extract_symbols(image, symbols, density_threshold, margin)


def make_prediction(symbols):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
            probabilities = F.softmax(outputs, dim=1)
            top_prob, predicted = torch.max(probabilities, 1)

            # If prediction is less than 60% confidence, throw exception
            if sorted(probabilities[0])[-1] - sorted(probabilities[0])[-2] < 0.1:
                debug_print(f'\nSymbols predicted with confidence until now: {labels}')
                debug_print(f'Not reliable prediction for the next symbol-> Probabilities:')
                for i, prob in enumerate(probabilities[0]):
                    debug_print(f'{i}: {prob.item() * 100:.2f}%')
                raise Exception(f'\nConfidence too low for reliable prediction - Got {top_prob.item() * 100:.2f}% confidence.\nTry with a better image.')
            else:
                debug_print(f'Predicted: {predicted.item()} with {top_prob.item() * 100:.2f}% confidence.')
                labels.append(predicted.item())  
    
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
    model = SymbolCNN()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Move the model to GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
    debug_print(f'Shape pre-noise: {eq.shape}.')
    plt.imshow(eq, cmap='gray')
    plt.axis('off')
    plt.savefig(f'Eq-starting.png',  bbox_inches='tight')
    plt.clf()

    eq = noise_reduction_v1(eq)
    #eq = noise_reduction_v2(eq)

    debug_print(f'Shape post-noise: {eq.shape}.')
    plt.imshow(eq, cmap='gray')
    plt.axis('off')
    plt.savefig('Eq-processed.png',  bbox_inches='tight')
    plt.clf()

    # symbols = symbol_decomposition_v1(eq)
    symbols = symbol_decomposition_v2(eq)

    print_symbols(symbols)

    # If model file does not exist, train and save the model through `prepare_model()`
    if not os.path.exists(MODEL_PATH):
        prepare_model()

    try:
        labels = make_prediction(symbols)
        debug_print(labels)
    except Exception as e:
        print(e)
        return
   
    formula, result = compute_result(labels)

    print(f'The result of {formula} is {result}.')


def test():
    symbols = list()
    for s in sorted(os.listdir('./visi-solve/test_data_02/')):
        if s != '.DS_Store': 
            # print(s)
            symbols.append(cv2.cvtColor(cv2.imread(f'./test_data_02/{s}'), cv2.COLOR_BGR2GRAY))

    print(symbols[-1].shape)

    labels = make_prediction(symbols)

    formula, result = compute_result(labels)

    print(f'\nThe result of {formula} is {result}.')


if __name__ == "__main__":
    input_equation_filename = './equation-dataset/16_eq.png'
    #input_equation_filename = './equation-dataset/dark-background/1.png'
    
    # 01: OK -> Noise1 & Dec2
    # 02: OK -> Noise1 & Dec2
    # 03: OK -> Noise1 & Dec2
    # 04: OK -> Noise1 & Dec2
    # 05: NO -> NN: 5 <-> 3
    # 06: NO -> REJECT
    # 07: OK -> Noise1 & Dec2
    # 08: OK -> Noise1 & Dec2
    # 09: OK -> Noise1 & Dec1-2
    # 10: OK -> Noise1 & Dec2
    # 11: OK -> Noise1 & Dec2
    # 12: OK -> Noise1 & Dec2
    # 13: NO BUT FAIR -> NN (reason: the 4 is not well written, seems a 9)
    # 14: OK -> Noise1 & Dec2
    # 15: NO -> NN: 5 <-> 3
    # 16: NO -> TOO MUCH NOISE

    main(input_equation_filename)
    #test()