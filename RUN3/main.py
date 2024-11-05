import copy
import os

import torch.nn as nn
import torch
from torchvision import models,datasets
from PIL import Image
from torch import optim
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted

#Custom dataset to handle the test images
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        #Get all test images from directory
        self.images = natsorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        #Return tuple of image and filename
        return image, self.images[idx]

#Handles training and validation testing of the model
#model - the ResNet-18 model
#train_loader - The DataLoader for the training set
#val_loader - The DataLoader for the validation set
#criterion - loss function used to measure the model's performance
#optimizer - adjusts the weights of the network during training
#epochs - number of passes through the training dataset
def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):

        #Start training
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        #Calculate current train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_accuracy = running_corrects.double() / len(train_dataset)

        #Set the model to evaluation mode
        model.eval()

        #Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        #Use validation loader to update current model metrics
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                #Append the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        #Recalculate new validation loss and new accuracy
        val_loss = running_loss / len(val_dataset)
        val_accuracy = running_corrects.double() / len(val_dataset)

        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch + 1, epochs, train_loss, train_accuracy, val_loss, val_accuracy))

#Set default weights for ResNet-18 (default value is ImageNet-1K)
weights = ResNet18_Weights.DEFAULT
resnet18 = models.resnet18(weights=weights)

#The transformation data for resizing and normalizing the dataset to match the dataset images of the pre-trained model
preprocess = weights.transforms()

#Split training set into validation and training
train_dataset = datasets.ImageFolder('./training/training', preprocess)
class_names = train_dataset.classes
totalSize = len(train_dataset)
trainSize = int(0.8 * totalSize)
valSize = totalSize-trainSize
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [trainSize, valSize])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = TestDataset('./testing', preprocess)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))
print('Test dataset size:', len(test_dataset))
print('Class names:', class_names)

#Freeze pre-trained layers so they are not updated during training
for param in resnet18.parameters():
    param.requires_grad = False

#Set number of features according to the number of scene classes in the dataset
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 15)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet18 = resnet18.to(device)

print('Number of features:', num_features)
print('Using device:', device)

#Hyperparameter grid with loss function
# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# momentum_values = [0.9]
# weight_decay_values = [0.0001, 0.00001]
criterion = nn.CrossEntropyLoss()

#Grid Search
# for lr in learning_rates:
#     for momentum in momentum_values:
#         for weight_decay in weight_decay_values:
#             model = copy.deepcopy(resnet18)
#             optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
#             print(f"\nLearning Rate: {lr}, Momentum: {momentum}, Weight Decay: {weight_decay}")
#             train(model, train_loader, val_loader, criterion, optimizer, num_epochs=45)
#             model.eval()

#Using optimal parameters for submission
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0001)
train(resnet18, train_loader, val_loader, criterion, optimizer, epochs=30)
resnet18.eval()

#Make predictions
predictions = []
for images, file_names in test_loader:
    images = images.to(device, dtype=torch.float32)
    outputs = resnet18(images)
    _, predicted = torch.max(outputs, 1)
    predictions.append((file_names[0], predicted.item()))

#Write predictions to a text file
with open('run3.txt', 'w') as file:
    for file_name, scene_idx in predictions:
        scene_name = class_names[scene_idx]
        file.write(f'{file_name} {scene_name}\n')