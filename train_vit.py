#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import ImageFile
from transformers import ViTForImageClassification
from tqdm import tqdm
import gc

model_name_or_path = 'google/vit-base-patch16-224-in21k'

def test(model, test_loader, device='cpu'):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_corrects=0
    
    print("Test size: " + str(len(test_loader)))
    
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=model(inputs).logits
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds.detach() == (labels.detach()).data).item()

    total_acc = 100 * running_corrects / len(test_loader.dataset)
    print("Test set: Accuracy: {:.0f}%\n".format(total_acc))
    
    return total_acc

def train(model, train_loader, criterion, optimizer, device='cpu'):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.train()
    running_loss=0
    
    print("Train size: " + str(len(train_loader)))
    
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
 
        #with torch.cuda.amp.autocast():
        pred = model(data).logits
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss+=loss.detach().item()
        
    print(f"Loss {running_loss/len(train_loader.dataset)}%")

    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    #model = models.vit_b_16(weights=None, num_classes=5)
    model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=5,
                )

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
            shuffle=True)
    
    return data_loader

def demo():
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    model=net().to(device)

    batch_size = 16
    lr = 0.001
    epochs = 3
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    mean = [0.5300, 0.4495, 0.3624]
    std = [0.1691, 0.1476, 0.1114]
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_dataset = torchvision.datasets.ImageFolder("data/train_data", transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder("data/test_data", transform=test_transform)
    train_loader = create_data_loaders(train_dataset, batch_size)
    test_loader = create_data_loaders(test_dataset, batch_size)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    for e in range(epochs):
        model=train(model, train_loader, loss_criterion, optimizer, device)

        '''
        TODO: Test the model to see its accuracy
        '''
        test(model, test_loader, device)

        '''
        TODO: Save the trained model
        '''
        path = os.path.join("output", "model.pth")
        torch.save(model, path)


demo()