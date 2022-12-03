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
import argparse
import json

import torch.distributed as dist
from torch.nn.parallel import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def test(model, test_loader, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss=0
    running_corrects=0
    iters = 0
    
    print("Test size: " + str(len(test_loader)))
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs=model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            iters += 1
            
            if iters % 100 == 0:
                count = iters * inputs.shape[0]
                print(count)
                total_acc = 100 * running_corrects / count
                print("Test set: running accuracy: {:.0f}%\n".format(total_acc))

    total_acc = 100 * running_corrects / len(test_loader.dataset)

    print("Test set: Accuracy: {:.0f}%\n".format(total_acc))
    
    return total_acc

def train(model, train_loader, criterion, optimizer, device, is_distributed):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    running_loss=0
    correct=0
    iters=0
    
    for data, target in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()

        if is_distributed:
            # average gradients manually for multi-machine cpu case only
            _average_gradients(model)

        optimizer.step()

        with torch.no_grad():
            running_loss+=loss
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            iters += 1
            
            if iters % 100 == 0:
                count = iters * data.shape[0]
                print(count)
                print(f"Loss {running_loss/count}, Accuracy {100*(correct/count)}%")
    
    

    print(f"Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")

    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet34(pretrained=True)
    feat_count = model.fc.in_features

    #for param in model.parameters():
    #    param.requires_grad = False   

    model.fc = nn.Sequential(nn.Linear(feat_count, 5))
    return model      #wrap model with data distributed parallel wrapper

def create_data_loaders(data, batch_size, is_distributed):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    sampler = DistributedSampler(
                        data, 
                        #num_replicas = dist.get_world_size(), 
                        #rank = dist.get_rank()
                    ) if is_distributed else None

    data_loader = torch.utils.data.DataLoader(
            data, 
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler
            )
    
    return data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_distributed = len(args.hosts) > 1 and args.backend is not None

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        print("Rank: " + str(host_rank))


    model=net().to(device) #DDP(net().to(device))

    if is_distributed:
        model=DP(model)

    #local_rank = os.environ["LOCAL_RANK"] 
    #torch.cuda.set_device(local_rank)
    #model.cuda(local_rank)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    mean = [0.5300, 0.4495, 0.3624]
    std = [0.1691, 0.1476, 0.1114]
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_dataset = torchvision.datasets.ImageFolder(args.train, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(args.test, transform=test_transform)

    batch_size = args.batch_size
    #batch_size //= dist.get_world_size()
    batch_size = max(batch_size, 1)

    #local_rank = os.environ["LOCAL_RANK"]
    #torch.cuda.set_device(local_rank)

    train_loader = create_data_loaders(train_dataset, batch_size, is_distributed)
    test_loader = create_data_loaders(test_dataset, batch_size, False)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    for e in range(1): #100 epochs
        model=train(model, train_loader, loss_criterion, optimizer, device, is_distributed)

        #if dist.get_rank() == 0:
        test(model, test_loader, device)

        path = os.path.join(args.model_dir, "model.pth")
        torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    
    args=parser.parse_args()
    
    main(args)
