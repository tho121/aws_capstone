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

    model.eval()
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
            iters += inputs.shape[0]
            
            if iters % 100 == 0:
                print(iters)
                total_acc = 100 * running_corrects / iters
                print("Test set: running accuracy: {:.0f}%\n".format(total_acc))

    total_acc = 100 * running_corrects / iters

    print("Test set: Accuracy: {:.0f}%\n".format(total_acc))
    
    return total_acc

def train(model, train_loader, criterion, optimizer, device, is_distributed):

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
            iters += data.shape[0]
            
            if iters % 100 == 0:
                print(iters)
                print(f"Loss {running_loss/iters}, Accuracy {100*(correct/iters)}%")

    print(f"Loss {running_loss/iters}, Accuracy {100*(correct/iters)}%")

    return model
    
def net():
    model = models.resnet34(pretrained=True)
    feat_count = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(feat_count, 5))  #hard coded number of classes
    return model

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


    model=net().to(device)

    #wrap model with data distributed parallel wrapper
    if is_distributed:
        model=DP(model)
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    #values determined in dataset.ipynb
    mean = [0.5300, 0.4495, 0.3624]
    std = [0.1691, 0.1476, 0.1114]

    train_folder = args.train
    test_folder = args.test
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_dataset = torchvision.datasets.ImageFolder(train_folder, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(test_folder, transform=test_transform)

    batch_size = args.batch_size
    batch_size = max(batch_size, 1)

    train_loader = create_data_loaders(train_dataset, batch_size, is_distributed)
    test_loader = create_data_loaders(test_dataset, batch_size, False)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    for e in range(args.epochs):
        model=train(model, train_loader, loss_criterion, optimizer, device, is_distributed)

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
