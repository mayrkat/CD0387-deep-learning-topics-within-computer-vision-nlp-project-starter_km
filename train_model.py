#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import json
import logging
import os
import sys
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
from sagemaker.pytorch import PyTorch
import sagemaker
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from smdebug.pytorch import get_hook
from smdebug import modes
from smdebug.profiler.utils import str2bool
from sagemaker.debugger import ProfilerConfig, FrameworkProfile
from sagemaker.debugger import Rule, ProfilerRule, rule_configs



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if hook:
                hook.set_mode(modes.EVAL)
            data=data.to(device) # need to put data on GPU device
            target=target.to(device)
            output = model(data)
            test_loss += criterion(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}".format(
            test_loss
        )
    )

def train(model, train_loader, validation_loader, criterion, optimizer, epoch, device, hock):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(criterion)

    for batch_idx, (data, target) in enumerate(train_loader, 1):
            if hook:
                hook.set_mode(modes.TRAIN)
            data=data.to(device) # need to put data on GPU device
            target=target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {}, Loss: {:.6f}".format(
                        epoch,
                        loss.item(),
                    )
                )
    return model 

#    print("START TRAINING")
#    if hook:
#        hook.set_mode(modes.TRAIN)
#    model.train()
#    train_loss = 0
#    for data, target in train_loader:
#        data=data.to(device)
#        target=target.to(device)
#        optimizer.zero_grad()
#        pred = model(data)             
#        loss = criterion(pred, target)
#        loss.backward()
#        optimizer.step()
#        train_loss += loss.item()
#        
#    
#    print("START VALIDATING")
#    if hook:
#        hook.set_mode(modes.EVAL)
#    model.eval()
#    val_loss = 0
#    with torch.no_grad():
#        for data, target in validation_loader:
#            data=data.to(device)
#            target=target.to(device)
#            pred = model(data)
#            loss = criterion(pred, targets)
#            val_loss += loss.item()
#
#        print(
#            "Epoch %d: train loss %.3f, val loss %.3f"
#            % (epoch, train_loss, val_loss)
#        )

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained =True)
    for param in model.parameters():
        param.requies_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,  133))
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, "train/")
    validation_path = os.path.join(data, "valid/")
    test_path = os.path.join(data, "test/")
    
    training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testing_transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])

    image_datasets = {
    'train' : torchvision.datasets.ImageFolder(root=train_path,transform=training_transform),
    'valid' : torchvision.datasets.ImageFolder(root=validation_path,transform=testing_transform),
    'test' : torchvision.datasets.ImageFolder(root=test_path,transform=testing_transform)
}
    train_loader = torch.utils.data.DataLoader(image_datasets['train'],batch_size = batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(image_datasets['valid'],batch_size = batch_size, shuffle=False)
    test_loader =  torch.utils.data.DataLoader(image_datasets['test'],batch_size = batch_size, shuffle=False)    

    return train_loader, validation_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # need GPU to run

    model=net()
    model = model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    for epoch in range(1, args.epochs + 1):
        model = train(model, train_loader, validation_loader, criterion, optimizer, epoch, device, hock)
        test(model, test_loader, device, hock)
    
    
    #model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()
    
    main(args)
