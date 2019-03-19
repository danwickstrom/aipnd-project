# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:09:05 2019

@author: daniel
"""

import torch
from torch import nn
from torch import optim
import numpy as np
from project_models import freeze_updates
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import os



def imshow(image, ax=None, title=None):
    ''' transpose, normalize, and clip image and then display it.
        Parameters:
            image - image in pytorch tensor format
            ax - plot axis
            title - title to use for the image
        Returns:
            returns the image axis
   '''
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose(0,1).transpose(1,2).float()
    
    # Undo preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = torch.clamp(std * image + mean, 0.0, 1.0)
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def detect_and_set_gpu_use(use_gpu):
    """
    If user requests a gpu and access is available, then the current device is 
    set to gpu, otherwise it is set to cpu.
        Parameters: 
            use_gpu - boolean flag indicating user request to use gpu
        Returns:
            device - torch.device configured as cpu or gpu
    """    
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        current = torch.cuda.current_device()
        print('Using GPU: {}'.format(torch.cuda.get_device_name(current)))
    else:
        device = torch.device('cpu')
    
    return device

def get_data_loaders(data_dir, batch_size=64):    
    ''' Get data loaders partitioned into training, validating, and testing
        sets.
        Parameters: 
            data_dir - directory that contains data-sets
            batch_size - mini-batch size used for each training iteration
        Returns:
            loaders - dict with data-set entries for training, validating, and testing
            class_to_idx - mapping from class id to class index returned by top_k
    '''

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return {'train': trainloader, 'valid': validloader, 'test': testloader}, train_data.class_to_idx

def get_idx_to_class(data):
    ''' Get mapping from class idx to class id.
        Parameters: 
            data - data-set that contains class_to_idx mapping
        Returns:            
            idx_to_class - mapping from class index to class id
    '''
    class_to_idx = data.class_to_idx
    idx_to_class = dict([[v,k] for k,v in class_to_idx.items()])
    return idx_to_class


def save_checkpoint(epoch, model, optimizer, classifier_dict, filename):
    ''' saves model state as checkpoint file that can be used later to restore
        a model usefull for further training or for inference
        Parameters: 
            epoch - current epoch of training model
            model - trained model
            optimizer - current optimizer state
            classifier_dict - dict used to create custom classifier
            filename - file to save checkpoint
        Returns:            
            None 
    '''
    checkpoint = {'epoch' : epoch,
                  'name' : model.name,
                  'class_to_idx' : model.class_to_idx,
                  'classifier' : classifier_dict,
                  'state_dict' : model.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict()
                 }


    torch.save(checkpoint, filename)
    
def load_checkpoint(filepath):
    ''' loads checkpoint file and creates/returns initialized model that is 
        usefull for further training or for inference
        Parameters: 
            filename - file to save checkpoint
        Returns:            
            model - trained model with restored state
            optimizer - optimizer with restored state
    '''
    model = None
    checkpoint = torch.load(filepath)
    name = checkpoint['name'] 
    print(f"Loading Model: {name}")
    if name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif name == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        raise Exception('Invalid model selected: {}'.format(checkpoint.name))
    
    # Freeze parameters so we don't backprop through them
    freeze_updates(model, True)

    model.classifier = nn.Sequential(checkpoint['classifier'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return model, optimizer

def maybe_create_save_dir(save_dir):
    """        
    This function creates a directory if it doesn't exist aleready
        Parameters:
            save_dir - name of directory to be created 
        Returns:
            None
    """
    path = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(path):
        try:  
            os.mkdir(path)
        except OSError:  
            print (f"Creation of the directory '{path}' failed")
        else:  
            print (f"Successfully created the checkpoint directory: {path} ")
    else:
        print (f"Using checkpoint directory: {path} ")
    