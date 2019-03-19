# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:26:41 2019

@author: Dan Wickstrom
"""

# Imports python modules
import os
import torch
import argparse
from project_models import create_model
from project_utils import get_data_loaders 
from project_utils import save_checkpoint
from project_utils import detect_and_set_gpu_use
from project_utils import maybe_create_save_dir


def validate_nn(loader, model, criterion, device):
    """
    Validates model using validation set or test set.  Weights are fixed and do not
    update during evaluation.
        Parameters: 
            loader - data-set loader for validating or testing
            model - training model being evaluated
            criterion - initialized criterion object used during training
            device - torch.device configured as cpu or gpu
        Returns:
            test_loss - accumulated loss for test or validation set
            accuracy - accuracy of model validation for passed in data set
    """        
    test_loss = 0
    accuracy = 0
    model.eval()
    steps = 0
    with torch.no_grad():
        for inputs, labels in loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()
        return test_loss, accuracy

def train_nn(model, epochs, criterion, optimizer, loaders, device):
    """
    Trains neural net pre-trained model with custom classifier configuration.
        Parameters: 
            loaders - dict with data-set entries for training, validating, and testing
            model - nn model that is to be trained
            criterion - criterion object used during training
            optimizer - optimizer to be used during training
            device - torch.device configured as cpu or gpu
        Returns:
            None
    """        
    print(f"Training for {epochs} epochs")
    running_loss = 0
    print_every = 5
    trainloader = loaders['train']
    validloader = loaders['valid']
    model.train()
    for epoch in range(epochs):
        #cpuStats()
        #memReport()
        steps = 0
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            #loss = torch.autograd.Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                valid_loss, accuracy = validate_nn(validloader, model, criterion, device)                    
                print(f"Step {steps}/{len(trainloader)}.. ",
                      f"Epoch {epoch+1}/{epochs}.. ",
                      f"Train loss: {running_loss/print_every:.3f}.. ",
                      f"Valid. loss: {valid_loss/len(validloader):.3f}.. ",
                      f"Valid. accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()            

def get_input_args():
    """        
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object
    """
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs=1, default='flowers', help='Path to data directory')
    parser.add_argument('--arch', default='vgg16', help='Model Architecture - vgg16 or densenet161')
    parser.add_argument('--batch_size', type=int, default='64', help='Mini-batch size')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default='0.003', help='Model learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--epochs', type=int, default='1', help='Number of epochs to train the model')
    parser.add_argument('--hidden_units', nargs='+', type=int, help='output sizes for each hidden layer - one per hidden layer')
    return parser.parse_args()
        
if __name__ == '__main__':
    in_args = get_input_args()
    device = detect_and_set_gpu_use(in_args.gpu)
                
    # create a directory to save checkpoints        
    maybe_create_save_dir(in_args.save_dir)
    
    # get data loaders partitioned by training, validation, and test
    loaders, class_to_idx = get_data_loaders(in_args.data_dir[0], 
                                             in_args.batch_size)
    
    # create model and move it to the selected device
    model, criterion, optimizer, classifier_dict = create_model(len(class_to_idx),
                                                                in_args.hidden_units, 
                                                                in_args.arch, 
                                                                in_args.learning_rate)        
    model.to(device)
    
    if in_args.gpu:
        torch.cuda.empty_cache() 
    train_nn(model, in_args.epochs, criterion, optimizer, loaders, device)    
    
    # check accuracy against test data-set
    testloader = loaders['test']
    loss, accuracy = validate_nn(testloader, model, criterion, device)
    print(f"Test loss: {loss/len(testloader):.3f}, ",
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    path = os.path.join(os.getcwd(), in_args.save_dir)
    if os.path.exists(path) and os.path.isdir(path):
        filepath = os.path.join(path, model.name + '-checkpoint.pth')
        print (f"Saving checkpoint in directory: {filepath} ")
        model.class_to_idx = class_to_idx
        save_checkpoint(in_args.epochs, model, optimizer, classifier_dict, filepath)
            

