# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:26:41 2019

@author: Dan Wickstrom
"""


from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict

def freeze_updates(model, freeze):
    """
    Use with pre-trained models to freeze parameters so the weights are not 
    updated through back-propagation while training.
        Parameters: 
            model - pre-trained model to have weights frozen
            freeze - boolean that indiates if weights should be frozen
        Returns:
            None - model is mutable type which is updated in place
    """    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = not freeze
        
def gen_hidden_units(n_classes, in_features, defaults, args):
    """
    Creates pre-trained model with cusomized classfier output section
        Parameters: 
            in_features - integer number of inputs to the classifier section
            n_classes - integer number of classes to be classified
            defaults - integer array that defines hidden unit inputs and outputs sizes
            args - integer array that defines hidden unit inputs and outputs sizes (passed from cmd line)
        Returns:
           output - array of torch.nn components that define classifier stage
    """    
    output = []        
    if args is None:
        sizes = defaults
    else:
        sizes = args
    sizes.insert(0, in_features)
    sizes.append(n_classes)
    for i in range(len(sizes)-2):
        output.append(("fc{}".format(i), nn.Linear(sizes[i], sizes[i+1], bias=True)))
        output.append(("relu{}".format(i), nn.ReLU()))
        output.append(("do{}".format(i), nn.Dropout(p=0.1)))

    i = len(sizes)-2
    output.append(("fc{}".format(i), nn.Linear(sizes[i], sizes[i+1], bias=True)))
    output.append(('output', nn.LogSoftmax(dim=1)))
            
    return output

def create_model(n_classes, hidden_units, name = 'vgg16', learning_rate=0.003):
    """
    Creates pre-trained model with cusomized classfier output section
        Parameters: 
            n_classes - integer number of classes to be classified
            name - name which identifies pre-trained model to be used
            learning_rate - used to set learning rate of the optimizer
        Returns:
           model - pre-trained model customized classifier output stage
           criterion - initialized criterion object to be used during training
           optimizer - optimizer initialized to train output stage
           classifier_dict - dictionary used to initialize the custom classifier
                             stage
    """    
    if name == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
        hidden_unit_defaults = [4096, 1024]
#        classifier_dict = OrderedDict([
#                                   ('fc1', nn.Linear(in_features, 4096, bias=True)),
#                                   ('relu1', nn.ReLU()),
#                                   ('do1', nn.Dropout(p=0.1)),
#                                   ('fc2', nn.Linear(4096, 1024, bias=True)),
#                                   ('relu2', nn.ReLU()),
#                                   ('do2', nn.Dropout(p=0.1)),
#                                   ('fc3', nn.Linear(1024, n_classes, bias=True)),
#                                   ('output', nn.LogSoftmax(dim=1))
#                                   ])

    elif name == 'densenet161':
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
        hidden_unit_defaults = [1024, 512]
#        classifier_dict = OrderedDict([
#                                   ('fc1', nn.Linear(in_features, 1024, bias=True)),
#                                   ('relu1', nn.ReLU()),
#                                   ('do1', nn.Dropout(p=0.1)),
#                                   ('fc2', nn.Linear(1024, 512, bias=True)),
#                                   ('relu2', nn.ReLU()),
#                                   ('do2', nn.Dropout(p=0.1)),
#                                   ('fc3', nn.Linear(512, n_classes, bias=True)),
#                                   ('output', nn.LogSoftmax(dim=1))
#                                   ])
    else:
        raise Exception('Invalid model selected: {}'.format(name))
        
    print(f"Model architecture: {name}")
    freeze_updates(model, True)
    hidden = gen_hidden_units(n_classes, in_features, hidden_unit_defaults, hidden_units)
    classifier_dict = OrderedDict(hidden)
    model.name = name        
    model.classifier = nn.Sequential(classifier_dict)
    criterion = nn.NLLLoss()    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer, classifier_dict
