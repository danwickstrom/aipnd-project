# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:09:05 2019

@author: daniel
"""

# Imports python modules
import argparse

def get_input_args():
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these command line arguments. If
    the user fails to provide some or all of the arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
        
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='checkpoints', help='Set directory to save checkpoints')
    parser.add_argument('--arch', default='vgg16', help='Pre-trained Model Architecture')
    parser.add_argument('--epochs', default='10', help='Number of training epochs')
    parser.add_argument('--learning_rate', default='0.03', help='Training learning rate')
    parser.add_argument('--gpu', default='true', help='Use GPU for training')
    parser.add_argument('--top_k', default='5', help='Return top k most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Mapping of categories to real names')
    return parser.parse_args()