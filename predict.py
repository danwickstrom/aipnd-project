
# Imports python modules
import json
import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision import datasets, models
from matplotlib import pyplot as plt

import argparse

device = torch.device('cpu')

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
    parser.add_argument('image_path', nargs=1, default=argparse.FileType('r'), help='Path to image file')
    parser.add_argument('checkpoint', nargs=1, default=argparse.FileType('r'), help='Path to checkpoint file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--top_k', type=int, default='5', help='Return top k most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Mapping of categories to real names')
    return parser.parse_args()

def imshow(image, ax=None, title=None):
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


def process_image(img, convert=True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    if convert:
        new_width = 224    
        new_height = 224
        width, height = img.size
        ratio = float(width)/float(height)        
        if width > height:
            width = int(256.0 * ratio)
            height = 256
        else:
            ratio = 1.0/ratio
            height = int(256 * ratio)
            width = 256

        img.thumbnail((width, height), Image.NEAREST)
        left = (width - new_width)/2
        right = (width + new_width)/2
        top = (height - new_height)/2
        bottom = (height + new_height)/2
        img = img.crop((left, top, right, bottom))        

    np_image = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image_flt = np_image/255
    norm_image = (image_flt - mean)/std
    
    trans_image = norm_image.transpose((2,0,1))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(trans_image, device=device)

def get_idx_to_class(data_dir):
    ''' Load idx_to_class mappings.
    '''    
    train_data = datasets.ImageFolder(data_dir + '/train')
    class_to_idx = train_data.class_to_idx
    idx_to_class = dict([[v,k] for k,v in class_to_idx.items()])
    return idx_to_class

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    idx_to_class = get_idx_to_class('./flowers')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path)
    pimage = process_image(image)
    inputs = pimage
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        # print(pimage[None].shape)
        logps = model.forward(inputs[None].float())
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(5, dim=1)
        probs = [float(p) for p in top_p[0]]
        classes = [idx_to_class[int(i)] for i in top_class[0]]
        
        return probs, classes

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier_dict = checkpoint['classifier']
    model.classifier = nn.Sequential(classifier_dict)
    model.load_state_dict(checkpoint['state_dict'])    
    del(checkpoint)
    
    return model



def cat_to_names():
    ''' Load category to name mappings.
    '''    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        num_classes = len(cat_to_name.keys())
        return cat_to_name, num_classes
    
    
if __name__ == '__main__':
    in_arg = get_input_args()
    device = device = torch.device("cpu")
    if in_arg.gpu and torch.cuda.is_available():
        print('Using GPU')
        device = torch.device("cuda")

    #fname = 'flowers/test/1/image_06743.jpg' 
    fname = in_arg.image_path[0]
    checkpoint = in_arg.checkpoint[0]
    print(checkpoint)
    cat_names_file = in_arg.category_names
    with open(cat_names_file, 'r') as f:
        cat_to_name = json.load(f)
    model_t = load_checkpoint('checkpoint.pth')
    #model_t = load_checkpoint(checkpoint)
    model_t.to(device);
    print('image: '.format(fname))
    probs, classes = predict(fname, model_t)
    print(probs)
    print(classes)
    names = [cat_to_name[c] for c in classes]
    print(names)

