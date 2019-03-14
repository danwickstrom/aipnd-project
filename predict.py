
# Imports python modules
import json
import torch
from PIL import Image
import numpy as np
from torchvision import datasets
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
    parser.add_argument('--gpu', default='true', help='Use GPU for training')
    parser.add_argument('--top_k', default='5', help='Return top k most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Mapping of categories to real names')
    return parser.parse_args()

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

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    idx_to_class = get_idx_to_class('./')
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

def cat_to_names():
    ''' Load category to name mappings.
    '''    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        num_classes = len(cat_to_name.keys())
        return cat_to_name, num_classes
    
def get_idx_to_class(data_dir):
    ''' Load idx_to_class mappings.
    '''    
    train_data = datasets.ImageFolder(data_dir + '/train')
    class_to_idx = train_data.class_to_idx
    idx_to_class = dict([[v,k] for k,v in class_to_idx.items()])
    return idx_to_class
    
if __name__ == '__main__':
    