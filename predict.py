
# Imports python modules
import json
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from project_utils import detect_and_set_gpu_use, load_checkpoint
import argparse

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
    # scale and crop image
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

    # normalize image
    np_image = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])    
    image_flt = np_image/255
    norm_image = (image_flt - mean)/std

    # transpose image    
    trans_image = norm_image.transpose((2,0,1))
    return torch.tensor(trans_image, device=device)

def get_idx_to_class(model):
    ''' Load idx_to_class mappings.
    '''    
    class_to_idx = model.class_to_idx
    idx_to_class = dict([[v,k] for k,v in class_to_idx.items()])
    return idx_to_class

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    idx_to_class = get_idx_to_class(model)
    image = Image.open(image_path)
    pimage = process_image(image)
    inputs = pimage
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(inputs[None].float())
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
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
    
def get_input_args():
    """        
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', nargs=1, default=argparse.FileType('r'), help='Path to image file')
    parser.add_argument('checkpoint', nargs=1, default=argparse.FileType('r'), help='Path to checkpoint file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--top_k', type=int, default='5', help='Return top k most likely classes')
    parser.add_argument('--category_names', help='Mapping of categories to real names')
    return parser.parse_args()

if __name__ == '__main__':
    # get cmd line args
    in_args = get_input_args()
    print(in_args.category_names)
    device = detect_and_set_gpu_use(in_args.gpu)

    # get image filename
    fname = in_args.image_path[0]
    print('image: {}'.format(fname))
    
    # and checkpoint file
    checkpoint = in_args.checkpoint[0]
    print(checkpoint)

    # load checkpoint file 
    model_t, _ = load_checkpoint(checkpoint)
    
    # move model to gpu/cpu and do prediction
    model_t.to(device);
    probs, classes = predict(fname, model_t, in_args.top_k)
    
    # print out results
    print(probs)
    print(classes)

    # and category filename
    names = []
    cat_names_file = in_args.category_names
    if not cat_names_file is None:
        with open(cat_names_file, 'r') as f:
            cat_to_name = json.load(f)
        names = [cat_to_name[c] for c in classes]
        print(names)

