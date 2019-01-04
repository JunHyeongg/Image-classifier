import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from PIL import Image
import numpy as np
from workspace_utils import active_session
import seaborn as sb
import argparse
import json

##################################################

parser = argparse.ArgumentParser(description = 'Predict the flower name and probability by using train model')
parser.add_argument('image_path', help = 'image directory')
parser.add_argument('--gpu', action = 'store_true', help = 'if use this, you can use GPU')     
parser.add_argument('--top_k', type = int, default = 5, help = 'predict top k list of flowers and their probability')
parser.add_argument('--category_names', default = 'cat_to_name.json', help = 'Use a mapping of categories to real names')                   
parser.add_argument('checkpoint', default = 'checkpoint', help = 'input your save point name')

args = parser.parse_args()

image_path = args.image_path
gpu = args.gpu
top_k = args.top_k
category_names = args.category_names
checkpoint = args.checkpoint

##################################################

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.state_dict = checkpoint['opti_state_dict']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epoch']

    return model

save_path = checkpoint + '.pth'
optimizer = optim.Adam
model = load_checkpoint(save_path)
                
# Gpu use
if gpu:
    device = torch.device('cuda')                   
else:
    device = torch.device('cpu')
                 
for param in model.parameters():
    param.requires_grad = False
        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    c_width, c_height = image.size
    if c_width < c_height:
        new_height = int(256 * c_height / c_width)
        image = image.resize((256, new_height))
    else:
        new_width = int(256 * c_width / c_height)
        image = image.resize((256, new_width))       
    
    precrop_width, precrop_height = image.size
    left = (precrop_width - 224) / 2
    top = (precrop_height - 224) / 2
    right = (precrop_width + 224) / 2
    bottom = (precrop_height + 224) / 2
    
    area = (left, top, right, bottom)
    crop_im = image.crop(area)
    
    np_image = np.array(crop_im) / 255
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_im = (np_image - means) / std
    img = normalized_im.transpose(2, 0, 1)
    
    return img

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)

    img_t = torch.from_numpy(process_image(img))
    img_t_float = img_t.type(torch.FloatTensor)
    img_t_float_d = img_t_float.unsqueeze_(0)
    
    image = img_t_float_d.to(device)
    model.to(device)
    model.eval()
    logps = model(image)
    ps = torch.exp(logps)
    top_k, top_class = ps.topk(topk, dim = 1)
    index_to_class = {i : c for c, i in model.class_to_idx.items()}
    
    
    class_array = np.array(top_class).reshape(-1)
    classes = list(map(lambda x : index_to_class[x], class_array))
    
    top_K = top_k.cpu()
    probs = top_K.detach().numpy().reshape(-1)
    model.train()
    return probs, classes
       
probs, classes = predict(image_path, model, top_k)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

flower_name = list(map(lambda x : cat_to_name[x], classes))
print(flower_name, probs)