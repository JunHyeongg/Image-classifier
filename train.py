import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
from workspace_utils import active_session
import argparse

#################################################
parser = argparse.ArgumentParser(description = 'Training part for Image Classifier')
parser.add_argument('data_dir', help = 'This is a data_directory')
parser.add_argument('--arch', choices = ['densenet121', 'vgg13'], default = 'densenet121', help = 'Choose Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Choose learning rate')
parser.add_argument('--hidden_1', type = int, default = 512, help = 'Choose number of first hidden layer')
parser.add_argument('--hidden_2', type = int, default = 256, help = 'Choose number of second hidden layer')                   
parser.add_argument('--gpu', action = 'store_true', help = 'if use this, you can use GPU')          
parser.add_argument('--epochs', type = int, default = 1, help = 'Choose repetition time')
parser.add_argument('--save_dir', help = 'Set directory to save checkpoint', default = 'checkpoint.pth')
                   
args = parser.parse_args()
                   
data_dir = args.data_dir                    
arch = args.arch
lr = args.learning_rate
h1 = args.hidden_1
h2 = args.hidden_2
gpu = args.gpu
E = args.epochs
save_dir = args.save_dir     

##################################################

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size = 32)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
  
# choose Arichitecture
if arch == 'densenet121':
    model = models.densenet121(pretrained = True)
    input_n, drop_out = 1024, 0.3      
                    
elif arch == 'vgg13':              
    model = models.vgg13(pretrained = True)         
    input_n, drop_out = 25088, 0.4              
                   
# Gpu mode 
if gpu:
    device = torch.device('cuda')                   
else:
    device = torch.device('cpu')
                 
for param in model.parameters():
    param.requires_grad = False
                   
# Building classifier                    
model.classifier = nn.Sequential(nn.Linear(input_n, h1),
                                 nn.ReLU(),
                                 nn.Dropout(drop_out),
                                 nn.Linear(h1, h2),
                                 nn.ReLU(),
                                 nn.Dropout(drop_out),
                                 nn.Linear(h2, 102),
                                 nn.LogSoftmax(dim = 1))
                    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
model.to(device)

# train model
epochs = E
steps = 0
running_loss = 0

with active_session():

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        else:
            valid_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                model.eval()
                for image, label in validloader:
                    image, label = image.to(device), label.to(device)
                    logps_v = model.forward(image)
                    
                    valid_loss += criterion(logps_v, label)

                    ps = torch.exp(logps_v)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = (top_class == label.view(*top_class.shape))
                    accuracy += torch.mean(equals.type(torch.FloatTensor))


            print(f'Step {steps}.. '
                  f'Epoch {e+1}/{epochs}.. '
                  f'Train loss: {running_loss/len(trainloader):.3f}.. '
                  f'Valid loss: {valid_loss/len(validloader):.3f}.. '
                  f'Valid accuracy: {100*accuracy/len(validloader):.3f}%')


            model.train()

model.eval()                   
# checkpoint options for save                   
model.class_to_idx = train_data.class_to_idx
checkpoint = {
              'arch' : 'densenet121',
              'classifier' : model.classifier,
              'class_to_idx' : model.class_to_idx,
              'state_dict' : model.state_dict(),
              'opti_state_dict' : optimizer.state_dict(),
              'epoch' : epochs
             }

torch.save(checkpoint, 'checkpoint.pth')
                   