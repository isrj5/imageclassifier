# Imports here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import OrderedDict



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')



args, _ = parser.parse_known_args()

def preprocessing(args):
    data_dir = args.data_dir
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'

    data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                                ]),
            'valid': transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])
                                ]),
            'test': transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                                ])
                                }
    
    # Load the datasets with ImageFolder
    image_datasets = {
            x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
            for x in list(data_transforms.keys())
            }
    # Using the image datasets, define the dataloaders
    dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True)
            for x in list(image_datasets.keys())
            }
    dataset_sizes = {
            x: len(dataloaders[x].dataset) 
            for x in list(image_datasets.keys())
            }    
    return image_datasets,dataloaders, dataset_sizes

def build_model(args, arch='densenet121',dropout=0.5, hidden_units = 120):
    
    structures = {"vgg16":25088
              "densenet121" : 1024,
              "alexnet" : 9216 }

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("INVALID MODEL")
        
    
        
    for param in model.parameters():
        param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[arch], hidden_units)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_units, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
        model.classifier = classifier
       
        model.cuda()
        
        return model 

    


def train_model(model,arch='densenet121', criterion, optimizer, exp_lr_scheduler,num_epochs=12):
    image_datasets,dataloaders, dataset_sizes= preprocessing()
    since = time.time()
    model= build_model('densenet121')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
   
     
    
           
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if gpu and torch.cuda.is_available():
                model.cuda()
           
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase],0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available() :
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward propagation and optimization
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # slosss in each epoch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copying
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Accuracy: {:4f}'.format(best_acc))

    
    model.load_state_dict(best_model_wts)
    
    return model
def save_checkpoint(args):

    image_datasets,dataloaders, dataset_sizes = preprocessing(args)
    structures = {"vgg16":25088
              "densenet121" : 1024,
              "alexnet" : 9216 }

    # 1. Load a pre-trained network
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # 2. Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

    classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[arch], args.hidden_units)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(args.hidden_units, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))

    # Reserve for final layer: ('output', nn.LogSoftmax(dim=1))
        
    model.classifier = classifier

    # 3. Train the classifier layers using backpropagation using the pre-trained network to get the features
    # 4. Track the loss and accuracy on the validation set to determine the best hyperparameters
        
    if args.gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            print ("Using GPU: "+ str(use_gpu))
        else:
            print("Using CPU since GPU is not available/configured")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model = train_model(args, model, criterion, optimizer, exp_lr_scheduler,num_epochs=args.epochs)

    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = args.epochs
    checkpoint = {'hidden_units': 120,
                  'batch_size': dataloaders['train'].batch_size,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
    torch.save(checkpoint, args.saved_model)
def main():
    parser = argparse.ArgumentParser(description='Flower Classifcation trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='densenet121', help='architecture [available: densenet121, vgg16,alexnet]', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=120, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=12, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--saved_model' , type=str, default='checkpoints.pth', help='path of your saved model')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    save_checkpoint(args)


if __name__ == "__main__":
    main()