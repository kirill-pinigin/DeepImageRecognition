import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from  CustomDataset import  CustomDataset

from DeepImageRecognition import DeepImageRecognition , MultiLabelLoss
from ResidualRecognitron import  ResidualRecognitron
from MobileRecognitron import MobileRecognitron

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',          type = str,   default='./CocoDatasetTags/', help='path to dataset')
parser.add_argument('--result_dir',        type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--recognitron',       type = str,   default='ResidualRecognitron', help='type of image generator')
parser.add_argument('--activation',        type = str,   default='ReLU', help='type of activation')
parser.add_argument('--criterion',         type = str,   default='BCE', help='type of criterion')
parser.add_argument('--optimizer',         type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--lr',                type = float, default=1e-4)
parser.add_argument('--weight_decay',      type = float, default=1e-4)
parser.add_argument('--dropout',           type = float, default=0.0)
parser.add_argument('--batch_size',        type = int,   default=128)
parser.add_argument('--epochs',            type = int,   default=64)
parser.add_argument('--pretrained',        type = bool,  default=True)
parser.add_argument('--transfer',          type = bool,  default=False)
parser.add_argument('--resume_train',      type = bool,  default=True)

args = parser.parse_args()

recognitron_types = {
                        'ResidualRecognitron'        : ResidualRecognitron,
                        'MobileRecognitron'          : MobileRecognitron,
                    }

activation_types =  {
                        'ReLU'     : nn.ReLU(),
                        'LeakyReLU': nn.LeakyReLU(),
                        'PReLU'    : nn.PReLU(),
                        'ELU'      : nn.ELU(),
                        'SELU'     : nn.SELU(),
                    }

criterion_types =   {
                        'MSE'            : nn.MSELoss(),
                        'L1'             : nn.L1Loss(),
                        'BCE'            : nn.BCELoss(),
                        'MultiLabelLoss' : MultiLabelLoss()
                    }

optimizer_types =   {
                        'Adam'   : optim.Adam,
                        'RMSprop': optim.RMSprop,
                        'SGD'    : optim.SGD
                    }

model = (recognitron_types[args.recognitron] if args.recognitron in recognitron_types else recognitron_types['ResidualRecognitron'])

function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])

recognitron = model(activation=function, pretrained=args.pretrained + args.transfer)

optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(recognitron.parameters(), lr = args.lr, weight_decay = args.weight_decay)

criterion = (criterion_types[args.criterion] if args.criterion in criterion_types else criterion_types['BCE'])

shuffle_list = { 'train' : True, 'val' : False}
augmentation = { 'train' : True, 'val' : False}

image_datasets = {x: CustomDataset(os.path.join(args.data_dir, x), os.path.join(args.data_dir, x +'/'  + x + '.csv'), augmentation=augmentation[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,shuffle=shuffle_list[x], num_workers=4)  for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=4)

framework = DeepImageRecognition(recognitron = recognitron, criterion = criterion, optimizer = optimizer, directory = args.result_dir)
framework.save("FAKE")

if args.transfer:
    framework.recognitron.freeze()
    framework.optimizer = (optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(recognitron.parameters(), lr = args.lr / 2, weight_decay = args.weight_decay)
    framework.approximate(dataloaders=dataloaders, num_epochs= int(args.epochs / 2) if int(args.epochs / 2) < 12 else 12, resume_train=args.resume_train, dropout_factor=args.dropout)
    framework.recognitron.unfreeze()

framework.approximate(dataloaders = dataloaders, num_epochs=args.epochs, resume_train=args.resume_train, dropout_factor=args.dropout)
framework.estimate(testloader)

