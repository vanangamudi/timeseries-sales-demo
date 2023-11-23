import logger
log = logger.get_logger(__file__, 'DEBUG')

import os
import sys
import pickle
import json
import argparse
import random

from pprint import pprint
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from model import LSTMModel
from dataset import TSDataset, load_data

from functools import partial

import utils
import trainer

# def load_data(path):
#     return pickle.load(open(path, 'rb'))

def plot(trainer, epoch, testsets):
    fig = plt.figure()
    colours = ['r','b','g','gold'] 
    symbols = ['o','o','^','^']
    sizes = [50, 50, 25, 25]
    offset = 0
    for i, (name, testset) in enumerate(testsets.items()):
        testloader = DataLoader(testset,
                                shuffle=hpconfig['testset_shuffle'],
                                batch_size = hpconfig["batch_size"],
                                )
                
        input_, target, output = trainer.eval_epoch(epoch, testloader)
        input_, target, output = [i.detach().cpu()
                                  for i in [input_, target, output]]

        plt.scatter(range(offset, offset + target.size(0)), target,
                    label = 'x - {}'.format(name),
                    s=sizes[2*i], c = colours[2*i],
                    marker = symbols[2*i], alpha = 0.75)
        
        plt.scatter(range(offset, offset + target.size(0)), output,
                    label='x\' - {}'.format(name),
                    s=sizes[2*i+1], c = colours[2*i+1],
                    marker = symbols[2*i+1], alpha = 0.75)
        offset += target.size(0)
    plt.legend()
    plt.show()

def mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def create_parser():
    parser = argparse.ArgumentParser(__file__)
    
    parser.add_argument('-o', '--output',
                        help='output file path',
                        default='data.csv')
    
    parser.add_argument('-c', '--config',
                        help='config file path',
                        default='config.json')

    parser.add_argument('-p', '--hpconfig',
                        help='hyperparameter config file path',
                        default='hpconfig.json')
    
    parser.add_argument('-d', '--dataset',
                        help='dataset file path',
                        )
    
    parser.add_argument('-r', '--reset',
                        action='store_true',
                        help='reset training from scratch',
                        )

    return parser

def collate_fn(data):
    input_, output = zip(*data)

    X = []
    for x in input_:
        X.append(torch.tensor([x.year, x.month, x.day, x.store_nbr, x.family]))
        
    return features.float(), labels.long(), lengths.long()

if __name__ == '__main__':
 
    parser = create_parser()      
    args   = parser.parse_args()  

    config   = json.load(open(args.config)) 
    hpconfig = json.load(open(args.hpconfig))
    
    config['hpconfig_name'] = args.hpconfig
    pprint(hpconfig)
    
    utils.init_config(config, hpconfig)
    assert utils.config != None

    pprint(config)

    trainset, testset = load_data(config, hpconfig, args.dataset)
    #plt.plot([r.sales for r in random.choice(trainset.input_)])
    #plt.show()
        
    #model = Model(utils.config, hpconfig,  input_.size(1),  output.size(0))
    
    trainloader = DataLoader(trainset,
                             collate_fn = collate_fn,
                             batch_size = hpconfig["batch_size"],
                             )
    
    testloader = DataLoader(testset,
                            collate_fn = collate_fn,
                            batch_size = hpconfig["batch_size"],
                            )

    for i in trainloader:
        break
    
    print(model)
    trainer = trainer.Trainer (
        utils.config,
        hpconfig,
        model,
        torch.nn.L1Loss(),
        torch.optim.Adam(model.parameters()),
        
        trainloader,
        testloader,        
        batch_size = 100,
    )

    if not args.reset:
        trainer.load_state()
    
    trainer.do_train()
    plot(trainer,
         None,
         dict(train=trainset,
              test=testset))
