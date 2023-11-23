import logger
log = logger.get_logger(__file__, 'DEBUG')

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from vocab import Vocab
from collections import namedtuple, defaultdict, Counter
from tqdm import tqdm

import csv
import pdb

Record = namedtuple('Record',
                    [
                        'id_',
                        'date', 'year', 'month', 'day',
                        'store_nbr', 'family', 'onpromotion',
                        'sales'
                    ])

def make_vocab(freq_dict):
    return Vocab(freq_dict, ['UNK'])

def load_sales_csv(path, separator=','):
    records = []
    families = []
    stores = []
    
    with open(path) as f:
        for i, line in tqdm(enumerate(
                csv.reader(f, delimiter=',', quotechar='"'))):
            if i==0: continue
            #if i > 100000: break
            try:
                id_, date, store_nbr, family, sales, onpromotion = line
                year, month, day = date.split('-')
                
                families.append(family)
                stores.append(store_nbr)
            
                records.append(Record(
                    id_,
                    date,
                    int(year),
                    int(month),
                    int(day),
                    store_nbr,
                    family,
                    float(onpromotion),
                    float(sales),
                ))
            except:
                log.exception(line)

    families = make_vocab(Counter(families))
    stores = make_vocab(Counter(stores))
    newrecs = []
    for i, r in enumerate(records):
        if float(r.store_nbr) != 1:
            continue
        newrecs.append(Record(
            r.id_,
            r.date,
            r.year,
            r.month,
            r.day,
            stores[r.store_nbr],
            families[r.family], 
            r.onpromotion,
            r.sales,
        ))

    return newrecs, families, stores

def make_timeseries(records):
    tsrecords = defaultdict(list)
    for record in records:
        index_group = tuple([record.year, record.store_nbr, record.family])
        tsrecords[index_group].append(record)
        
    for key, tsrecord in tsrecords.items():
        tsrecords[key] = sorted(tsrecord, key = lambda x: (x.year, x.month, x.day))
        
    return tsrecords

def split_traintest(tsrecords, ratio=0.8):
    trainset, testset = defaultdict(list), defaultdict(list)
    for key, tsrecord in tsrecords.items():
        pivot = int(len(tsrecord) * ratio)
        trainset[key] = tsrecord[:pivot]
        testset[key] = tsrecord[pivot:]

    return trainset, testset

def sliding_windows(tsrecords, window_size):
    X = []
    Y = []

    for key, tsrecord in tsrecords.items():
        for i in range( len(tsrecord) - window_size - 1 ):
            # x = [
            #     (r.year, r.month, r.day, r.store_nbr, r.family, r.onpromotion)
            #     for r in tsrecord[i:i+window_size]
            # ]
            # y = [ tsrecord[i+window_size].sales ]
            x = tsrecord[i:i+window_size]
            y = tsrecord[i+window_size]
            X.append(x)
            Y.append(y)
        
    return X, Y

def load_data(config, hpconfig, path):
    sales_csv, fvocab, svocab = load_sales_csv(path)
    tssales = make_timeseries(sales_csv)
    trainset, testset = split_traintest(tssales)

    trainset = TSDataset(config, hpconfig,
                         *sliding_windows(trainset, hpconfig['window_size']))
    testset = TSDataset(config, hpconfig,
                        *sliding_windows(testset, hpconfig['window_size']))

    return trainset, testset

class TSDataset(Dataset):
    def __init__(self, config, hpconfig, input_, output):

        self.config   = config
        self.hpconfig = hpconfig

        assert len(input_) == len(output) ,\
            '{} != {}'.format(len(input_), len(output))
        
        self.input_, self.output = input_, output
        log.info('shapes: input_, output: {}, {}'
                 .format(len(self.input_), len(self.output)))

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, index):
        return [self.input_[index],
                self.output[index]]

