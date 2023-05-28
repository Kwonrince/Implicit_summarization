import os
import torch
import datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import BartTokenizer
import json
import pyarrow as pa
import pandas as pd

#%%
class BartDataset(Dataset):
    def __init__(self, datapath, val=False):
        self.val = val
        if self.val:
            self.dataset = datasets.load_from_disk(os.path.join(datapath, 'full/validation'))
        else:
            self.dataset = datasets.load_from_disk(os.path.join(datapath, 'full/train'))

    def __getitem__(self, idx):
        if self.val:
            input_ids = torch.tensor(self.dataset[idx]['input_ids'])
            attention_mask = torch.tensor(self.dataset[idx]['attention_mask'])
            decoder_input_ids = torch.tensor(self.dataset[idx]['decoder_input_ids'])
            decoder_attention_mask = torch.tensor(self.dataset[idx]['decoder_attention_mask'])
            labels = torch.tensor(self.dataset[idx]['labels'])
            
            return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels
            
        else:
            positive_masks = torch.tensor(self.dataset[idx]['positive_masks'])
            negative_masks = torch.tensor(self.dataset[idx]['negative_masks'])
            input_ids = torch.tensor(self.dataset[idx]['input_ids'])
            attention_mask = torch.tensor(self.dataset[idx]['attention_mask'])
            decoder_input_ids = torch.tensor(self.dataset[idx]['decoder_input_ids'])
            decoder_attention_mask = torch.tensor(self.dataset[idx]['decoder_attention_mask'])
            labels = torch.tensor(self.dataset[idx]['labels'])
            
            if (positive_masks.sum() == 0) or (negative_masks.sum() == 0):
                return None
            
            return positive_masks, negative_masks, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels
    
    def __len__(self):
        return len(self.dataset)

class TestDataset(Dataset):
    def __init__(self, split='test'):
        # self.dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)
        self.dataset = datasets.load_dataset("xsum", split=split)
        # self.dataset = datasets.load_dataset("reddit_tifu", 'long', split='train[37925:]')
        # self.dataset = datasets.load_dataset("samsum", split=split)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
    def __getitem__(self, idx):
        # d = self.tokenizer.encode(self.dataset['article'][idx])
        d = self.tokenizer.encode(self.dataset['document'][idx])
        # d = self.tokenizer.encode(self.dataset['documents'][idx])
        # d = self.tokenizer.encode(self.dataset['dialogue'][idx])
        if len(d) > 1024:
            d = d[:1023] + [2]
        else:
            d = d+[1]*(1024-len(d))
        
        tokenized_d = torch.tensor(d)
        # target = self.dataset['highlights'][idx]
        target = self.dataset['summary'][idx]
        # target = self.dataset['tldr'][idx]
        # target = self.dataset['summary'][idx]
        
        return tokenized_d, target
    
    def __len__(self):
        return len(self.dataset)


def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data



class NYTTestDataset(Dataset):
    def __init__(self, split='test'):
        self.dataset = load_jsonl('./nyt/test.jsonl')
        self.test_doc = [contents['document']['text'] for contents in self.dataset]
        self.test_sum = [contents['summary']['text'] for contents in self.dataset]
        self.dataset = pd.DataFrame({'text':self.test_doc, 'summary':self.test_sum})
        self.dataset = datasets.Dataset(pa.Table.from_pandas(self.dataset))
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
    def __getitem__(self, idx):
         d = self.tokenizer.encode('\n'.join(self.dataset[idx]['text']))
         if len(d) > 1024:
             d = d[:1023] + [2]
         else:
             d = d+[1]*(1024-len(d))
         
         tokenized_d = torch.tensor(d)
         target = self.dataset[idx]['summary']
         # target = self.dataset['summary'][idx]
         # target = self.dataset['tldr'][idx]
         # target = self.dataset['summary'][idx]
         
         return tokenized_d, target
     
    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_loader(batch_size, num_workers, datapath):    
    train_loader = DataLoader(dataset=BartDataset(datapath),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(dataset=BartDataset(datapath, val=True),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, val_loader


def get_dist_loader(batch_size, num_workers, datapath):
    train_dataset = BartDataset(datapath)
    val_dataset = BartDataset(datapath, val=True)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              pin_memory=True,
                              batch_size=batch_size,
                              shuffle=None,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(dataset=val_dataset,
                            sampler=val_sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=num_workers)
    
    return train_loader, val_loader, train_sampler, val_sampler