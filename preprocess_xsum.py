import datasets
from rouge_score import rouge_scorer
from nltk import tokenize
from transformers import BartTokenizer, AddedToken
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings(action='ignore')


def preprocess_xsum(model_name='facebook/bart-base', article_length=1024, summary_length=128, train_split='train', val_split="validation"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'bos_token':AddedToken('<s>', lstrip=False)})
    tokenizer.add_special_tokens({'eos_token':AddedToken('</s>', lstrip=False)})
    
    train_data = datasets.load_dataset("xsum", split=train_split)
    validation_data = datasets.load_dataset("xsum", split=val_split)#)['validation']

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    positive_masks = []
    negative_masks = []
    
    print("Prepairing Datasets.....")
    for idx in tqdm(range(len(train_data))):
        token_target = train_data['document'][idx].split('\n')
        token_ref = train_data['summary'][idx]
        rouge_df = pd.DataFrame(columns=['rougeL', 'target'])
        
        for j in range(len(token_target)):
            scores = scorer.score(token_target[j], token_ref)
            temp = [{'rougeL':scores['rougeL'][2], 'target':j}]
            rouge_df = pd.concat([rouge_df, pd.DataFrame(temp)], ignore_index=True)

        unimportance_idx = sorted(rouge_df.loc[rouge_df['rougeL'] <= rouge_df['rougeL'].quantile(.15)]['target'].unique())
        importance_idx = sorted(rouge_df.loc[rouge_df['rougeL'] >= rouge_df['rougeL'].quantile(.85)]['target'].unique())
        
        token_positive = token_target.copy()
        for i in importance_idx:
            token_positive[i] = "<s>" + token_positive[i] + "</s>"
        token_positive = '\n'.join(token_positive)
        
        temp = np.array([-10]+tokenizer.encode(token_positive, add_special_tokens=False)+[-20])
        p_idx = {'start':list(np.where(temp==0)[0]), 'end':list(np.where(temp==2)[0])}
    
        positive_mask = np.array([0] * article_length)
        for i in range(len(p_idx['start'])):
            positive_mask[(p_idx['start'][i]-i*2) : (p_idx['end'][i]-(i*2+2)+1)] = 1
        if len(positive_mask) < article_length:
            positive_masks.append(np.append(positive_mask, np.array([0]*(article_length-len(positive_mask)))))
        else:
            positive_masks.append(positive_mask[:article_length])
        
        token_negative = token_target.copy()
        for j in unimportance_idx:
            token_negative[j] = "<s>" + token_negative[j] + "</s>"
        token_negative = '\n'.join(token_negative)
        
        temp = np.array([-10]+tokenizer.encode(token_negative, add_special_tokens=False)+[-20])
        n_idx = {'start':list(np.where(temp==0)[0]), 'end':list(np.where(temp==2)[0])}
    
        negative_mask = np.array([0] * article_length)
        for i in range(len(n_idx['start'])):
            negative_mask[(n_idx['start'][i]-i*2) : (n_idx['end'][i]-(i*2+2)+1)] = 1
        if len(negative_mask) < article_length:
            negative_masks.append(np.append(negative_mask, np.array([0]*(article_length-len(negative_mask)))))
        else:
            negative_masks.append(negative_mask[:article_length])
    
    train_data = train_data.add_column('positive_masks', positive_masks)
    train_data = train_data.add_column('negative_masks', negative_masks)
    
    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        remove_columns=["document", "summary", "id"]
    )
    
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                               "decoder_attention_mask", "labels", "positive_masks", "negative_masks"],
    )
    
    validation_data = validation_data.map(
        process_data_to_model_inputs,
        batched=True,
        remove_columns=["document", "summary", "id"]
    )
    
    validation_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                               "decoder_attention_mask", "labels"],
    )
    
    return train_data, validation_data

def process_data_to_model_inputs(batch, article_length=1024, summary_length=128):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=article_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=summary_length+1)
    
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = [ids[:-1] if ids[-1] != 2 else ids[:-2] + [tokenizer.eos_token_id] for ids in outputs.input_ids]
    batch['decoder_attention_mask'] = [[0 if token == tokenizer.pad_token_id else 1 for token in inputs] for inputs in batch['decoder_input_ids']]
    
    batch["labels"] = [ids[1:] + [tokenizer.pad_token_id] for ids in batch['decoder_input_ids']]
    # We have to make sure that the PAD token is ignored for calculating the loss
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
      
    return batch


def save(train_data, validation_data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    train_data.save_to_disk(f'{path}/train')
    validation_data.save_to_disk(f'{path}/validation')

if __name__ == '__main__':
    past_t = 0
    past_v = 0
    for i, (t, v) in enumerate(zip(range(10000, 204045, 10000), range(539, 11332, 539))):
        train_data, validation_data = preprocess_xsum(train_split='train['+str(past_t)+f':{t}]', val_split='validation['+str(past_v)+f':{v}]')
        save(train_data, validation_data, path=f'./xsum_15/full_{i}')
        past_t = t
        past_v = v
    train_data, validation_data = preprocess_xsum(train_split='train['+str(past_t)+':]', val_split='validation['+str(past_v)+':]')
    save(train_data, validation_data, path='./xsum_15/full_20')

    train_data = datasets.load_from_disk('./xsum_15/full_0/train')
    for i in tqdm(range(1, 21)):
        new_train_data = datasets.load_from_disk(f'./xsum_15/full_{i}/train')
        train_data = datasets.concatenate_datasets([train_data, new_train_data])
    
    validation_data = datasets.load_from_disk('./xsum_15/full_0/validation')
    for i in tqdm(range(1, 21)):
        new_validation_data = datasets.load_from_disk(f'./xsum_15/full_{i}/validation')
        validation_data = datasets.concatenate_datasets([validation_data, new_validation_data])

    save(train_data, validation_data, path='./xsum_15/full')
