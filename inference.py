import datasets
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_file", type=str, help="save file location")
args = parser.parse_args()

rouge = datasets.load_metric("rouge")
bscore = datasets.load_metric("bertscore")

test_data = dataset.TestDataset(split='test')
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

ckpt = torch.load(args.save_file, map_location=torch.device('cuda'))
print(ckpt['args'])

ckpt_keys = ckpt['state_dict'].keys()
ckpt_keys = list(ckpt_keys)
if ckpt_keys[0][:5] == 'model':    
    new_ckpt = dict((k[6:], v) for (k, v) in ckpt['state_dict'].items())
    model.load_state_dict(new_ckpt, strict=False)
else:
    model.load_state_dict(ckpt['state_dict'])

#%%
for batch in tqdm(test_loader):
    doc = batch[0]
    target = list(batch[1])
    summary_ids = model.generate(doc.cuda(), min_length=11, max_length=62, length_penalty=1.0, early_stopping=True, num_beams=6, no_repeat_ngram_size=3)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    # with open('./outputs/xsum/bart-gold.txt', 'a') as f:
    #     for sent in target:
    #         sent = sent.replace("\n", " ")
    #         f.write(sent+'\n')
    
    # with open('./outputs/xsum/bart-triplet32.txt', 'a') as f:
    #     for sent in summary:
    #         sent = sent.replace("\n", " ")
    #         f.write(sent+'\n')

    rouge.add_batch(predictions=summary, references=target)
    bscore.add_batch(predictions=summary, references=target)

score = rouge.compute(rouge_types=['rouge1','rouge2','rougeL','rougeLsum'], use_stemmer=True)
results = bscore.compute(model_type="distilbert-base-uncased", device='cuda')

print(score['rouge1'].mid.fmeasure,
      '\n',score['rouge2'].mid.fmeasure,
      '\n',score['rougeL'].mid.fmeasure,
      '\n',score['rougeLsum'].mid.fmeasure)

print(np.mean(results['f1']))