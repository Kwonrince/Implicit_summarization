import os
import time
import datasets

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from dataset import get_dist_loader, get_loader
from model import TripletNetwork
from transformers import AdamW, get_scheduler
from train_utils import (cal_running_avg_loss, eta, progress_bar,
                         time_since, user_friendly_time)
#%%
class Trainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = args.tokenizer
        self.model_dir = args.model_dir
        self.rouge = datasets.load_metric("rouge")
        
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None
        
        self.model = None
        self.optimizer = None
        
    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]
        self.args.rank = self.args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                world_size=self.args.world_size, rank=self.args.rank)
        self.model = TripletNetwork(self.args)
        
        torch.cuda.set_device(self.args.gpu)
        self.model.cuda(self.args.gpu)
        
        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        self.args.workers = (self.args.workers + ngpus_per_node - 1) // ngpus_per_node
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self.get_data_loader()
        
        param = self.model.parameters()
        
        self.optimizer = AdamW(param, lr=self.args.lr, no_deprecation_warning=True)
        self.lr_scheduler = get_scheduler("linear",
                                          optimizer=self.optimizer,
                                          num_warmup_steps=int(self.args.num_warmup_steps * self.args.num_epochs * len(self.train_loader)),
                                          num_training_steps=self.args.num_epochs * len(self.train_loader))
        self.model = DistributedDataParallel(self.model,
                                             device_ids=[self.args.gpu],
                                             find_unused_parameters=True)
        
        cudnn.benchmark = True
        
    def get_data_loader(self):
        train_loader, val_loader, train_sampler, val_sampler = get_dist_loader(batch_size=self.args.batch_size,
                                                                               num_workers=self.args.workers,
                                                                               datapath=self.args.dataset_path)
        
        return train_loader, val_loader, train_sampler, val_sampler
    
    def train(self, model_path=None):
        running_avg_loss = 0.0
        running_avg_t_loss = 0.0
        
        best_loss = 1e5
        batch_nb = len(self.train_loader)
        step = 1
        self.model.zero_grad()
        for epoch in range(1, self.args.num_epochs+1):
            start = time.time()
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                batch = tuple(v.to(self.args.gpu) for v in batch) # batch : {positive_masks:tensor, negati~~~} => (tensor, tensor, ~~~)
            
                if self.args.triplet:
                    positive_masks, negative_masks, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
                    
                    nll, t_loss = self.model(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              decoder_input_ids=decoder_input_ids,
                                              decoder_attention_mask=decoder_attention_mask,
                                              labels=labels,
                                              positive_masks=positive_masks,
                                              negative_masks=negative_masks,
                                              triplet=self.args.triplet)

                else:
                    _, _, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
                    
                    nll, t_loss, _ = self.model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                decoder_input_ids=decoder_input_ids,
                                                decoder_attention_mask=decoder_attention_mask,
                                                labels=labels,
                                                positive_masks=None,
                                                negative_masks=None,
                                                triplet=self.args.triplet)
                    
                loss = nll + self.args.weight * t_loss
                # loss = loss / self.args.accum_steps
                loss.backward()
                                
                
                # if (batch_idx + 1) % self.args.accum_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()
                
                running_avg_loss = cal_running_avg_loss(nll.item(), running_avg_loss)
                running_avg_t_loss = cal_running_avg_loss(t_loss.item(), running_avg_t_loss)
                
                msg = "{}/{} {} - ETA : {} - nll: {:.4f}, triplet: {:.4f}".format(
                    batch_idx, batch_nb,
                    progress_bar(batch_idx, batch_nb),
                    eta(start, batch_idx, batch_nb),
                    running_avg_loss, running_avg_t_loss)
                print(msg, end="\r")
                step += 1

            # evaluate model on validation set
            if self.args.rank == 0:
                val_nll, rouge1, rouge2, rougel = self.evaluate(msg)
                # if val_nll < best_loss:
                #     best_loss = val_nll
                self.save_model(val_nll, epoch)

                print("Epoch {} took {} - Train NLL: {:.4f} - val NLL: "
                      "{:.4f} - triplet: {:.4f} - Rouge1: {:.4f} - Rouge2: {:.4f} - RougeL:{:.4f} ".format(epoch,
                                                                                                            user_friendly_time(time_since(start)),
                                                                                                            running_avg_loss,
                                                                                                            val_nll,
                                                                                                            running_avg_t_loss,
                                                                                                            rouge1,
                                                                                                            rouge2,
                                                                                                            rougel))
                
    def evaluate(self, msg):
        val_batch_nb = len(self.val_loader)
        val_losses = []
        self.model.eval()
        for i, batch in enumerate(self.val_loader, start=1):
            batch = tuple(v.to(self.args.gpu) for v in batch)
            
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
            
            with torch.no_grad():
                nll, _, predictions = self.model(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  decoder_input_ids=decoder_input_ids,
                                                  decoder_attention_mask=decoder_attention_mask,
                                                  labels=labels)
            
            self.rouge.add_batch(predictions=predictions, references=labels)
            
            msg2 = "{} =>   Evaluating : {}/{}".format(msg, i, val_batch_nb)
            print(msg2, end="\r")
            val_losses.append(nll.item())

        score = self.rouge.compute(rouge_types=['rouge1','rouge2','rougeLsum'])
        val_loss = np.mean(val_losses)

        return val_loss, score['rouge1'][1][2], score['rouge2'][1][2], score['rougeLsum'][1][2]
    
    
    def save_model(self, loss, epoch):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {"args":self.args,
                "state_dict":model_to_save.state_dict(), # model_to_save.model.state_dict()
                }
        model_save_path = os.path.join(
            self.model_dir, "{}_{:.4f}.pt".format(epoch, loss))
        torch.save(ckpt, model_save_path)