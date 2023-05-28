import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from pprint import pprint

class TripletNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(args.model_name)
        self.config = self.model.config
        
        self.mulattn_p = nn.MultiheadAttention(embed_dim=self.config.d_model, num_heads=self.config.decoder_attention_heads, dropout=self.config.dropout, batch_first=True)
        self.dropout_p = nn.Dropout(self.config.dropout)
        self.layernorm_p = nn.LayerNorm(self.config.d_model)
        self.ffn_p = nn.Sequential(nn.Linear(self.config.d_model, self.config.d_model*4),
                           nn.ReLU(),
                           nn.Dropout(self.config.dropout),
                           nn.Linear(self.config.d_model*4, self.config.d_model),
                           nn.Dropout(self.config.dropout))
        self.layernorm_pp = nn.LayerNorm(self.config.d_model)
        self.mulattn_n = nn.MultiheadAttention(embed_dim=self.config.d_model, num_heads=self.config.decoder_attention_heads, dropout=self.config.dropout, batch_first=True)
        self.dropout_n = nn.Dropout(self.config.dropout)
        self.layernorm_n = nn.LayerNorm(self.config.d_model)
        self.ffn_n = nn.Sequential(nn.Linear(self.config.d_model, self.config.d_model*4),
                                   nn.ReLU(),
                                   nn.Dropout(self.config.dropout),
                                   nn.Linear(self.config.d_model*4, self.config.d_model),
                                   nn.Dropout(self.config.dropout))
        self.layernorm_nn = nn.LayerNorm(self.config.d_model)
        
        self.proj = nn.Sequential(nn.Linear(self.config.d_model, self.config.d_model*4),
                                  nn.ReLU(),
                                  nn.Linear(self.config.d_model*4, self.config.d_model))
        
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels, positive_masks=None, negative_masks=None, triplet=False):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        
        encoder_outputs = encoder(input_ids = input_ids,
                                  attention_mask = attention_mask)
        
        encoder_hidden_states = encoder_outputs[0]
                
        decoder_outputs = decoder(input_ids = decoder_input_ids,
                                  attention_mask = decoder_attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attention_mask)
        
        decoder_hidden_states = decoder_outputs[0]
        
        lm_logits = self.model.lm_head(decoder_hidden_states)
        predictions = lm_logits.argmax(dim=2)
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        nll = criterion(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if triplet:
            kp_p_mask = (1-positive_masks).type(torch.bool)
            kp_n_mask = (1-negative_masks).type(torch.bool)
            
            hp, _ = self.mulattn_p(decoder_hidden_states, encoder_hidden_states, encoder_hidden_states, key_padding_mask=kp_p_mask)
            hp = self.dropout_p(hp)
            hp = hp + decoder_hidden_states
            hp = self.layernorm_p(hp)
            hp_ = self.ffn_p(hp)
            hp = hp_ + hp
            hp = self.layernorm_pp(hp)
            
            hn, _ = self.mulattn_n(decoder_hidden_states, encoder_hidden_states, encoder_hidden_states, key_padding_mask=kp_n_mask)
            hn = self.dropout_n(hn)
            hn = hn + decoder_hidden_states
            hn = self.layernorm_n(hn)
            hn_ = self.ffn_n(hn)
            hn = hn_ + hn
            hn = self.layernorm_nn(hn)
            
            hp = self.proj(hp)
            hn = self.proj(hn)
            ha = self.proj(decoder_hidden_states)
            
            t_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=2.0, reduction='mean')(ha, hp, hn)
            
            # t_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=2), margin=1.0, reduction='none')(ha, hp, hn)
            # t_loss = t_loss.masked_fill(t_loss==1, 0.0)
            
            # length = torch.sum(decoder_attention_mask, 1, keepdim=True).float()
            # t_loss = (t_loss.sum(dim=1, keepdim=True) / (length + 1e-9)).mean()
            
            return nll, t_loss
            
        return nll, torch.tensor(0), predictions
            
    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / (length + 1e-9)
        
        return avg_hidden
