import sys
import os
import importlib
import models

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchsparse.nn as spnn
from torchsparse.utils.collate import sparse_collate

from transformers import BertConfig
importlib.reload(models)

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "models"))  # HACK add the lib folder

from models.basic_blocks import SparseConvEncoder
from models.landlang_module import LandLangModule


class CityRefer(nn.Module):
    def __init__(self, args, input_feature_dim=0, num_object_class=None, vocab_size=None, pad_token_id=0):
        super().__init__()
        self.args = args
        self.input_feature_dim = input_feature_dim        
        self.num_object_class = num_object_class
        self.drop_rate = args.drop_rate

        self.use_lang_classifier=(not args.no_lang_cls),
        hidden_size = args.hidden_size
        
        # --------- Language Encoder ---------
        embed_dim = hidden_size  
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id) #, **factory_kwargs)
        self.lang_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=args.num_hidden_layers,
            batch_first=True,
            bidirectional=True, 
        )        

        # --------- Point Encoder ---------
        # Sparse Volumetric Backbone
        self.sparse_conv = SparseConvEncoder(self.input_feature_dim) # self.input_feature_dim = 3 -> 128
        self.pooling = spnn.GlobalMaxPool()
        self.fuse = nn.Sequential(
            nn.Linear(128 + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )            
        
        # --------- Geo Encoder ---------
        self.max_num_object = args.max_num_object if args.num_cands < 0 else args.num_cands
        self.geo_gru = nn.GRU(
            input_size=embed_dim, #128,
            hidden_size=hidden_size,
            num_layers=args.num_hidden_layers,
            batch_first=True,
            bidirectional=True, 
        )

        # --------- Landmark Encoder ---------
        if self.args.use_landmark_name:
            self.landmark_lang = LandLangModule(num_object_class, vocab_size, True, args.use_bidir, hidden_size, hidden_size, args.max_num_landmark, pad_token_id) 
            self.landmark_concat = nn.Linear(hidden_size * 2, hidden_size)        
            
        # --------- Language Classifier ---------
        if self.use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),                
                nn.Dropout(self.drop_rate), #p=args.lang_pdrop),
                nn.Linear(hidden_size, num_object_class),
            )  
            
        # --------- Confidence Esitimator ---------
        self.object_cls = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(self.drop_rate),
                nn.Linear(hidden_size, 1)
        )                  

    def forward(self, data_dict):
        device = data_dict['input_ids'].device
        
        input_ids = data_dict['input_ids']
        input_shape = input_ids.size()
        
        # --- Language Encoding ---
        lang_embeds = self.word_embeddings(input_ids)        
        lang_tokens_len = data_dict['attention_mask'].sum(axis=1).long().cpu() # attention_mask
        lang_feat = pack_padded_sequence(lang_embeds, lang_tokens_len, batch_first=True, enforce_sorted=False)
        _, lang_last = self.lang_gru(lang_feat) 
        lang_last = lang_last.sum(dim=0)
       
        # --- Geo Encoding ---
        geo_feats = data_dict['geo_feats']
        geo_len = data_dict["geo_len"] 
        geo_mask = data_dict['geo_mask'] # 
        geo_feats = sparse_collate(sum(geo_feats, [])).to(device)
        geo_feats = self.sparse_conv(geo_feats)
        # geo_feats: total_num_objects, feature_size
        geo_feats = self.pooling(geo_feats)  

        batch_size = len(geo_len)
        geo_feats_list = []
        cursor = 0
        for i in range(batch_size):
            num_obj = geo_len[i] 
            # geo_feat: num_obj, feature_size
            geo_feat = geo_feats[cursor:cursor + num_obj]
            geo_feats_list.append(geo_feat)
            cursor += num_obj
            
        # batch, max_num_object, feat_size
        geo_feats = pad_sequence(geo_feats_list, batch_first=True, padding_value=0.0)
        if geo_feats.shape[1] < self.max_num_object:
            geo_feats = torch.cat(
                [geo_feats, torch.zeros(geo_feats.shape[0], self.max_num_object - geo_feats.shape[1], geo_feats.shape[2]).to(geo_feats.device)], 1
            )        

        landmark_feats = None
        landmark_mask  = None
        if self.args.use_landmark:                
            # batch, num_landmarks, num_points
            landmark_feats = data_dict['landmark_feats']
            landmark_len = data_dict["landmark_len"]
            landmark_mask = data_dict['landmark_mask']

            landmark_feats = sparse_collate(sum(landmark_feats, [])).to(device)
            landmark_feats = self.sparse_conv(landmark_feats)
            # landmark_feats: total_num_landmarks (batch * num_landmarks), feature_size (128)
            landmark_feats = self.pooling(landmark_feats) # [16, 128] 16 = 2 * 8

            cursor = 0
            landmark_feats_list = []
            for num_landmark in data_dict['landmark_len'].long().cpu():                
                # geo_feat: num_obj, feature_size
                landmark_feat = landmark_feats[cursor:cursor + num_landmark]
                landmark_feats_list.append(landmark_feat)
                cursor += num_landmark
            # batch, max_num_landmark, feat_size
            landmark_feats = pad_sequence(landmark_feats_list, batch_first=True, padding_value=0.0)

            ### language module
            if self.args.use_landmark_name:
                data_dict = self.landmark_lang(data_dict)        
                landmark_name_feats = data_dict['landmark_name_feats']
                landmark_feats = torch.cat([landmark_feats, landmark_name_feats], 2)
                landmark_feats = self.landmark_concat(landmark_feats)
                
            if landmark_feats.shape[1] < self.args.max_num_landmark:
                 landmark_feats = torch.cat(
                    [landmark_feats, torch.zeros(batch_size, self.args.max_num_landmark - landmark_feats.shape[1], landmark_feats.shape[2]).to(landmark_feats.device)], 1
                )
            data_dict['landmark_feats'] = landmark_feats          
            # geo_feats: batch, max_num_object + max_num_land, hidden_size
            geo_feats = torch.cat([geo_feats, landmark_feats], dim=1)
        
        batch_size = lang_last.shape[0]
        # geo_feats: batch, max_num_object + max_num_landmark, hidden_size
        geo_feats = torch.cat([lang_last.reshape(batch_size, 1, -1).repeat(1, geo_feats.shape[1], 1), geo_feats], dim=2)
        geo_feats = self.fuse(geo_feats)
        
        # geo_output
        geo_output, _ = self.geo_gru(geo_feats)
        batch_size = lang_last.shape[0]
        object_conf_feat = geo_output[:, :self.max_num_object]
        
        # batch, max_num_objects
        data_dict["object_scores"] = self.object_cls(object_conf_feat).squeeze(-1) 
            
        if self.use_lang_classifier:
            # lang_scores: batch, obj_cat
            data_dict["lang_scores"] = self.lang_cls(lang_last)
            
        return data_dict
