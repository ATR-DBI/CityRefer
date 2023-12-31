import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchsparse.nn as spnn
from torchsparse.utils.collate import sparse_collate

from models.transformer import MultiHeadAttention
#from models.lang_module import LangModule

from models.basic_blocks import SparseConvEncoder
from models.landlang_module import LandLangModule


class RefNet(nn.Module):
    def __init__(self, args, input_feature_dim=0, num_object_class=None, vocab_size=None, pad_token_id=0):
        super().__init__()
        self.args = args
        self.num_object_class = num_object_class
        self.match_type = args.match_type
        self.num_proposal = args.max_num_object if args.num_cands < 0 else args.num_cands # self.max_num_object

        self.use_lang_classifier=(not args.no_lang_cls)
        self.drop_rate = args.drop_rate        
        
        hidden_size = args.hidden_size

        # --------- Point Encoder ---------
        # Sparse Volumetric Backbone
        self.sparse_conv = SparseConvEncoder(input_feature_dim) # self.input_feature_dim = 3 -> 128
        self.pooling = spnn.GlobalMaxPool()        

        # --------- Language Encoder ---------
        embed_dim = args.hidden_size  
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id) #, **factory_kwargs)
        self.lang_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=args.num_hidden_layers,
            batch_first=True,
            bidirectional=True, 
        )        
        
        # --------- Landmark Encoder ---------
        if self.args.use_landmark_name:
            self.landmark_lang = LandLangModule(num_object_class, vocab_size, True, args.use_bidir, hidden_size, hidden_size, args.max_num_landmark, pad_token_id) 
            self.landmark_concat = nn.Linear(hidden_size * 2, hidden_size)                

        # --------- Proposal Matching ---------
        # Match the generated proposals and select the most confident ones
        if self.match_type == "ScanRefer":
            # # ********* ScanRefer Matching *********
            self.match = MatchModule(args)
        elif self.match_type == "Transformer":
            # # ********* Transformer Matching *********
            self.match = TransformerMatchModule(args) 
        else:
            raise NotImplementedError("Matching type not supported.")
        
        # --------- Language Classifier ---------
        if self.use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),                
                nn.Dropout(self.drop_rate), #p=args.lang_pdrop),
                nn.Linear(hidden_size, num_object_class),
            )
        
        
    def lang_encode(self, data_dict):
        # --- Language Encoding ---
        input_ids = data_dict['input_ids']
        input_shape = input_ids.size()  
        # see https://github.com/daveredrum/D3Net/blob/main/model/lang_module.py                
        # lang_embeds: batch, num_words (seq_len), hidden (emb_size)
        lang_embeds = self.word_embeddings(input_ids)
        lang_tokens_len = data_dict['attention_mask'].sum(axis=1).long() # attention_mask
        lang_feat = pack_padded_sequence(lang_embeds, lang_tokens_len.cpu(), batch_first=True, enforce_sorted=False)
        lang_hiddens, lang_last = self.lang_gru(lang_feat)
        lang_hiddens, _ = pad_packed_sequence(lang_hiddens, batch_first=True)

        # lang_hiddens: batch, num_words, hidden_size    
        lang_hiddens = (lang_hiddens[:, :, :int(lang_hiddens.shape[-1] / 2)] + lang_hiddens[:, :, int(lang_hiddens.shape[-1] / 2):]) / 2
        lang_last = lang_last.sum(dim=0)
        
        data_dict['lang_last'] = lang_last
        data_dict['lang_hiddens'] = lang_hiddens
        
        # lang_last: batch, hidden_size
        lang_last = data_dict['lang_last'] 
        # lang_feat: batch, num_proposal, hidden_size
        lang_feat = lang_last.unsqueeze(1).repeat(1, self.num_proposal, 1)        
        data_dict['lang_feat'] = lang_feat
        
        # sentence mask
        seq_len = lang_embeds.shape[1]
        lengths = lang_tokens_len.unsqueeze(1).repeat(1, seq_len) # batch_size, seq_len
        idx = torch.arange(0, seq_len).unsqueeze(0).repeat(lengths.shape[0], 1).type_as(lengths).long() # batch_size, seq_len
        lang_masks = (idx < lengths).float() # batch_size, seq_len        
        data_dict['lang_masks'] = ~lang_masks.bool() 

        return data_dict
    
                
    def object_encode(self, data_dict):
        device = data_dict['input_ids'].device
        # --- Geo ---
        geo_feats = data_dict['geo_feats'] 
        geo_coords = geo_feats # batch, geo_len (num_points, xyz)
        geo_len = data_dict["geo_len"] 
        geo_feats = sparse_collate(sum(geo_feats, [])).to(device)
        geo_feats = self.sparse_conv(geo_feats)
        # geo_feats: num_proposal, feature_size
        geo_feats = self.pooling(geo_feats)  

        batch_size = len(geo_len)
        geo_feats_list = []
        geo_center_list = []
        cursor = 0
        for i in range(batch_size):
            num_obj = geo_len[i] 
            # geo_feat: num_obj, feature_size
            geo_feat = geo_feats[cursor:cursor + num_obj]
            # geo_center: num_obj, 3 (xyz)
            geo_center = torch.stack([geo_coord.coords.float().mean(dim=0) for geo_coord in geo_coords[i]]) if num_obj != 0 else torch.zeros(0, 3)
            geo_feats_list.append(geo_feat)
            geo_center_list.append(geo_center)
            cursor += num_obj
            
        # geo_feats: batch, num_proposal, feat_size
        geo_feats = pad_sequence(geo_feats_list, batch_first=True, padding_value=0.0)
        # geo_centers: batch, num_proposal, 3
        geo_centers = pad_sequence(geo_center_list, batch_first=True, padding_value=0.0)
        geo_centers = geo_centers.to(device)
        
        if geo_feats.shape[1] < self.num_proposal:
            geo_feats = torch.cat(
                [geo_feats, torch.zeros(geo_feats.shape[0], self.num_proposal - geo_feats.shape[1], geo_feats.shape[2]).to(geo_feats.device)], 1
            )
            geo_centers = torch.cat(
                [geo_centers, torch.zeros(geo_centers.shape[0], self.num_proposal - geo_centers.shape[1], geo_centers.shape[2]).to(geo_centers.device)], 1
            )            

        data_dict['geo_feats'] = geo_feats
        data_dict['geo_centers'] = geo_centers

        return data_dict


    def forward(self, data_dict):
        #######################################
        #                                     #
        #           POINT BRANCH              #
        #                                     #
        #######################################
        data_dict = self.object_encode(data_dict)

        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang_encode(data_dict)

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################

        # --------- PROPOSAL MATCHING ---------
        data_dict = self.match(data_dict)
        
        if self.use_lang_classifier:
            # lang_scores: batch, obj_cat
            data_dict["lang_scores"] = self.lang_cls(data_dict['lang_last'])        

        return data_dict

       
class MatchModule(nn.Module):
    def __init__(self, args):
        super().__init__() 
        #self.num_proposal = cfg.model.max_num_proposal
        self.num_proposal = args.max_num_object if args.num_cands < 0 else args.num_cands
        #self.lang_size = lang_size
        hidden_size = args.hidden_size
        lang_size = hidden_size
        #self.det_channel = cfg.model.m
        self.det_channel = hidden_size
          
        self.fuse = nn.Sequential(
            nn.Conv1d(lang_size + self.det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        
        # self.match = nn.Conv1d(hidden_size, 1, 1)
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # features: batch, num_proposal, feat_size
        features = data_dict['geo_feats']
        # lang_feat: 
        lang_feat = data_dict['lang_feat']

        # fuse
        features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposal, feat_size + lang_size
        features = features.permute(0, 2, 1).contiguous() # batch_size , feat_size + lang_size, num_proposal
        
        # fuse features
        features = self.fuse(features) # batch_size, hidden_size, num_proposal

        # match
        confidences = self.match(features).squeeze(1) # batch_size, num_proposal
        data_dict["object_scores"] = confidences # batch_size, num_proposal
        
        return data_dict


class TransformerMatchModule(nn.Module):
    def __init__(self, args, head=4, depth=2, use_dist_weight_matrix=True):
        super().__init__()

        self.use_dist_weight_matrix = use_dist_weight_matrix
        self.num_proposal = args.max_num_object if args.num_cands < 0 else args.num_cands
        hidden_size = args.hidden_size
        lang_size = hidden_size
        self.hidden_size = hidden_size
        self.head = head
        self.depth = depth - 1
        self.det_channel = hidden_size
        self.chunk_size = 1

        self.features_concat = nn.Sequential(
            nn.Conv1d(self.det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, 1, 1)
        )

        self.lang_fc = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(hidden_size)
        )
        self.lang_self_attn = MultiHeadAttention(d_model=hidden_size, d_k=16, d_v=16, h=head)

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))  # k, q, v

    def multiplex_attention(self, v_features, l_features, l_masks, dist_weights, attention_matrix_way):
        batch_size, num_words, _ = l_features.shape

        lang_self_masks = l_masks.reshape(batch_size, 1, 1, -1).contiguous().repeat(1, self.head, num_words, 1)
        l_features = self.lang_fc(l_features)
        l_features = self.lang_self_attn(l_features, l_features, l_features, lang_self_masks)
        lang_cross_masks = l_masks.reshape(batch_size, 1, 1, -1).contiguous().repeat(1, self.head, self.num_proposal, 1)
        v_features = self.cross_attn[0](v_features, l_features, l_features, lang_cross_masks)

        for _ in range(self.depth):
            v_features = self.self_attn[_+1](v_features, v_features, v_features, attention_weights=dist_weights, way=attention_matrix_way)
            v_features = self.cross_attn[_+1](v_features, l_features, l_features, lang_cross_masks)

        # match
        v_features_agg = v_features.permute(0, 2, 1).contiguous()
        confidence = self.match(v_features_agg).squeeze(1)  # batch_size, num_proposal

        return confidence
    

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.use_dist_weight_matrix:
            # Attention Weight
            objects_center = data_dict["geo_centers"]

            N_K = objects_center.shape[1]
            center_A = objects_center[:, None, :, :].repeat(1, N_K, 1, 1)
            center_B = objects_center[:, :, None, :].repeat(1, 1, N_K, 1)
            dist = (center_A - center_B).pow(2)
            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]
            dist_weights = 1 / (dist+1e-2)
            norm = torch.sum(dist_weights, dim=2, keepdim=True)
            dist_weights = dist_weights / norm

            dist_weights = torch.cat([dist_weights for _ in range(self.head)], dim=1).detach()
            attention_matrix_way = "add"
        else:
            dist_weights = None
            attention_matrix_way = "mul"

        # object size embedding
        # features: batch, num_proposal, feat_size
        features = data_dict["geo_feats"]
        # B, N = features.shape[:2]
        features = features.permute(0, 2, 1)
        features = self.features_concat(features).permute(0, 2, 1)
        batch_size, num_proposal = features.shape[:2]
        
        objectness_masks = data_dict['geo_mask'][:, :, None]
        features = self.self_attn[0](features, features, features, attention_weights=dist_weights, way=attention_matrix_way)
        len_nun_max = self.chunk_size
        data_dict["random"] = random.random()

        # copy paste
        feature0 = features.clone()
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposal
            obj_lens = torch.zeros(batch_size).type_as(feature0).int()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]
        
        feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(-1, num_proposal, self.hidden_size)
        if dist_weights is not None:
            dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(-1, self.head, num_proposal, num_proposal)

        # v_features: batch, num_proposal, hidden_size
        v_features = feature1

        # l_features: batch, num_words, hidden_size    
        l_features = data_dict["lang_hiddens"]
        # l_masks: batch_size, num_words   
        l_masks = data_dict["lang_masks"]
        cluster_ref = self.multiplex_attention(v_features, l_features, l_masks, dist_weights, attention_matrix_way)
        
        data_dict["object_scores"] = cluster_ref # batch, num_objs
        return data_dict
