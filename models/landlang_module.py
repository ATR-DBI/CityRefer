import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class LandLangModule(nn.Module):
    def __init__(self, num_object_class, vocab_size, use_lang_classifier=True, use_bidir=False, 
        embed_dim=256, hidden_size=256, max_num_landmark=128, padding_idx=0):
        super().__init__() 

        self.num_object_class = num_object_class
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.max_num_landmark = max_num_landmark
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx) #, **factory_kwargs)   
       
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size
        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_object_class),
                nn.Dropout()
            )

    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        input_ids = data_dict["landmark_tokens"]
        word_embs = self.word_embeddings(input_ids)
        landmark_tokens_len = data_dict['landmark_tokens_mask'].sum(axis=1).long().cpu()
        lang_feat = pack_padded_sequence(word_embs, landmark_tokens_len, batch_first=True, enforce_sorted=False)
    
        # encode description
        _, lang_last = self.gru(lang_feat)
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

        cursor = 0
        landmark_name_feats = []
        for num_landmark in data_dict['landmark_len'].long().cpu():
            landmark_name_feat = lang_last[cursor:cursor+num_landmark]
            landmark_name_feats.append(landmark_name_feat)
            cursor += num_landmark
        landmark_name_feats = pad_sequence(landmark_name_feats, batch_first=True)
        
        # store the encoded language features
        data_dict["landmark_name_feats"] = landmark_name_feats # B, max_landmark_len, hidden_size

        return data_dict

    def length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, "Length shape should be 1 dimensional."
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask
