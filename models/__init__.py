#from .flashbert import BertForPreTraining

from transformers import (
    BertTokenizer,
    BertConfig,
)

def build_model(args):
    if "bert" in args.model_name:
        # if args.scratch:
        #     config = DebertaV2Config.from_pretrained(
        #         pretrained_model_name_or_path=args.model_name, local_files_only=True
        #     )
        #     model = DebertaV2ForMaskedLM(
        #         features_dim=args.features_dim if args.use_video else 0,
        #         max_feats=args.max_feats,
        #         freeze_lm=args.freeze_lm,
        #         freeze_mlm=args.freeze_mlm,
        #         ft_ln=args.ft_ln,
        #         ds_factor_attn=args.ds_factor_attn,
        #         ds_factor_ff=args.ds_factor_ff,
        #         dropout=args.dropout,
        #         n_ans=args.n_ans,
        #         freeze_last=args.freeze_last,
        #         config=config,
        #     )
        # else:
        #     model = DebertaV2ForMaskedLM.from_pretrained(
        #         features_dim=args.features_dim if args.use_video else 0,
        #         max_feats=args.max_feats,
        #         freeze_lm=args.freeze_lm,
        #         freeze_mlm=args.freeze_mlm,
        #         ft_ln=args.ft_ln,
        #         ds_factor_attn=args.ds_factor_attn,
        #         ds_factor_ff=args.ds_factor_ff,
        #         dropout=args.dropout,
        #         n_ans=args.n_ans,
        #         freeze_last=args.freeze_last,
        #         pretrained_model_name_or_path=args.model_name,
        #         local_files_only=True,
        #     )
        pass
            

def get_tokenizer(args):
    if "bert" in args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name, local_files_only=True
        )
    else:
        raise NotImplementedError
    return tokenizer