import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.dataset import ReferenceDataset
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval

from models import get_tokenizer
from models.cityrefer import CityRefer
from models.util import mask_tokens
from models.util import get_mask
from models.refnet import RefNet


def overwrite_config(args, past_args):
    for k, v in past_args.items():
        if hasattr(args, k): # skip if args has past_args
            continue
        setattr(args, k, v)
    return args   


def get_dataloader(args, DC, CONF, scanrefer, all_scene_list, split, use_cache=False):
    dataset = ReferenceDataset(
        args,
        DC=DC,
        CONF=CONF,
        dataset=args.dataset,
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_cache=use_cache,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataset, dataloader


def get_model(args, DC, CONF, tokenizer):
    # load model
    input_channels = int(args.use_color) * 3 + int(args.use_height) + 3 # xyz

    if args.model == 'cityrefer':    
        model = CityRefer(
            args = args,
            input_feature_dim = input_channels,
            num_object_class = DC.num_class,
            vocab_size = tokenizer.vocab_size,
            pad_token_id = tokenizer.pad_token_id,            
        )
    elif args.model == 'refnet':
        model = RefNet(
            args = args,
            input_feature_dim = input_channels,
            num_object_class = DC.num_class, # num_class
            vocab_size = tokenizer.vocab_size,
            pad_token_id = tokenizer.pad_token_id,
        )        
    else:
        raise NotImplementedError

    model_name = "model.pth"
    path = os.path.join(args.folder, model_name)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda()

    return model


def get_scanrefer(scanrefer, scenes_to_use=None):
    if scenes_to_use is not None:
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer if data["scene_id"] in scenes_to_use])))
    else:        
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))    
    new_scanrefer = []
    for data in scanrefer:
        if data["scene_id"] in scene_list:
            new_scanrefer.append(data)        
    assert len(new_scanrefer) != 0
    return new_scanrefer, scene_list


def eval_ref(args, CONF, DC):
    print("evaluate...")
    if args.eval_split == 'train':
        SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "cityrefer/meta_data", "CityRefer_train.json")))  
    elif args.eval_split == 'val':
        SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "cityrefer/meta_data", "CityRefer_val.json")))  
    elif args.eval_split == 'test':
        SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "cityrefer/meta_data", "CityRefer_test.json")))  
    else:
        raise NotImplementedError
    
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(SCANREFER, args.scenes_to_use)

    # dataloader
    _, dataloader = get_dataloader(args, DC, CONF, scanrefer, scene_list, args.eval_split, use_cache=(not args.no_cache))

    # model
    tokenizer = get_tokenizer(args)
    model = get_model(args, DC, CONF, tokenizer)

    # random seeds
    seeds = [args.seed + i * 100 for i in range(args.repeat)] 

    # evaluate
    print("evaluating...")
    score_path = os.path.join(args.folder, "scores."+args.eval_split+".pkl")
    pred_path = os.path.join(args.folder, "predictions."+args.eval_split+".pkl")        

    if not os.path.exists(score_path) or args.force:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        predictions_all = {}

        for seed in seeds:
            print('seed', seed)
            # for reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            for data_dict in tqdm(dataloader):
                for key in data_dict:
                    if key in ['object_cat', 'lidar', 'point_min', 'point_max', 'mlm_label',
                            'ref_center_label', 'ref_size_residual_label']:
                        data_dict[key] = data_dict[key].cuda()

                # text encoding
                query = data_dict["query"]
                encoded_query = tokenizer(
                    query,
                    add_special_tokens=True,
                    max_length=args.max_desc_len,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                )
                
                inputs = encoded_query["input_ids"]
                data_dict['input_ids'] = inputs.cuda() #to(device)
                data_dict['labels'] = None
                data_dict['attention_mask'] = encoded_query["attention_mask"].cuda() #to(device) 
                data_dict['geo_mask'] = get_mask(data_dict['geo_len'], args.max_num_object).cuda() #to(device) 
                
                # landmark
                if 'landmark_len' in data_dict:
                    landmark_texts = sum(data_dict["landmark_texts"], [])
                    encoded_landmark = tokenizer(
                        landmark_texts,
                        add_special_tokens=True,
                        max_length=args.max_land_len,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                    )
                    data_dict['landmark_tokens'] = encoded_landmark["input_ids"].cuda() 
                    data_dict['landmark_tokens_mask'] = encoded_landmark["attention_mask"].cuda() #to(device) 
                    data_dict['landmark_mask'] = get_mask(data_dict['landmark_len'], args.max_num_landmark).cuda() # to(device)
                    
                # feed
                with torch.amp.autocast(device_type='cuda', enabled=(not args.no_amp)):
                    data_dict = model(data_dict)
                    data_dict = get_loss(
                        args=args,
                        data_dict=data_dict,
                        config=DC,
                    )

                data_dict = get_eval(
                    args=args,
                    data_dict=data_dict,
                    config=DC,
                )

                ref_acc += data_dict["ref_acc"]
                ious += data_dict["ref_iou"]
                masks += data_dict["ref_multiple_mask"]
                others += data_dict["ref_others_mask"]
                lang_acc.append(data_dict["lang_acc"].item())

                # store predictions
                ids = data_dict["scan_idx"].detach().cpu().numpy()
                for i in range(ids.shape[0]):
                    idx = ids[i]
                    scene_id = scanrefer[idx]["scene_id"]
                    object_id = scanrefer[idx]["object_id"]
                    ann_id = scanrefer[idx]["ann_id"]

                    #'''
                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if object_id not in predictions[scene_id]:
                        predictions[scene_id][object_id] = {}

                    if ann_id not in predictions[scene_id][object_id]:
                        predictions[scene_id][object_id][ann_id] = {}

                    predictions[scene_id][object_id][ann_id]["pred_bbox"] = data_dict["pred_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["gt_bbox"] = data_dict["gt_bboxes"][i]
                    predictions[scene_id][object_id][ann_id]["iou"] = data_dict["ref_iou"][i]

            predictions_all[seed] = predictions

            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        with open(pred_path, "wb") as f:
            pickle.dump(predictions_all, f)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)

    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(
                    ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]),
                    ious[i] >= 0.25)].shape[0] \
                                     / ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                              others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(
                    np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                                    / ious[i][np.logical_and(masks[i] == multiple_dict[k],
                                                             others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                                 / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(
                masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                                 / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(
                others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {:.4f}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {:.4f}".format(np.mean(lang_acc)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str, help="Folder containing the model") # e.g., outputs/sensaturban/test/checkpoints/2023-05-30_13-07-29/
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument('--scenes_to_use', required=False, nargs="*", type=str, help='a list of scenes')
    parser.add_argument('--eval_split', type=str, required=True, default='test')
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="random seed")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.") 
    args = parser.parse_args()
    
    # overwrite
    train_args = json.load(open(os.path.join(args.folder, "info.json")))
    args = overwrite_config(args, train_args)      

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"    
    
    # get dataset config
    if args.dataset == 'sensaturban':
        from lib.config import CONF
        from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
        DC = SensatUrbanDatasetConfig()    
    else:
        raise NotImplementedError       
    CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, 'outputs', args.dataset, args.log_dir, 'checkpoints')    

    # evaluate
    eval_ref(args, CONF, DC)
