import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np
import subprocess
from torch.utils.data import DataLoader
from datetime import datetime
from socket import gethostname

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from lib.dataset import ReferenceDataset 
from lib.solver import Solver
from lib.config import CONF
from models import get_tokenizer
from models.cityrefer import CityRefer
from models.refnet import RefNet


def get_commit_hash():
    #cmd = "git rev-parse --short HEAD"
    cmd = "git rev-parse HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash

print(sys.path, '<< sys path')

def get_dataloader(DC, CONF, scanrefer, all_scene_list, split, augment=False, shuffle=False, use_cache=False):
    dataset = ReferenceDataset(
        args,
        DC=DC,
        CONF=CONF,
        dataset=args.dataset,
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        augment=augment,
        use_cache=use_cache,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=dataset.collate_fn)
    return dataset, dataloader


def get_model(args, DC, CONF, tokenizer):
    # initiate model
    input_channels = int(args.use_color) * 3 + int(args.use_height) + 3 # xyz
    
    if args.model == 'cityrefer':
        model = CityRefer(
            args = args,
            input_feature_dim = input_channels,
            num_object_class = DC.num_class, # num_class
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
    
    # pre-trained model
    if args.pretrain:
         raise NotImplementedError

    model = model.cuda()
    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))
    return num_params


def get_solver(args, DC, CONF, dataloader):
    tokenizer = get_tokenizer(args)
    model = get_model(args, DC, CONF, tokenizer)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, CONF.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    solver = Solver(
        args,
        model=model,
        tokenizer=tokenizer,
        DC=DC,
        CONF=CONF,
        dataloader=dataloader,
        optimizer=optimizer,
        scaler=GradScaler(enabled=not args.no_amp),
        stamp=stamp,
        val_step=args.val_step,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        use_amp=(not args.no_amp)
    )
    num_params = get_num_params(model)
    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params
    
    info["git_commit_hash"] = get_commit_hash()
    info["hostname"] = gethostname()

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
        
    # save commandline 
    cmd = " ".join([v for v in sys.argv])
    cmd_file = os.path.join(root, "cmdline.txt")
    open(cmd_file, 'w').write(cmd)        


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, train_scenes_to_use=None, val_scenes_to_use=None):
    # get initial scene list
    if train_scenes_to_use is not None:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train if data["scene_id"] in train_scenes_to_use])))
    else:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        
    if val_scenes_to_use is not None:
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val if data["scene_id"] in val_scenes_to_use])))
    else:        
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
        
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = []
    for data in scanrefer_val:
        if data["scene_id"] in val_scene_list:
            new_scanrefer_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


def train(args):
    if args.dataset == 'sensaturban':
        from lib.config import CONF
        from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
        DC = SensatUrbanDatasetConfig()    
    else:
        raise NotImplementedError       
    CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, 'outputs', args.dataset, args.log_dir, 'checkpoints')

    # init training dataset
    print("preparing data...")

    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "cityrefer/meta_data", "CityRefer_train.json")))
    SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "cityrefer/meta_data", "CityRefer_val.json")))        
    
    scanrefer_train, scanrefer_val, all_scene_list= get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.train_scenes_to_use, args.val_scenes_to_use)
    scanrefer = {"train": scanrefer_train, "val": scanrefer_val}

    train_dataset, train_dataloader = get_dataloader(DC, CONF, scanrefer, all_scene_list, "train", augment=args.augment, shuffle=True, use_cache=(not args.no_cache))
    val_dataset, val_dataloader = get_dataloader(DC, CONF, scanrefer, all_scene_list, "val", augment=False, shuffle=False, use_cache=(not args.no_cache))
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, DC, CONF, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


def get_parser():
    parser = argparse.ArgumentParser(description='CityRefer')
    # General
    parser.add_argument("--model", type=str, help="", default="cityrefer")
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu index", default="0")
    parser.add_argument("--num_workers", type=int, help="num_workers", default=4)
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=20)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--optim", type=str, help="learning rate", default='adam')
    parser.add_argument("--no_amp", action='store_true', help="not use amp")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--drop_rate", type=float, help="weight decay", default=0.1)
    parser.add_argument("--num_points", type=int, default=-1, help="Point Number [default: 40000]")
    parser.add_argument("--num_inst_points", type=int, default=1024, help="Point Number of each instance [default: 1024]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument('--train_scenes_to_use', required=False, nargs="*", type=str, help='a list of scenes')
    parser.add_argument('--val_scenes_to_use', required=False, nargs="*", type=str, help='a list of scenes')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cache", action="store_true", help="Use cache data")
    # Data
    parser.add_argument("--dataset", type=str, help="refer dataset", default="sensaturban") # scannet, sensaturban, cityrefer
    parser.add_argument("--num_cands", type=int, help="number of object cadidates for grounding", default=10) # 
    parser.add_argument("--no_gt_instance", action="store_true", help="Not use GT instances")
    parser.add_argument("--augment", action="store_true", help="Do NOT use augment on trainingset (not used)")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_height", action="store_true", help="Use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_landmark", action="store_true", help="Use landmark in input.")    
    parser.add_argument("--use_landmark_name", action="store_true", help="Use landmark name in input.")    
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    # Tokenizer
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name", default="bert-base-uncased", 
            choices=("bert-base-uncased",)
    )
    # Model (CityRefer)
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="") # scannet: 0.02
    parser.add_argument("--voxel_size_ap", type=float, default=0.33, help="") 
    parser.add_argument("--voxel_size_glp", type=float, default=0.33, help="") 
    parser.add_argument("--features_dim", type=int, default=128, help="")
    parser.add_argument("--hidden_size", type=int, default=128, help="") 
    parser.add_argument("--max_desc_len", type=int, default=128, help="maximum number of tokens in the input text")
    parser.add_argument("--max_land_len", type=int, default=32, help="maximum number of tokens in the landmark names")
    parser.add_argument("--max_num_object", type=int, default=600, help="maximum number of objects")
    parser.add_argument("--max_num_landmark", type=int, default=192, help="maximum number of landmarks")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="")
    # Model (RefNet)
    parser.add_argument("--match_type", type=str, help="", default="ScanRefer")
    # Loss weight
    parser.add_argument("--ref_weight", type=float, default=10, help="")
    parser.add_argument("--mlm_weight", type=float, default=1, help="")
    parser.add_argument("--lang_weight", type=float, default=1, help="")
    # Config    
    parser.add_argument("--log_dir", type=str, default='test', help='path to log file')
    # Pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
   
    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
