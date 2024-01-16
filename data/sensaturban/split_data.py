# refered from https://github.com/meidachen/STPLS3D/blob/main/HAIS/data/prepare_data_inst_instance_stpls3d.py
import glob
import json
import math
import re, os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from tool import DataProcessing

random.seed(3824) 
np.random.seed(3824)

def splitPointCloud(cloud, size=50.0, stride=50):
    limitMax = np.amax(cloud[:, 0:3], axis=0)
    width = int(np.ceil((limitMax[0] - size) / stride)) + 1
    depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond = xcond & ycond
        block = cloud[cond, :]
        blocks.append(block)
    return blocks


def getFiles(files, fileSplit):
    res = []
    for filePath in files:
        name = os.path.basename(filePath)
        if name.strip('.ply') in fileSplit:
            res.append(filePath)
    return res


def dataAug(points, semanticKeep):
    angle = random.randint(1, 359)
    angleRadians = math.radians(angle)
    rotationMatrix = np.array([[math.cos(angleRadians), -math.sin(angleRadians), 0],
                               [math.sin(angleRadians),
                                math.cos(angleRadians), 0], [0, 0, 1]])
    points[:, :3] = points[:, :3].dot(rotationMatrix)
    pointsKept = points[np.in1d(points[:, 6], semanticKeep)]
    return pointsKept


def _read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file

def get_false_segments(false_seg_dir):
    false_segs = []
    print('Read false segments')
    for false_seg_file in glob.glob(os.path.join(false_seg_dir, '*.csv')):
        print('loading:', false_seg_file)
        false_seg_df = pd.read_csv(false_seg_file)
        false_segs.append(false_seg_df)
    print()

    false_seg_df = pd.concat(false_segs)
    #false_seg_df

    false_object_ids = set()
    for i, row in false_seg_df.iterrows():
        false_object_id = row.area+'_block_'+str(row.block_id)+'_'+str(row.object_id)
        false_object_ids.add(false_object_id)    
    return false_object_ids


def preparePthFiles(args, files, split, outPutFolder, false_segments, AugTimes=0, crop_size=50):
    # save the coordinates so that we can merge the data to a single scene
    # after segmentation for visualization
    outJsonPath = os.path.join(outPutFolder, 'coordShift.json')
    coordShift = {}
    coordShift['globalShift'] = {}
    
    # used to increase z range if it is smaller than this,
    # over come the issue where spconv may crash for voxlization.
    zThreshold = 6

    # {0: 'Ground',
    # 1: 'High Vegetation',
    # 2: 'Buildings',
    # 3: 'Walls',
    # 4: 'Bridge',
    # 5: 'Parking',
    # 6: 'Rail',
    # 7: 'traffic Roads',
    # 8: 'Street Furniture',
    # 9: 'Cars',
    # 10: 'Footpath',
    # 11: 'Bikes',
    # 12: 'Water'}

    # Map relevant classes to {1,...,14}, and ignored classes to -100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
        remapper[x] = i

    # Instance to use
    # [0:'Ground', 2:'Building', 4:'Bridge', 5:'Parking', 9:'Car']        

    # Map instance to -100 based on selected semantic
    # (change a semantic to -100 if you want to ignore it for instance)
    remapper_disableInstanceBySemantic = np.ones(150) * (-100)
    #for i, x in enumerate([0, -100, 2, -100, 4, 5, -100, -100, -100, 9, -100, -100, -100]):
    for i, x in enumerate([0, -100, 2, -100, -100, 5, -100, -100, -100, 9, -100, -100, -100]):
        remapper_disableInstanceBySemantic[x] = i

    # only augment data for these classes
    #semanticKeep = [0, 2, 3, 7, 8, 9, 12, 13]
    #semanticKeep = [0, 2, 4, 5, 9]
    semanticKeep = [0, 2, 5, 9]
    
    counter = 0
    for file in files:
       
        print('loading:', file)
        seg_file = re.sub('.ply', '.segs.json', file)
        scene_id = os.path.basename(seg_file).rstrip('.segs.json')

        #
        # Read ply file
        #
        xyz, rgb, labels = DataProcessing.read_ply_data(file)    

        #
        # Add instance label
        #
        instance_db = _read_json(seg_file)
        labels = labels[:, np.newaxis]
        empty_instance_label = np.full(labels.shape, -100)
        labels = np.hstack((labels, empty_instance_label))
        for instance in instance_db["segGroups"]:
            occupied_indices = instance['pointIds']

            if scene_id+"_"+str(instance["id"]) in false_segments:
                print('false segments:', scene_id+"_"+str(instance["id"]))
                continue
            labels[occupied_indices, 1] = int(instance["id"])

        for AugTime in range(AugTimes + 1):
            #
            # Sampling
            if args.sample_type == 'grid':
                raise NotImplemented
                #sub_xyz, sub_rgb, sub_labels = DataProcessing.grid_sub_sampling(xyz, rgb, labels.astype(np.int32), args.grid_size)
            else:
                sub_xyz, sub_rgb, sub_labels = DataProcessing.random_sub_sampling(xyz, rgb, labels, args.random_sample_ratio)
            #
            # Concat
            #
            points = np.hstack((sub_xyz, sub_rgb, sub_labels)) 
            if split != 'test' and AugTime == 0:                    
                coordShift['globalShift'][os.path.basename(file).rstrip('.ply')] = list(points[:, :3].min(0))
            points[:, :3] = points[:, :3] - points[:, :3].min(0)

            #   
            # Augmentaion
            #
            if AugTime != 0:
                points = dataAug(points, semanticKeep)
            name = os.path.basename(file).strip('.ply') + '_%d' % AugTime            
           
            blocks = splitPointCloud(points, size=crop_size, stride=crop_size)
            for blockNum, block in enumerate(blocks):
                if (len(block) > 10000):
                    outFilePath = os.path.join(outPutFolder, name + str(blockNum) + '_inst_nostuff.pth')
                    outLabelPath = os.path.join(outPutFolder, name + str(blockNum) + '_inst_label.pth')
                    print('processing::', outFilePath)
                    
                    if (block[:, 2].max(0) - block[:, 2].min(0) < zThreshold):
                        block = np.append(
                            block, [[
                                block[:, 0].mean(0), block[:, 1].mean(0), block[:, 2].max(0) +
                                (zThreshold -
                                 (block[:, 2].max(0) - block[:, 2].min(0))), block[:, 3].mean(0),
                                block[:, 4].mean(0), block[:, 5].mean(0), -100, -100
                            ]],
                            axis=0)
                        print('range z is smaller than threshold ')
                        print(name + str(blockNum) + '_inst_nostuff')
                    if split != 'test' and AugTime == 0:
                        outFileName = name + str(blockNum) + '_inst_nostuff'
                        coordShift[outFileName] = list(block[:, :3].mean(0))
                    coords = np.ascontiguousarray(block[:, :3] - block[:, :3].mean(0))

                    # coords = block[:, :3]
                    colors = np.ascontiguousarray(block[:, 3:6]) / 127.5 - 1

                    coords = np.float32(coords)
                    colors = np.float32(colors)
                    if split != 'test':
                        sem_labels = np.ascontiguousarray(block[:, -2])
                        sem_labels = sem_labels.astype(np.int32)
                        # original
                        orig_sem_labels = sem_labels.copy()
                        sem_labels = remapper[np.array(sem_labels)]

                        instance_labels = np.ascontiguousarray(block[:, -1]) 
                        instance_labels = instance_labels.astype(np.float32)
                        orig_instance_labels = instance_labels.copy().astype(np.int32)
                        disableInstanceBySemantic_labels = np.ascontiguousarray(block[:, -2])
                        disableInstanceBySemantic_labels = disableInstanceBySemantic_labels.astype(np.int32)
                        disableInstanceBySemantic_labels = remapper_disableInstanceBySemantic[np.array(disableInstanceBySemantic_labels)]
                        instance_labels = np.where(disableInstanceBySemantic_labels == -100, -100, instance_labels)

                        # map instance from 0.
                        # [1:] because there are -100
                        uniqueInstances = (np.unique(instance_labels))[1:].astype(np.int32)
                        remapper_instance = np.ones(50000) * (-100)
                        for i, j in enumerate(uniqueInstances):
                            remapper_instance[j] = i

                        instance_labels = remapper_instance[instance_labels.astype(np.int32)]

                        uniqueSemantics = (np.unique(sem_labels))[1:].astype(np.int32)

                        if split == 'train' and (len(uniqueInstances) < 10 or
                                                 (len(uniqueSemantics) >=
                                                  (len(uniqueInstances) - 2))):
                            print('unique insance: %d' % len(uniqueInstances))
                            print('unique semantic: %d' % len(uniqueSemantics))
                            print()
                            counter += 1
                        else:
                            torch.save((coords, colors, sem_labels, instance_labels), outFilePath)
                            torch.save((orig_sem_labels, orig_instance_labels), outLabelPath)
                    else:
                        torch.save((coords, colors), outFilePath)
        print()
        
    print('Total skipped file :%d' % counter)
    json.dump(coordShift, open(outJsonPath, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="scans")    
    parser.add_argument("--out_dir", type=str, default=".")    
    parser.add_argument("--split_data", type=str, default="meta_data/balance_split")
    parser.add_argument("--false_seg_dir", type=str, default="meta_data/false_segments")
    parser.add_argument("--sample_type", type=str, default="random")
    parser.add_argument("--grid_size", type=float, default=0.2)
    parser.add_argument("--random_sample_ratio", type=int, default=10)
    parser.add_argument("--train_crop_size", type=int, default=50)
    parser.add_argument("--val_crop_size", type=int, default=250)
    parser.add_argument("--aug_times", type=int, default=6)
    args = parser.parse_args()

    filesOri = sorted(glob.glob(args.data_dir + '/*/*.ply'))
    split_dir = os.path.basename(args.split_data)

    if args.sample_type == "grid":
        out_dir = os.path.join(args.out_dir, split_dir, (args.sample_type + '-{:.3f}'.format(args.grid_size)) + '_crop-'+str(args.train_crop_size))
    else:
        out_dir = os.path.join(args.out_dir, split_dir, (args.sample_type + '-{:d}'.format(args.random_sample_ratio)) + '_crop-'+str(args.train_crop_size))
        
    trainSplit = [line.strip() for line in open(os.path.join(args.split_data, 'sensaturban_train.txt')).readlines()]    
    false_segments = get_false_segments(args.false_seg_dir)

    # val with val_crop
    valSplit = [line.strip() for line in open(os.path.join(args.split_data, 'sensaturban_val_val.txt')).readlines()]        
    split = 'val_val_'+str(args.val_crop_size)+'m'
    valFiles = getFiles(filesOri, valSplit)
    valOutDir = os.path.join(out_dir, split)
    print(valOutDir)
    os.makedirs(valOutDir, exist_ok=True)
    preparePthFiles(args, valFiles, split, valOutDir, false_segments, crop_size=args.val_crop_size)

    # test with val_crop
    testSplit = [line.strip() for line in open(os.path.join(args.split_data, 'sensaturban_val_test.txt')).readlines()]        
    split = 'val_test_'+str(args.val_crop_size)+'m'
    testFiles = getFiles(filesOri, testSplit)
    testOutDir = os.path.join(out_dir, split)
    os.makedirs(testOutDir, exist_ok=True)
    preparePthFiles(args, testFiles, split, testOutDir, false_segments, crop_size=args.val_crop_size)

    # train with val_crop
    trainFiles = getFiles(filesOri, trainSplit)
    split = 'train_'+str(args.val_crop_size)+'m'
    trainOutDir = os.path.join(out_dir, split)
    os.makedirs(trainOutDir, exist_ok=True)
    preparePthFiles(args, trainFiles, split, trainOutDir, false_segments, crop_size=args.train_crop_size)    

