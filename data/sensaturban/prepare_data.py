#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import glob
import json
import torch
import numpy as np
import pandas as pd
import multiprocessing


def read_mask(filename):
    #print('\t\tLoading:', os.path.basename(filename))
    insMask = pd.read_csv(filename, delimiter=',', header=None).values
    insMask = np.squeeze(insMask, axis=1)    
    return filename, insMask

    
def read_masks(insMaskFilePathList, num_processes=1):
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        results = pool.map(read_mask, insMaskFilePathList)
        pool.close()
        pool.join()
        return {filename:insMask for filename, insMask in results}
    else:
        mask_dic = {}
        for insMaskFile in insMaskFilePathList:
            mask_dic[insMaskFile] = read_mask(insMaskFile)[-1]
        return mask_dic


def get_shifted_xyz(split_fname, insDir, result_dir, coordShift, insLabel=1, num_processes=1):
    insFilePath = os.path.join(insDir, split_fname + '_inst_nostuff.pth')
    insLabelPath = os.path.join(insDir, split_fname + '_inst_label.pth')    
    #
    semanticPredDir = os.path.join(result_dir, 'semantic_pred')
    insMaskDir = os.path.join(result_dir, 'pred_instance/predicted_masks')    
    # coords/birmingham_block_11_00.npy
    xyz, color, sem_labels, instance_labels = torch.load(insFilePath)
    
    # gt semantic labels & instance labels
    print('\tloading:', insLabelPath)
    label_ids, instance_ids = torch.load(insLabelPath)
    label_ids = label_ids.astype(np.int32)
    instance_ids = instance_ids.astype(np.int32) # -100: unannotated 
    
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
    
    insMaskFilePathList = sorted(glob.glob(insMaskDir + '/%s*.txt' % split_fname))
    # predicted instance ids
    instance_ids_pg = np.zeros(len(xyz), dtype=np.int32)  # 0: unannotated    
    mask_dic = read_masks(insMaskFilePathList, num_processes)
    
    # predicted semantic labels
    label_ids_pg = np.zeros(len(xyz), dtype=np.int32)
    pred_inst_dir = os.path.join(result_dir, 'pred_instance')
    pred_inst_file = os.path.join(pred_inst_dir, split_fname + '.txt')
    sem_lab_dic = {os.path.basename(mask_fname):(int(sem_lab)-1) for mask_fname, sem_lab, conf in [inst.split() for inst in pd.read_table(pred_inst_file, header=None)[0]]}
    
    for insMaskPath in insMaskFilePathList:
        insMask = mask_dic[insMaskPath]
        instance_ids_pg[insMask != 0] = insLabel
        label_ids_pg[insMask != 0] = sem_lab_dic[os.path.basename(insMaskPath)]
        insLabel+=1
    
    xyz = xyz + np.array([float(value) for value in coordShift[split_fname+'_inst_nostuff']])
    return xyz, color, label_ids, instance_ids, label_ids_pg, instance_ids_pg, insLabel


def handle_process(block_fname, insntance_dir, result_dir, coordShift, semantic_keep, objectId_to_objectName, output_dir, num_processes=1):
    xyz_list = []
    color_list = []
    label_ids_list = []    
    instance_ids_list = [] 
    label_ids_pg_list = [] 
    instance_ids_pg_list = []
    
    insLabel = 1
    for instance_file in glob.glob(os.path.join(insntance_dir, block_fname+'_*_inst_nostuff.pth')):
        split_fname = os.path.basename(instance_file).rstrip('_inst_nostuff.pth')
        print('\tsplit:', split_fname)        
        xyz, color, label_ids, instance_ids, label_ids_pg, instance_ids_pg, insLabel = get_shifted_xyz(split_fname, insntance_dir, result_dir, coordShift, insLabel, num_processes)
        xyz_list.append(xyz)
        color_list.append(color)
        label_ids_list.append(label_ids)
        instance_ids_list.append(instance_ids)
        label_ids_pg_list.append(label_ids_pg)
        instance_ids_pg_list.append(instance_ids_pg)
        
    coords = np.concatenate(xyz_list)
    globalShift = coordShift['globalShift'][block_fname]
    
    colors = np.concatenate(color_list)
    label_ids = np.concatenate(label_ids_list)
    all_instance_ids = np.concatenate(instance_ids_list)
    label_ids_pg = np.concatenate(label_ids_pg_list)
    instance_ids_pg = np.concatenate(instance_ids_pg_list)
    instance_bboxes = {}
    # Normalize xyz coords
    colors = (colors + 1) * 127.5
    colors = colors.astype(np.uint8)

    all_instance_id_to_label_id = {instance_id:label_id for instance_id, label_id in zip(all_instance_ids, label_ids)}
    instance_ids = all_instance_ids.copy()
    instance_ids[~np.isin(label_ids, semantic_keep)] = -100
    
    landmark_names = {}
    for instance_id in list(set(all_instance_ids)): 
        if instance_id in objectId_to_objectName:
            landmark_names[instance_id] = objectId_to_objectName[instance_id]
            
    landmark_ids = all_instance_ids.copy()
    landmark_ids[~np.isin(all_instance_ids, list(landmark_names.keys()))] = -100

    for instance_id in list(set(instance_ids)):
        instance_id = int(instance_id)
        if instance_id == -100: # no annotation
            continue
        # semantic label
        label_id = all_instance_id_to_label_id[instance_id] 

        obj_pc = coords[instance_ids == instance_id, 0:3]
        if len(obj_pc) == 0:
            continue
            
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin,
            label_id, instance_id])  # also include object id
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        
        instance_bboxes[instance_id] = bbox

    output_file = os.path.join(output_dir, block_fname + '.pth')
    print('\tsaving:', output_file)
    torch.save((coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, landmark_names, landmark_ids, globalShift), output_file)
    

def _read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file


def get_object_id_to_name(bbox_dir):
    objectId_to_objectName_dic = {}
    print('Preparing object_id_to_name dictionary...')
    for bbox_file in glob.glob(os.path.join(bbox_dir, '*_bbox.json')):
        bbox_json = _read_json(bbox_file)
        print('\tloading:', os.path.basename(bbox_file))
        # int -> str
        objectId_to_objectName_dic[bbox_json['sceneId']] = {bbox['id']:bbox['object_name'] for bbox in bbox_json['bboxes'] if len(bbox['object_name']) > 0}
    print()
    return objectId_to_objectName_dic        

#
# Arguments
#
def parse_args():
    parser = argparse.ArgumentParser('Data Preparision')
    parser.add_argument('--split_type', type=str, default='balance_split')
    parser.add_argument('--data_type', type=str, default='random-50_crop-50')
    parser.add_argument('--eval_type', type=str, default='val_val_250m')
    parser.add_argument('--root_dir', type=str, default='.')    
    parser.add_argument('--semantic_keep', nargs='+', default=[0, 2, 5, 9], help='semantic keep ids')
    parser.add_argument('--bbox_dir', type=str, default='../cityrefer/box3d/')
    parser.add_argument('--output_dir', type=str, default='pointgroup_data')    
    parser.add_argument('--num_processes', type=int, default=20)
    return parser.parse_args()

#
# Main
#
if __name__ == '__main__':
    args = parse_args()

    split_type = args.split_type
    data_type = args.data_type
    eval_type = args.eval_type
    root_dir = args.root_dir
    semantic_keep = args.semantic_keep
    bbox_dir = args.bbox_dir
    output_dir = args.output_dir
    num_processes = args.num_processes
    
    insntance_dir = os.path.join(root_dir, split_type, data_type, eval_type)
    coordShift_file = os.path.join(insntance_dir, 'coordShift.json')
    coordShift = json.load(open(coordShift_file))
    result_dir = os.path.join(root_dir, 'results', split_type, data_type, eval_type)
    output_dir = os.path.join(args.output_dir, split_type, data_type)
    
    os.makedirs(output_dir, exist_ok=True)
    block_fnames = list(set([os.path.basename(tmpFile).rsplit('_', 3)[0] for tmpFile in glob.glob(os.path.join(insntance_dir, '*.pth'))]))

    objectId_to_objectName_dic = get_object_id_to_name(bbox_dir)
    for i, block_fname in enumerate(block_fnames):
        print(f"block {i+1} / {len(block_fnames)} : {block_fname}")
        objectId_to_objectName = objectId_to_objectName_dic[block_fname]
        handle_process(block_fname, insntance_dir, result_dir, coordShift, semantic_keep, objectId_to_objectName, output_dir, num_processes)
    print()
