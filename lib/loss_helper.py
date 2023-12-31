import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box_batch, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape

        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=0)
        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=0).mean()

        return loss


class RankingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RankingLoss, self).__init__()
        self.m = 0.2
        self.gamma = 64
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, sim, label):
        loss_v = 0
        loss_l = 0
        loss_loc = 0
        batch_size = label.shape[0]
        delta_p = 1 - self.m
        delta_n = self.m

        for i in range(batch_size):
            temp_label = label[i]
            index = temp_label > 0.5
            index = index.nonzero().squeeze(1)
            if index.shape[0] > 0:
                pos_sim = torch.index_select(sim[i], 0, index)
                alpha_p = torch.clamp(0.8 - pos_sim.detach(), min=0)
                logit_p = - alpha_p * (pos_sim - delta_p) * self.gamma
            else:
                logit_p = torch.zeros(1)[0].cuda()

            index = (temp_label < 0.25)
            index = (index).nonzero().squeeze(1)

            neg_v_sim = torch.index_select(sim[i], 0, index)
            if neg_v_sim.shape[0] > 20:
                index = neg_v_sim.topk(10, largest=True)[1]
                neg_v_sim = torch.index_select(neg_v_sim, 0, index)

            alpha_n = torch.clamp(neg_v_sim.detach() - 0.2, min=0)
            logit_n = alpha_n * (neg_v_sim - delta_n) * self.gamma

            loss_loc += self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        if self.reduction == 'mean':
            loss = (loss_l + loss_v + loss_loc) / batch_size
        return loss


class SimCLRLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SimCLRLoss, self).__init__()
        self.m = 0.2
        self.gamma = 64
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, sim, label):
        sim = torch.exp(7 * sim)
        loss = - torch.log((sim * label).sum() / (sim.sum() - (sim * label).sum() + 1e-8))

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=5, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, score, label):
        score *= self.gamma
        sim = (score*label).sum()
        neg_sim = score*label.logical_not()
        neg_sim = torch.logsumexp(neg_sim, dim=0) # soft max
        loss = torch.clamp(neg_sim - sim + self.margin, min=0).sum()
        return loss


class SegLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(SegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt)**self.gamma) * self.alpha * logpt
        return loss



def compute_box_loss(data_dict, box_mask):
    """ Compute 3D bounding box loss.
    Args:
        data_dict: dict (read-only)
    Returns:
        center_loss
        size_reg_loss
    """

    # Compute center loss
    pred_center = data_dict['center']
    pred_size_residual = data_dict['size_residual']

    gt_center = data_dict['ref_center_label']
    gt_size_residual = data_dict['ref_size_residual_label']

    creterion = nn.SmoothL1Loss(reduction='none')
    center_loss = creterion(pred_center, gt_center)
    center_loss = (center_loss * box_mask.unsqueeze(1)).sum() / (box_mask.sum() + 1e-6)
    size_loss = creterion(pred_size_residual, gt_size_residual)
    size_loss = (size_loss * box_mask.unsqueeze(1)).sum() / (box_mask.sum() + 1e-6)

    return center_loss, size_loss


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss


def get_ref_loss(args, data_dict, config, reference=True, use_lang_classifier=False):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    # get ref gt
    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)
    ref_gt_bbox = get_3d_box_batch(ref_gt_obb[:, 3:6], ref_gt_obb[:, 6], ref_gt_obb[:, 0:3])

    object_scores = data_dict['object_scores']
    pred_obb_batch = data_dict['pred_obb_batch']
    batch_size = len(pred_obb_batch)
    cluster_label = []
    box_mask = torch.zeros(batch_size).cuda()

    # Reference loss
    criterion_ref = SoftmaxRankingLoss()
    ref_loss = torch.zeros(1).cuda().requires_grad_(True)

    for i in range(batch_size):
        pred_obb = pred_obb_batch[i]  # (num, 7) num < max_num_object
        num_filtered_obj = pred_obb.shape[0]

        if num_filtered_obj == 0:
            cluster_label.append([])
            box_mask[i] = 1
            continue

        label = np.zeros(num_filtered_obj)
        pred_bbox = get_3d_box_batch(pred_obb[:, 3:6], pred_obb[:, 6], pred_obb[:, 0:3])
        ious = box3d_iou_batch(pred_bbox, np.tile(ref_gt_bbox[i], (num_filtered_obj, 1, 1))) # num_obj
        label[ious.argmax()] = 1  # treat the bbox with highest iou score as the gt

        # label: num_filtered_obj
        label = torch.FloatTensor(label).cuda()
        cluster_label.append(label)
        if num_filtered_obj == 1: continue
        # object_scores: batch, max_num_obj
        score = object_scores[i][:num_filtered_obj]

        if ious.max() < 0.2: continue
        ref_loss = ref_loss + criterion_ref(score, label.clone())

    # ref_loss    
    ref_loss = ref_loss / batch_size
    data_dict['ref_loss'] = ref_loss
    data_dict['cluster_label'] = cluster_label
    # total loss
    data_dict['loss'] = args.ref_weight * ref_loss 
    
    if 'lang_scores' in data_dict:
        # language classification loss
        data_dict['lang_loss'] = compute_lang_classification_loss(data_dict)
        data_dict['loss'] += args.lang_weight * data_dict['lang_loss']    
    else:
        data_dict['lang_loss'] = torch.zeros(1).cuda().requires_grad_(True)
    
    if 'mlm_loss' in data_dict:
        data_dict['loss'] += args.mlm_weight * data_dict['mlm_loss'] 
    else:
        data_dict['mlm_loss'] = torch.zeros(1).cuda().requires_grad_(True)
        
    return data_dict


def get_loss(args, data_dict, config, reference=True, use_lang_classifier=False):
    if args.model == 'cityrefer':
        data_dict = get_ref_loss(args, data_dict, config, reference, use_lang_classifier)
    elif args.model == 'refnet':
        data_dict = get_ref_loss(args, data_dict, config, reference, use_lang_classifier)
    else:
        raise NotImplementedError    
    return data_dict
