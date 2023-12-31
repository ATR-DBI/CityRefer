"""Util"""
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
from typing import Tuple, List


class Util:

    @staticmethod
    def l1norm(X, dim, eps=1e-8):
        """L1-normalize columns of X
        """
        norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
        X = torch.div(X, norm)
        return X

    @staticmethod
    def l2norm(X, dim, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True)
        norm = torch.sqrt(norm + eps)
        X = torch.div(X, norm)
        return X

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    @staticmethod
    def inter_relation(K, Q, xlambda):
        """
        Q: (batch, queryL, d)
        K: (batch, sourceL, d)
        return (batch, queryL, sourceL)
        """
        batch_size, queryL = Q.size(0), Q.size(1)
        batch_size, sourceL = K.size(0), K.size(1)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        queryT = torch.transpose(Q, 1, 2)

        attn = torch.bmm(K, queryT)
        attn = nn.LeakyReLU(0.1)(attn)
        attn = Util.l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size * queryL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        return attn

    @staticmethod
    def intra_relation(K, Q, xlambda):
        """
        Q: (n_context, sourceL, d)
        K: (n_context, sourceL, d)
        return (n_context, sourceL, sourceL)
        """
        batch_size, sourceL = K.size(0), K.size(1)
        K = torch.transpose(K, 1, 2).contiguous()
        attn = torch.bmm(Q, K)

        attn = attn.view(batch_size * sourceL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        attn = attn.view(batch_size, sourceL, -1)
        return attn


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: Nx3 '''

    dropout_ratio = np.random.random() * max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

    if len(drop_idx) > 0:
        pc[drop_idx,:] = pc[0,:]  # set to the first point

    return pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high)
    batch_data *= scales

    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)

    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(float)
    batch_data += jittered_data

    return batch_data

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            NxC array
        Output:
            NxC array
    """
    idx = np.arange(batch_data.shape[0])
    np.random.shuffle(idx)
    return batch_data[idx,:]

def rotate_point_cloud_z(pc):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(pc, rotation_matrix)
    return rotated_data

def show_point_clouds(pts, out):
    fout = open(out, 'w')
    for i in range(pts.shape[0]):
        fout.write('v %f %f %f %d %d %d\n' % (
            pts[i, 0], pts[i, 1], pts[i, 2], 0, 255, 255))
    fout.close()


def tensor2points(tensor, offset=(-80., -80., -5.), voxel_size=(.05, .05, .1)):
    indices = tensor.float()
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, :3] = indices[:, :3] * voxel_size + offset + .5 * voxel_size
    return indices


# lengths = torch.Tensor([3, 5])
# max_length = 10
# get_mask(lengths, max_length)
# tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
def get_mask(lengths, max_length):
    """Computes a batch of padding masks given batched lengths"""
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask


def mask_tokens(inputs, tokenizer, mlm_probability):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def adjust_learning_rate(
    optimizer,
    curr_step: int,
    num_training_steps: int,
    args,
):
    num_warmup_steps: int = round(args.fraction_warmup_steps * num_training_steps)
    if args.schedule == "linear_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
    else:  # constant LR
        gamma = 1

    optimizer.param_groups[0]["lr"] = args.lr * gamma