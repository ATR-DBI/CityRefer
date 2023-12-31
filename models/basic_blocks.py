import math
import torch
import torch.nn as nn
import torchsparse.nn as spnn

from torchsparse import SparseTensor
#from torch_geometric.nn import MessagePassing, knn

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, transpose=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,
                        transposed=transpose),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SparseConvEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x


class BEVEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, ks=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.stage4(x)
        return x


def spcrop(inputs, loc_min, loc_max):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    valid_flag = ((coords[:, :3] >= loc_min) & (coords[:, :3] < loc_max)).all(-1)
    output_coords = coords[valid_flag]
    output_features = features[valid_flag]
    return SparseTensor(output_features, output_coords, cur_stride)


class SparseCrop(nn.Module):
    def __init__(self, loc_min, loc_max):
        super().__init__()
        self.loc_min = loc_min
        self.loc_max = loc_max

    def forward(self, inputs):
        return spcrop(inputs, self.loc_min, self.loc_max)


class ToDenseBEVConvolution(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape,
                 offset: list = [0, 0, 0],
                 z_dim: int = 1,
                 use_bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset = torch.cuda.IntTensor([list(offset) + [0]])
        self.z_dim = z_dim
        self.n_kernels = int(shape[self.z_dim])
        self.bev_dims = [i for i in range(3) if i != self.z_dim]
        self.bev_shape = shape[self.bev_dims]
        self.kernel = nn.Parameter(torch.zeros(self.n_kernels, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if use_bias else 0
        self.init_weight()

    def __repr__(self):
        return 'ToDenseBEVConvolution(in_channels=%d, out_channels=%d, n_kernels=%d)' % (
            self.in_channels,
            self.out_channels,
            self.n_kernels
        )

    def init_weight(self):
        std = 1. / math.sqrt(self.in_channels)
        self.kernel.data.uniform_(-std, std)

    def forward(self, inputs):
        features = inputs.F
        coords = inputs.C
        cur_stride = inputs.s

        kernels = torch.index_select(self.kernel, 0, coords[:, self.z_dim].long() // cur_stride)
        sparse_features = (features.unsqueeze(-1) * kernels).sum(1) + self.bias
        sparse_coords = (coords - self.offset).t()[[3] + self.bev_dims].long()
        sparse_coords[1:] = sparse_coords[1:] // cur_stride
        batch_size = sparse_coords[0].max().item() + 1
        sparse_coords = sparse_coords[0] * int(self.bev_shape.prod()) + sparse_coords[1] * int(self.bev_shape[1]) + \
                        sparse_coords[2]
        bev = torch.cuda.sparse.FloatTensor(
            sparse_coords.unsqueeze(0),
            sparse_features,
            torch.Size([batch_size * int(self.bev_shape.prod()), sparse_features.size(-1)]),
        ).to_dense()
        return bev.view(batch_size, *self.bev_shape, -1).permute(0, 3, 1, 2).contiguous()  # To BCHW


def tensor2points(tensor, offset=(-80., -80., -5.), voxel_size=(.05, .05, .1)):
    indices = tensor.float()
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, :3] = indices[:, :3] * voxel_size + offset + .5 * voxel_size
    return indices

