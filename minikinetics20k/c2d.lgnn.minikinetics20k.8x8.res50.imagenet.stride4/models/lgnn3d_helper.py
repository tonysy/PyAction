#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
"""LatentGNN 3D for Video Recognition"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentGNN3D(nn.Module):
    """
    Build LatentGNN 3D as a generic family of building blocks for capturing long-range dependencies efficiently. LatentGNN3D computes the response at a position as a weighted sum of the features at all positions.
    The building block can be plugged into many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1905.11634
    """
    def __init__(
        self,
        dim,
        dim_inner,
        num_nodes,
        # pool_size,
        channel_stride = 2,
        mode='asymmetric',
        norm_func = F.normalize,
        latent_nonlocal = True,
        instantiation ='softmax',
        norm_type = 'batchnorm',
        latent_value_transform = True,
        latent_skip = True,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            num_nodes (int): number of latent nodes for LatentGNN
            mode (str): mode for projection and re-projection
                "asymmetric":
                "symmetric":
            norm_func (function): function for normalization to make the feature scale consistent.
        """
        super(LatentGNN3D, self).__init__()
        self.dim = dim
        self.dim_inner = dim // channel_stride
        self.mode = mode
        self.num_nodes = num_nodes
        # self.use_pool = (
        #     False if pool_size is None else any((size > 1 for size in pool_size))
        # )
        self.norm_func = norm_func
        self.latent_nonlocal = latent_nonlocal
        self.instantiation = instantiation
        self.norm_type = norm_type

        self.latent_value_transform = latent_value_transform if self.latent_nonlocal else False
        self.latent_skip = latent_skip if self.latent_nonlocal else False

        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum

        self._construct_lgnn(zero_init_final_norm)

    def _construct_lgnn(self, zero_init_final_norm):
        # Projection and Re-projection layer
        if self.mode == 'asymmetric':
            self.psi_v2l = nn.Conv3d(
                self.dim, self.num_nodes, kernel_size=1, stride=1, padding=0
            )
            self.psi_l2v = nn.Conv3d(
                self.dim, self.num_nodes, kernel_size=1, stride=1, padding=0
            )
        elif self.mode == 'symmetric':
            self.psi = nn.Conv3d(
                self.dim, self.num_nodes, kernel_size=1, stride=1, padding=0
            )
        else:
            raise NotImplementedError
        
        # Norm layer for the final output
        if self.norm_type == "batchnorm":
            self.norm_layer = nn.BatchNorm3d(
                self.dim, eps=self.norm_eps, momentum=self.norm_momentum
            )
            # Zero initializing the final bn.
            self.norm_layer.transform_final_bn = zero_init_final_norm
        elif self.norm_type == "layernorm":
            # In Caffe2 the LayerNorm op does not contain the scale an bias
            # terms described in the paper:
            # https://caffe2.ai/docs/operators-catalogue.html#layernorm
            # Builds LayerNorm as GroupNorm with one single group.
            # Setting Affine to false to align with Caffe2.
            self.norm_layer = nn.GroupNorm(1, self.dim, eps=self.norm_eps, affine=False)
        elif self.norm_type == "none":
            # Does not use any norm.
            pass
        else:
            raise NotImplementedError(
                "Norm type {} is not supported".format(self.norm_type)
            )

        # self.gamma = nn.Parameter(torch.zeros(1))

        # Final Conv 
        # self.conv_final = nn.Conv3d(
        #         self.dim, self.dim, kernel_size=1, stride=1, padding=0
        #     )
        # self.conv_final = nn.Sequential(
        #         nn.Conv3d(
        #             self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        #         ),
        #         nn.Conv3d(
        #             self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        #         )
        #     )

        # Latent Nonlocal operator
        if self.latent_nonlocal:
            
            self.conv_qk = nn.Conv1d(
                self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
            )
            # Whe
            if self.latent_value_transform:
                self.conv_v = nn.Conv1d(
                    self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
                )

                self.conv_out = nn.Conv1d(
                    self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
                )
    
    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()

        ##################################################
        # Step-1: Visible-to-Latent Space                #
        ##################################################

        # 1. Get the projection adjacency matrix
        if self.mode == 'asymmetric':
            # N, C, T, H, W --> N, D, T, H, W
            v2l_graph_adj = self.psi_v2l(x).view(N, self.num_nodes, -1)

            # N, C, T, H, W --> N, D, T, H, W
            l2v_graph_adj = self.psi_l2v(x).view(N, self.num_nodes, -1)

        elif self.mode == 'symmetric':
            # N, C, T, H, W --> N, D, T, H, W
            v2l_graph_adj = l2v_graph_adj = self.psi(x).view(N, self.num_nodes, -1)

        else:
            raise NotImplementedError
        
        # 2. Conduct projection operation
        v2l_graph_adj = F.softmax(v2l_graph_adj, dim=1)
        v2l_graph_adj_ = v2l_graph_adj / (1e-6 + v2l_graph_adj.sum(dim=2, keepdim=True))

        # (N, D, THW) * (N, C, THW) => (N, C, D)
        z = torch.einsum("ndt,nct->ncd", (v2l_graph_adj_, x.view(N,C,-1)))
        z = self.norm_func(z, dim=1) # L2-norm

        ##################################################
        # Step-2: Latent Space Message Propogation       #
        ##################################################
        if self.latent_nonlocal:
            # (N, C, D) => (N, C', D)
            q = k = self.conv_qk(z)
            # (N, C', D) * (N, C', D) => (N, D, D)
            adj_matrix = torch.einsum('nct,ncp->ntp', (q, k))
            
            if self.instantiation == 'softmax':
                # Normalizing the affinity tensors theta_phi before softmax
                adj_matrix = adj_matrix * (self.dim_inner ** -0.5)
                adj_matrix = F.softmax(adj_matrix, dim=2)
            elif self.instantiation == 'dot_product':
                adj_matrix = adj_matrix / self.num_nodes
            else:
                raise NotImplementedError("Unkown norm type {}".format(self.instantiation))
            
            if self.latent_value_transform:
                # (N, D, D) * (N, C', D) => (N, C', D)
                value = torch.einsum("ntp,ncp->nct", (adj_matrix, self.conv_v(z)))
                # (N, C', D) => (N, C, D)
                value = self.conv_out(value)
            else:
                # (N, D, D) * (N, C, D) => (N, C, D)
                value = torch.einsum("ntp,ncp->nct", (adj_matrix, z))

            if self.latent_skip:
                z = value + z
            else:
                z = value    

        ##################################################
        # Step-3: Visible-to-Latent Space                #
        ##################################################
        l2v_graph_adj = F.softmax(l2v_graph_adj, dim=1)

        # (N, C, D) * (N, D, THW)  => (N, C, THW)
        visible = torch.bmm(z,l2v_graph_adj)
        visible = visible.view(N, C, T, H, W)
        # visible = F.normalize(visible, dim=1)
        # visible = self.conv_final(visible)

        visible = self.norm_layer(visible)


        # out = x_identity + self.gamma * visible
        out = x_identity + visible

        # if H == 28:
        #     # import pdb;pdb.set_trace()
        #     print('X Norm Mean: {:4f}, Visible Norm Mean: {:4f}'.format(x_identity.norm(dim=1).mean() ,visible.norm(dim=1).mean()))
        #     print('X/Visible Norm Mean Ratio: {:4f}'.format(x_identity.norm(dim=1).mean()/visible.norm(dim=1).mean()))
        
        out = F.relu(out, inplace=True)

        return out


if __name__ == '__main__':
    latent_node = 10

    model = LatentGNN3D(
        dim=512,
        dim_inner=256,
        num_nodes=latent_node,
    )
    inputs = torch.randn(1, 512, 8, 28, 28)
    out = model(inputs)
    from fvcore.nn.flop_count import flop_count
    flop_dict = flop_count(model, (inputs,))
    print("LatentGNN3D Flops:{}G".format(sum(list(flop_dict.values()))))
    from lgnn_helper import LatentGNN
    latent_model = LatentGNN(
        in_channels=512,
        channel_stride=2,
        latent_dims=[latent_node],
        # num_kernels=[1]
    )

    inputs_lgnn = torch.randn(1, 512, 8, 28*28)
    flop_dict_lgnn = flop_count(latent_model, (inputs_lgnn,))
    print("LatentGNN Flops:{}G".format(sum(list(flop_dict_lgnn.values()))))

    from pyaction.models.nonlocal_helper import Nonlocal
    nonlocal_model = Nonlocal(
        dim=512,
        dim_inner=256,
        pool_size=[1, 2, 2],
    )
    flop_dict_nln = flop_count(nonlocal_model, (inputs,))
    print("Nonlocal Flops:{}G".format(sum(list(flop_dict_nln.values()))))

    
    print("Nonlocal: LatentGNN3D={}".format(
        sum(list(flop_dict_nln.values())) / sum(list(flop_dict.values())) ))
    
    print("Nonlocal: LatentGNN={}".format(
        sum(list(flop_dict_nln.values())) / sum(list(flop_dict_lgnn.values())) ))