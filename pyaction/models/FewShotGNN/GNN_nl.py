import torch
import torch.nn as nn
import torch.nn.functional as F

from .GLayer import GLayer


class GNN_nl(nn.Module):
    """
    Multi-layer GNN(with 2 operators)
    # 然后这里要注意它多层Gconv的实现
    # 每次过完一层Gconv，它会把x_new信号concat在之前的信号后面!!!
    # 这里比较疑惑，如果不concat，直接x=x_new会怎样?
    
    activation: sigmoid
    """

    def __init__(self, dim_features):
        super(GNN_nl, self).__init__()
        self.dim_features = dim_features  # [d_in, d1, d2, .., d_out]  这里d_out要压到n_way作为logits,然后最后一层不加bn!!!
        self.num_layers = len(dim_features) - 1
        # self.activation = F.sigmoid

        # if args.dataset == 'mini_imagenet':
        #     self.num_layers = 2
        # else:
        #     self.num_layers = 2

        ##### test test #####
        ratio = [2,2,1,1]
        hidden_size = [r * dim_features[0] for r in ratio]

        cur_sum_dim = 0  # sum(dim(0,..,cur))
        for i in range(self.num_layers):
            cur_sum_dim += dim_features[i]
            G = GLayer(dim_in=cur_sum_dim, dim_out=dim_features[i+1], metric_hidden_size=hidden_size, is_out_layer=(i == self.num_layers-1))
            self.add_module('layer_{}'.format(i), G)

    def forward(self, x):
        """
        N = n_support+1
        x: tensor [batchsize, N, imagefeature_dim + nclass]
        """
        W_id = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)  # [batchsize, N, N, 1] checked
        W_id = W_id.cuda()

        for i in range(self.num_layers):
            x_new = self._modules['layer_{}'.format(i)](x, W_id)

            if i == self.num_layers-1:
                # output layer
                logits = x_new[:, -1, :].squeeze(-1)  # 这里取第-1因为我在fewshotGNN中把query样例放在了最后面  # [batchsize, nclass]
                # preds = self.activation(logits)
                # return preds
                return logits
            else:
                x = torch.cat([x, x_new], 2)
