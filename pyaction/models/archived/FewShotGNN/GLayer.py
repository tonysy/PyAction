import torch
import torch.nn as nn
import torch.nn.functional as F

"""
功能:
单层gnn

# W_id在外面生成比较省空间

non-linearity: Leaky-ReLU

"""

def Gmul(W, x):
    """
    可以看成分块矩阵运算
    Input:
    W is a tensor of size (bs, N, N, J)
    x is a tensor of size (bs, N, num_features)
    Output:
    x
    """
    N = W.size()[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)

    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    # 返回的每个信号由J部分concat而成，分别由J个算子求得，一共J*num_feature维
    return output 

class Wcompute(nn.Module):

    """
    Implement 2 graph intrinsic linear operators:
    1. RowSoftmax(MLP(distance)) default:distance=abs(diff),MLP=True
    2. Identity
    """
    # todo:
    
    # 计算相似度矩阵(mlp over abs diff)并返回相似度矩阵和一个identity拼成的矩阵

    # 这个模块因为它metric里有个mlp，所以需要给定mlp的size
    # 所以它需要提供的参数有，输入信号的维度input_features和一些变量来指定mlp hidden的size

    # 哎就是有N个维度为input_features的feature vector
    # 首先求pair-wise的差的绝对值，得到N*N个维度为input_features的vector
    # 然后把每个vector过一个mlp，就得到distance值，是一个scaler
    # 这里mlp的实现就写成N*N图上的1*1conv，两者等价
    # mlp的深度写死了,4层hidden,每层的维度就等于nf*ratio
    # input_features最终会压成num_operator，也就是1

    def __init__(self, dim_in, metric, metric_with_mlp=True, hidden_size=None, drop=False):  # num_operators=1, nf, ratio=[2,2,1,1], activation='softmax'
        super(Wcompute, self).__init__()
        self.metric = metric
        self.metric_with_mlp = metric_with_mlp
        
        if self.metric_with_mlp:
            # MLP of 4 hidden layers
            assert len(hidden_size) == 4

            # Layer-1
            self.conv2d_1 = nn.Conv2d(dim_in, hidden_size[0], 1, stride=1)
            self.bn_1 = nn.BatchNorm2d(hidden_size[0])
            self.drop = drop
            if self.drop:
                self.dropout = nn.Dropout(0.3)

            # Layer-2
            self.conv2d_2 = nn.Conv2d(hidden_size[0], hidden_size[1], 1, stride=1)
            self.bn_2 = nn.BatchNorm2d(hidden_size[1])
            
            # Layer-3
            self.conv2d_3 = nn.Conv2d(hidden_size[1], hidden_size[2], 1, stride=1)
            self.bn_3 = nn.BatchNorm2d(hidden_size[2])
            
            # Layer-4
            self.conv2d_4 = nn.Conv2d(hidden_size[2], hidden_size[3], 1, stride=1)
            self.bn_4 = nn.BatchNorm2d(hidden_size[3])
            
            # Layer-out
            self.conv2d_last = nn.Conv2d(hidden_size[3], 1, 1, stride=1)  # 所以最后会把input_features维压缩成一个scaler，即为距离


    def forward(self, x, W_id):

        # Compute raw distance
        W1 = x.unsqueeze(2)  # (bs, N, 1, dim_in)
        W2 = torch.transpose(W1, 1, 2)  # (bs, 1, N, dim_in)

        if self.metric == "ABS_DIFF":
            # Get pair-wise abs diff matrix
            W_new = torch.abs(W1 - W2) # (bs, N, N, dim_in)
        else:
            #################
        # Transpose for the conv afterwards
        W_new = torch.transpose(W_new, 1, 3)  # (bs, dim_in, N, N)

        # MLP for each x(by 1x1 conv)

        # Layer-1
        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        # Layer-2
        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        # Layer-3
        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        # Layer-4
        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        # Layer-out
        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3)  # (bs, N, N, 1)

        # Normalize each row as a stochastic kernel with softmax
        W_new = W_new - W_id.expand_as(W_new) * 1e8  # 把W_ii变成负无穷, softmax之后就是0
        W_new = torch.transpose(W_new, 2, 3)  # (bs, N, 1, N) 为了后面变成(bs*N, N)然后softmax
        W_new = W_new.contiguous()
        W_new_size = W_new.size()
        W_new = W_new.view(-1, W_new.size(3))
        W_new = F.softmax(W_new)
        W_new = W_new.view(W_new_size)
        W_new = torch.transpose(W_new, 2, 3)  # (bs, N, N, 1)

        # Concat the 2 matrixes
        W_new = torch.cat([W_id, W_new], 3)

        return W_new

class Gconv(nn.Module):

    """
    Input:
    W, x

    Output:
    W, x
    """

    # 注意:
    # 它这里会各个算子的信号concat在一起
    # 然后过一个linear变换到num_out维度

    # 每层Gconv，根据一般形式:
    # 有J个算子(graph intrinsic linear operators, 大小应该只能是N*N的)
    # Gconv的输出等于，每个算子乘在输入信号x左边，然后每个信号xi从nf_input维度线性变换到nf_output维度(线性变换个数和算子个数对应)


    def __init__(self, dim_in, dim_out, J, bn_bool=True):
        super(Gconv, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.J = J
        
        # self.num_inputs = J*nf_input
        # self.num_outputs = nf_output

        self.fc = nn.Linear(J*dim_in, dim_out)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, W, x):

        x = Gmul(W, x) # out has size (bs, N, J*dim_in)

        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.J * self.dim_in)
        x = self.fc(x) # has size (bs*N, dim_out)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.dim_out)
        return W, x


class GLayer(nn.Module):
    def __init__(self, dim_in, dim_out, metric='ABS_DIFF', metric_with_mlp=True, metric_mlp_hidden_size=None, is_out_layer=False):
        super(GLayer, self).__init__()
        self.module_w = Wcompute(dim_in, metric=metric, metric_with_mlp=metric_with_mlp, hidden_size=metric_mlp_hidden_size)
        self.module_gconv = Gconv(dim_in, dim_out, J=2, bn_bool=not is_out_layer)
        self.non_linear = F.leaky_relu
        self.is_out_layer = is_out_layer

    def forward(self, x, W_id):
        W = self.module_w(x, W_id)  # [batchsize, N, N, J] 前一个是identity; 后一个对角线为0，每行sum为1(但每列不是); checked
        x = self.module_gconv(W, x)[-1]  # gconv returns (W, x), so get the latter
        if not self.is_out_layer:
            x = self.non_linear(x)
        return x