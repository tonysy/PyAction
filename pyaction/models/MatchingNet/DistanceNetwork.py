##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDistanceNetwork(nn.Module):
    def __init__(self):
        super(CosineDistanceNetwork, self).__init__()

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # eps = 1e-10
        # similarities = []
        # for support_image in support_set:
        #     sum_support = torch.sum(torch.pow(support_image, 2), 1)
        #     support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
        #     dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
        #     cosine_similarity = dot_product * support_magnitude
        #     similarities.append(cosine_similarity)
        # similarities = torch.stack(similarities)
        
        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=2)
        target_vec = F.normalize(target_vec, p=2, dim=1)

        # Dot product
        similarities = []  # [nsupport * tensor(bs)]
        for support_vec in support_vecs:
            # [bs, 64]->[bs,1,64] & [bs, 64]->[bs,64,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            similarities.append(torch.bmm(support_vec.unsqueeze(1), target_vec.unsqueeze(2)).squeeze(1).squeeze(1))  

        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class EuclideanDistanceNetwork(nn.Module):
    def __init__(self):
        super(EuclideanDistanceNetwork, self).__init__()

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        similarities = []
        for support_vec in support_vecs:
            similarities.append(torch.pow(support_vec-target_vec, 2).sum(1))

        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameMaxCosineDistanceNetwork(nn.Module):
    """
    Distance of video A and B: max(cos(frame_A_i, frame_B_j))
    """
    def __init__(self, nframes):
        super(FrameMaxCosineDistanceNetwork, self).__init__()
        self.nframes = nframes

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        similarities = []
        for support_vec in support_vecs:
            sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
            vt = target_vec.transpose(-1, -2)  # torch.Size([bs, 2048, 8])
            D = torch.bmm(sv, vt)  # S*Q  torch.Size([bs, 8, 8])
            d_max = torch.max(D.view(bs, -1), dim=-1).values  # torch.Size([bs]) checked
            similarities.append(d_max)  # torch.Size([bs])
            
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameStraightAlignCosineDistanceNetwork(nn.Module):
    def __init__(self, nframes):
        super(FrameStraightAlignCosineDistanceNetwork, self).__init__()
        self.nframes = nframes

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        # Recover shape
        support_vecs = support_vecs.view(seqlen, bs, -1)  # torch.Size([5, bs, 16384])
        target_vec = target_vec.view(bs, -1)  # torch.Size([bs, 16384])

        # Dot product
        similarities = []  # [nsupport * tensor(bs)]
        for support_vec in support_vecs:
            # [bs, dim]->[bs,1,dim] & [bs, dim]->[bs,dim,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            similarities.append(torch.bmm(support_vec.unsqueeze(1), target_vec.unsqueeze(2)).squeeze(1).squeeze(1))  

        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameGreedyAlignCosineDistanceNetwork(nn.Module):
    def __init__(self, nframes):
        super(FrameGreedyAlignCosineDistanceNetwork, self).__init__()
        self.nframes = nframes

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        similarities = []
        for support_vec in support_vecs:
            sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
            vt = target_vec.transpose(-1, -2)  # torch.Size([bs, 2048, 8])
            D = torch.bmm(sv, vt)  # S*Q  torch.Size([bs, 8, 8])
            
            # 每行取最大值和每列取最大值，两者其实不等价
            # 这里以query的视角为准(i.e. 每列取最大值)，这样5个D都是以D的视角，貌似比较可比
            # x = torch.max(D, 1)  # checked
            similarities.append(torch.max(D, 1)[0].sum(1))  # torch.Size([bs])
            
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameCosineDistanceMeanNetwork(nn.Module):
    def __init__(self, nframes):
        super(FrameCosineDistanceMeanNetwork, self).__init__()
        self.nframes = nframes

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        similarities = []
        for support_vec in support_vecs:
            sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
            vt = target_vec.transpose(-1, -2)  # torch.Size([bs, 2048, 8])
            D = torch.bmm(sv, vt)  # S*Q  torch.Size([bs, 8, 8])
            d_mean = torch.mean(D.view(bs, -1), dim=-1)  # torch.Size([bs]) checked
            similarities.append(d_mean)  # torch.Size([bs])
            
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameCosineDistanceSumNetwork(nn.Module):
    def __init__(self, nframes):
        super(FrameCosineDistanceSumNetwork, self).__init__()
        self.nframes = nframes

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        similarities = []
        for support_vec in support_vecs:
            sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
            vt = target_vec.transpose(-1, -2)  # torch.Size([bs, 2048, 8])
            D = torch.bmm(sv, vt)  # S*Q  torch.Size([bs, 8, 8])
            d_sum = torch.sum(D.view(bs, -1), dim=-1)  # torch.Size([bs]) checked
            similarities.append(d_sum)  # torch.Size([bs])
            
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameMeanCosineDistanceNetwork(nn.Module):
    def __init__(self, nframes, norm=None):
        super(FrameMeanCosineDistanceNetwork, self).__init__()
        self.nframes = nframes
        self.norm = norm if norm is not None else True

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        if self.norm:
            # Normalize
            support_vecs = F.normalize(support_vecs, p=2, dim=-1)
            target_vec = F.normalize(target_vec, p=2, dim=-1)
            
        # print(support_vecs.shape, "!!-------!!!!!")
        # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")

        # Fuse!
        support_vecs = torch.mean(support_vecs, dim=-2)  # torch.Size([seqlen, bs, 2048])
        target_vec = torch.mean(target_vec, dim=-2)  # torch.Size([bs, 2048])

        # Dot product
        similarities = []  # [nsupport * tensor(bs)]
        for support_vec in support_vecs:
            # [bs, d]->[bs,1,d] & [bs, d]->[bs,d,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")
            similarities.append(torch.bmm(support_vec.unsqueeze(1), target_vec.unsqueeze(2)).squeeze(1).squeeze(1))  

        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class FrameOTAMDistanceNetwork(nn.Module):
    def __init__(self, nframes, lam=None, ndirection=None):
        super(FrameOTAMDistanceNetwork, self).__init__()
        self.nframes = nframes
        self.lam = lam if lam else 0.1  # lambda
        self.ndirection = ndirection if ndirection else 2  # 1 or 2

    def dp(self, D):
        """
        D: [bs, nframe, nframe] frame cos distance matrix of a pair of videos
        version: 2.0
        """

        bs, n, m = D.shape
        assert n == m == self.nframes

        """
        DP = torch.zeros(bs, n, n+1)
                
        assert DP[:,0,1:].shape == D[:,0].shape
        DP[:,0,1:] = D[:,0]  # row 0

        if self.ndirection == -1:
            import pdb; pdb.set_trace()

        for i in range(1, n):
            d = torch.stack((DP[:,i-1,:-1]/self.lam, DP[:,i-1,1:]/self.lam), dim=-1)
            assert d.shape == (bs, n, 2)
            DP[:,i,1:] = self.lam * torch.logsumexp(d, dim=-1)

            if self.ndirection == -1:
                import pdb; pdb.set_trace()

        # the last column
        d = DP[:,:,-1]
        assert d.shape == (bs, n)
        # maximum of the last column
        ret = self.lam * torch.logsumexp(d/self.lam, dim=-1)
        assert ret.shape == (bs,)  # (bs) false
        
        if self.ndirection == -1:
            import pdb; pdb.set_trace()

        return ret.cuda()
        """

        DP = torch.zeros(bs, n, n).cuda()

        # col 0
        DP[:,:,0] = D[:,:,0]

        # row 0
        for i in range(1, n):
            DP[:,0,i] = DP[:,0,i-1] + D[:,0,i]

        if self.ndirection == -1:
            import pdb; pdb.set_trace()

        for i in range(1, n):
            d = torch.stack((DP[:,:-1,i-1]/self.lam, DP[:,1:,i-1]/self.lam), dim=-1)
            assert d.shape == (bs, n-1, 2)
            DP[:,1:,i] = self.lam * torch.logsumexp(d, dim=-1) + D[:,1:,i]  # forgot to add D[:,1:,i]

            if self.ndirection == -1:
                import pdb; pdb.set_trace()

        # the last column
        d = DP[:,:,-1]
        assert d.shape == (bs, n)

        # maximum of the last column
        ret = self.lam * torch.logsumexp(d/self.lam, dim=-1)
        assert ret.shape == (bs,)  # (bs) false

        if self.ndirection == -1:
            import pdb; pdb.set_trace()

        return ret.cuda()

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        similarities = []
        for support_vec in support_vecs:
            sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
            vt = target_vec.transpose(-1, -2)  # torch.Size([bs, 2048, 8])
            D = torch.bmm(sv, vt)  # S*Q  torch.Size([bs, 8, 8])
            if self.ndirection == 2:
                similarity = (self.dp(D) + self.dp(D.transpose(1,2))) / 2
            else:  # 1
                similarity = self.dp(D)
            similarities.append(similarity)  # torch.Size([bs])
            
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class TemporalGNN(nn.Module):
    """
    message-passing among frames
    act on a single video!!
    """

    def __init__(self, nframes, cos_scaler=None):
        super(TemporalGNN, self).__init__()
        self.nframes = nframes
        self.ln = nn.Linear(2048,2048)

        self.softmax = nn.Softmax(dim=-1)
        self.cos_scaler = cos_scaler if cos_scaler else 32 #########################
        print("TemporalGNN!!!!!!!!!!!!")

    def forward(self, vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        bs, df = vec.shape  # torch.Size([bs, 16384])
        frames = vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])
        frames = F.normalize(frames, p=2, dim=-1)

        # get affinity mat D
        vf = frames
        fv = frames.transpose(-1, -2)  # [bs,df,nf]
        D = torch.bmm(vf, fv)  # [bs,nf,nf]

        #####################################
        # version 2.0

        # 1. let diagonal of D be zeros
        D.diagonal(dim1=-2, dim2=-1).zero_()

        # 2. normalize rows of D by softmax
        assert self.cos_scaler > 0
        D *= self.cos_scaler
        D = self.softmax(D)
        D = D.cuda()

        #####################################

        # add
        frames += F.leaky_relu(self.ln(torch.bmm(D, frames)))

        vec = frames.view(bs, -1)
        return vec


class FrameMeanLearnableDistanceNetwork(nn.Module):
    def __init__(self, nframes, num_hidden=None, dim_hidden=None):
        super(FrameMeanLearnableDistanceNetwork, self).__init__()
        self.nframes = nframes
        self.num_hidden = num_hidden if num_hidden else 1
        self.dim_hidden = dim_hidden if dim_hidden else 2730  # 2/3 of dim_input(2048+2048)

        assert self.num_hidden == 1 ###
        self.linear1 = nn.Linear(in_features=4096, out_features=self.dim_hidden)
        self.bn1 = nn.BatchNorm1d(num_features=self.dim_hidden)
        self.linear2 = nn.Linear(in_features=self.dim_hidden, out_features=1)

        print("FrameMeanLearnableDistance!!!!!!!!!!!!!!!")

    def mlp_distance(self, sv, qv):
        # sv(qv): (bs, 2048)
        input = torch.cat((sv, qv), dim=-1)
        assert input.shape[0] == sv.shape[0] == qv.shape[0] and input.shape[-1] == 4096

        hidden1 = F.relu(self.bn1(self.linear1(input)))
        output = self.linear2(hidden1)
        return output

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        # print(support_vecs.shape, "!!-------!!!!!")
        # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")

        # Fuse!
        support_vecs = torch.mean(support_vecs, dim=-2)  # torch.Size([seqlen, bs, 2048])
        target_vec = torch.mean(target_vec, dim=-2)  # torch.Size([bs, 2048])

        # Learnable distance
        similarities = []  # [nsupport * tensor(bs)]
        for support_vec in support_vecs:
            # [bs, d]->[bs,1,d] & [bs, d]->[bs,d,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")
            similarities.append(self.mlp_distance(support_vec, target_vec).squeeze())

        similarities = torch.stack(similarities)  # [nsupport, bs]
        assert similarities.shape == (5, bs)
        return similarities


class FrameMeanMeanCosineDistanceNetwork(nn.Module):
    def __init__(self, nframes):
        super(FrameMeanMeanCosineDistanceNetwork, self).__init__()
        self.nframes = nframes

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # single vec size: (N, T * H * W * C)

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 8*7*7*2048])
        assert df == self.nframes*7*7*2048
        support_vecs = support_vecs.view(seqlen, bs, 7*7*self.nframes, -1)  # torch.Size([5, bs, 8*7*7, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 8*7*7*2048])
        target_vec = target_vec.view(bs, 7*7*self.nframes, -1)  # torch.Size([bs, 8*7*7, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        # print(support_vecs.shape, "!!-------!!!!!")
        # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")

        # Fuse!
        support_vecs = torch.mean(support_vecs, dim=-2)  # torch.Size([seqlen, bs, 2048])
        target_vec = torch.mean(target_vec, dim=-2)  # torch.Size([bs, 2048])

        # Dot product
        similarities = []  # [nsupport * tensor(bs)]
        for support_vec in support_vecs:
            # [bs, d]->[bs,1,d] & [bs, d]->[bs,d,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")
            similarities.append(torch.bmm(support_vec.unsqueeze(1), target_vec.unsqueeze(2)).squeeze(1).squeeze(1))

        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


class MHATT_temporal(nn.Module):
    """
    multi-head self-attention among frame features
    act on a single video!!

    with simple position encoding!!!
    """

    def __init__(self, nframes, nhead, pre_norm=None):
        super(MHATT_temporal, self).__init__()
        self.nframes = nframes
        self.nhead = nhead
        self.pre_norm = True if pre_norm else False  # Default: do not normalize input features

        # self.softmax = nn.Softmax(dim=-1)
        self.self_attn = nn.MultiheadAttention(2048+8, nhead)

    def forward(self, vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # Get frame features
        bs, df = vec.shape  # torch.Size([bs, 16384])
        frames = vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Pre-Normalize
        if self.pre_norm:
            frames = F.normalize(frames, p=2, dim=-1)

        # Calc position encoding
        pos = torch.linspace(-1, 1, steps=self.nframes).repeat(bs,1).unsqueeze(-1).repeat(1,1,8)  # [-1,1] # .repeat(bs, 1)
        assert pos.shape == (bs, self.nframes, 8)

        # Concat position to the feature
        frames = torch.cat((frames, pos.cuda()), -1)
        assert frames.shape == (bs, self.nframes, 2048+8)

        # Transpose for nn.mhatten
        frames = frames.transpose(0, 1)
        assert frames.shape == (self.nframes, bs, 2048+8)

        ### TODO: QKV with mlp

        # clone needed?
        q = frames.clone()
        k = frames.clone()
        v = frames

        # self-attention
        frames = self.self_attn(q, k, value=v)[0].transpose(0, 1).reshape(bs, -1)
        assert frames.shape == (bs, self.nframes*2056)

        ### TODO: Implementation of Feedforward model: ffn ###

        return frames


if __name__ == '__main__':
    unittest.main()