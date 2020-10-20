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

# # Frechet mean
# import geomstats.backend as gs
# from geomstats.learning.frechet_mean import FrechetMean
# from geomstats.geometry.hypersphere import Hypersphere


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
    def __init__(self, nframes):
        super(FrameMeanCosineDistanceNetwork, self).__init__()
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

# class Fre(nn.Module):
#     def __init__(self, mean):
#         super(Fre, self).__init__()
#         self.mean = mean

#     def forward(self, X):
#         self.mean.fit(X)
#         return self.mean.estimate_.cuda()

class FrameFrechetMeanCosineDistanceNetwork(nn.Module):
    def __init__(self, nframes):
        super(FrameFrechetMeanCosineDistanceNetwork, self).__init__()
        self.nframes = nframes
        self.sphere = Hypersphere(dim=2047)  ###
        self.mean = FrechetMean(metric=self.sphere.metric)
        # self.frechet_mean = Fre(self.mean)

    def frechet_mean(self, X):
        """
        input:
        X: [n, n_features]
        mean: FrechetMean object
        output:
        x_bar: [n_features]
        """
        X=X.cpu()
        self.mean.fit(X)
        return self.mean.estimate_.cuda()

    def forward(self, support_vecs, target_vec):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """

        # print("Frechet!~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Reshape
        seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
        support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

        bs, df = target_vec.shape  # torch.Size([bs, 16384])
        target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

        # Normalize
        support_vecs = F.normalize(support_vecs, p=2, dim=-1)
        target_vec = F.normalize(target_vec, p=2, dim=-1)

        """
        # Frechet mean!
        support_vecs = support_vecs.view(seqlen*bs, self.nframes, -1).cuda()
        support_means = []
        for support_vec in support_vecs:
            support_means.append(self.frechet_mean(support_vec).cuda())
        support_vecs = torch.stack(support_means).cuda()  
        support_vecs = support_vecs.view(seqlen, bs, -1)  # torch.Size([seqlen, bs, 2048])
        assert support_vecs.shape == (seqlen, bs, 2048)

        target_means = []
        for target in target_vec:  # bs
            target_means.append(self.frechet_mean(target).cuda())
        target_means = torch.stack(target_means).cuda()  # torch.Size([bs, 2048])
        assert target_means.shape == (bs, 2048)

        target_vec = target_means

        # support_vecs = torch.mean(support_vecs, dim=-2)  # torch.Size([seqlen, bs, 2048])
        # target_vec = torch.mean(target_vec, dim=-2)  # torch.Size([bs, 2048])

        # Dot product
        similarities = []  # [nsupport * tensor(bs)]
        for support_vec in support_vecs:
            # [bs, d]->[bs,1,d] & [bs, d]->[bs,d,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")
            similarities.append(torch.bmm(support_vec.unsqueeze(1), target_vec.unsqueeze(2)).squeeze(1).squeeze(1))  

        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities

        """

        # Frechet mean!
        support_means = []
        for support_vec in support_vecs.view(seqlen*bs, self.nframes, -1):
            support_means.append(self.frechet_mean(support_vec))

        support_means = torch.stack(support_means).cuda()  
        support_means = support_means.view(seqlen, bs, -1)  # torch.Size([seqlen, bs, 2048])
        assert support_means.shape == (seqlen, bs, 2048)

        target_means = []
        for target in target_vec:  # bs
            target_means.append(self.frechet_mean(target))
        target_means = torch.stack(target_means).cuda()  # torch.Size([bs, 2048])
        assert target_means.shape == (bs, 2048)

        # Dot product
        similarities = []  # [nsupport * tensor(bs)]
        for support_mean in support_means:
            # [bs, d]->[bs,1,d] & [bs, d]->[bs,d,1] => [bs, 1, 1].squeeze(1).squeeze(1) => [bs]
            # print(target_vec.shape, "!!!!!!!!!!!!!!!!!!!!!!!")
            similarities.append(torch.bmm(support_mean.unsqueeze(1), target_means.unsqueeze(2)).squeeze(1).squeeze(1))  
        
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


        """
        # It is NOT using target/support _means, just for debug!!!!!!!!!!!
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
        """


def self_message_passing(vec_frames, depth=1):
    # expected size: [bs, nframes, dim_feature]
    # expect normalized features
    bs, nframes, dim_feature = vec_frames.shape

    for i in range(depth):
        # Normalize first!
        vec_frames = F.normalize(vec_frames, p=2, dim=-1)
        vf = vec_frames  # [bs,nf,df]
        fv = vec_frames.transpose(-1, -2)  # [bs,df,nf]
        D = torch.bmm(vf, fv)  # [bs,nf,nf]
        vec_frames = torch.bmm(D, vf)

    return vec_frames  # [bs,nf,df] unnormalized


class FrameSelfContextMeanNetwork(nn.Module):
    """
    message-passing among frames
    then mean pooling to get one feature vector for the video
    """
    def __init__(self, nframes):
        super(FrameSelfContextMeanNetwork, self).__init__()
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

        tv = self_message_passing(target_vec)  # still torch.Size([bs, 8, 2048])
        tv_mean = torch.sum(tv, dim=1)  # torch.Size([bs, 2048]) checked   ## caution: not normalized before mean operation!

        # Normalize again!!!
        tv_mean = F.normalize(tv_mean, p=2, dim=-1)

        similarities = []
        for support_vec in support_vecs:
            sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
            sv = self_message_passing(sv)
            sv_mean = torch.sum(sv, dim=1)  # torch.Size([bs, 2048])
            
            # Normalize again!!!
            sv_mean = F.normalize(sv_mean, p=2, dim=-1)

            # [bs,d]->[bs,1,d] & [bs,d]->[bs,d,1] => [bs,1,1].squeeze(1).squeeze(1) => [bs]
            similarities.append(torch.bmm(sv_mean.unsqueeze(1), tv_mean.unsqueeze(2)).squeeze(1).squeeze(1))  
            
        similarities = torch.stack(similarities)  # [nsupport, bs]
        return similarities


# class FrameSelfContextStraightAlignNetwork(nn.Module):
#     """
#     message-passing among frames
#     then mean pooling to get one feature vector for the video
#     """
#     def __init__(self, nframes):
#         super(FrameSelfContextStraightAlignNetwork, self).__init__()
#         self.nframes = nframes

#     def forward(self, support_vecs, target_vec):
#         """
#         Produces pdfs over the support set classes for the target set image.
#         :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, dim_feature]
#         :param target_vec: The embedding of the target image, tensor of shape [batch_size, dim_feature]
#         :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
#         """

#         # Reshape
#         seqlen, bs, df = support_vecs.shape  # torch.Size([5, bs, 16384])
#         support_vecs = support_vecs.view(seqlen, bs, self.nframes, -1)  # torch.Size([5, bs, 8, 2048])

#         bs, df = target_vec.shape  # torch.Size([bs, 16384])
#         target_vec = target_vec.view(bs, self.nframes, -1)  # torch.Size([bs, 8, 2048])

#         # Normalize
#         support_vecs = F.normalize(support_vecs, p=2, dim=-1)
#         target_vec = F.normalize(target_vec, p=2, dim=-1)

#         tv = self_message_passing(target_vec)  # still torch.Size([bs, 8, 2048])

#         # Normalize again!!!
#         tv = F.normalize(tv, p=2, dim=-1)

#         tv = tv.view(bs, -1)  # torch.Size([bs, 16384])

#         similarities = []
#         for support_vec in support_vecs:
#             sv = support_vec.view(target_vec.shape)  # torch.Size([bs, 8, 2048])
#             sv = self_message_passing(sv)
#             # Normalize again!!!
#             sv = F.normalize(sv, p=2, dim=-1)
            

#             # [bs,d]->[bs,1,d] & [bs,d]->[bs,d,1] => [bs,1,1].squeeze(1).squeeze(1) => [bs]
#             similarities.append(torch.bmm(sv_mean.unsqueeze(1), tv_mean.unsqueeze(2)).squeeze(1).squeeze(1))  
            
#         similarities = torch.stack(similarities)  # [nsupport, bs]
#         return similarities


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


if __name__ == '__main__':
    unittest.main()