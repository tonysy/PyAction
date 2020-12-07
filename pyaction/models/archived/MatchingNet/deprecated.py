def self_message_passing(self, vec_frames, depth=1):
    """
    version 1.0
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
    """

    # version 2.0

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
        tv_mean = torch.sum(tv,
                            dim=1)  # torch.Size([bs, 2048]) checked   ## caution: not normalized before mean operation!

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

# # Frechet mean
# import geomstats.backend as gs
# from geomstats.learning.frechet_mean import FrechetMean
# from geomstats.geometry.hypersphere import Hypersphere

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
        X = X.cpu()
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
        for support_vec in support_vecs.view(seqlen * bs, self.nframes, -1):
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