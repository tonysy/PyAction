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
        :param support_vecs: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param target_vec: The embedding of the target image, tensor of shape [batch_size, 64]
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


if __name__ == '__main__':
    unittest.main()