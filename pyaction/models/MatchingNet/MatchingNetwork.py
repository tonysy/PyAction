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
import numpy as np

#from Classifier import Classifier
from pyaction.models.video_model_builder import ResNetModel
from .BidirectionalLSTM import BidirectionalLSTM
from .DistanceNetwork import CosineDistanceNetwork, EuclideanDistanceNetwork
from .AttentionalClassify import AttentionalClassify
import torch.nn.functional as F

from pyaction.utils.freeze import freeze  # Freeze network function

"""
Currently: only support one-shot!!!!!!!!!!!!!!!
"""
class MatchingNetwork(nn.Module):
    def __init__(self, cfg): # num_channels=1 nClasses = 0, image_size = 28
        super(MatchingNetwork, self).__init__()
        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.cfg = cfg
        self.num_classes_per_set = cfg.FEW_SHOT.CLASSES_PER_SET
        self.num_samples_per_class = cfg.FEW_SHOT.SAMPLES_PER_CLASS
        self.fce = cfg.FEW_SHOT.FCE

        #####################################################################
        # self.g = Classifier(layer_size = 64, num_channels=num_channels,
        #                     nClasses= nClasses, image_size = image_size )
        #####################################################################
        self.g = ResNetModel(cfg)

        if hasattr(cfg, "FREEZE_RESNET_EXCEPT_NONLOCAL") and cfg.FREEZE_RESNET_EXCEPT_NONLOCAL:
            freeze(self.g, freeze_bn_stats=cfg.FREEZE_BN_STATS, freeze_nln=False)

        if self.fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], vector_dim=cfg.RESNET.FEATURE_DIM)  # self.g.outSize
        
        self.dn = CosineDistanceNetwork()
        if hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "EUCLIDEAN":
            self.dn = EuclideanDistanceNetwork()

        self.classify = AttentionalClassify()


    def forward(self, support_images, support_labels_one_hot, target_images, target_labels=None):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_images: A tensor containing the support set images [batch_size, sequence_size, 3, 8, 224, 224]
        :param support_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_images: A tensor containing the target images (image to produce label for) [batch_size, ntest, 3, 8, 224, 224]
        :param target_labels: A tensor containing the target labels [batch_size, ntest]
        :return: 
        """
        n_target = target_images.size(1)

        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_images.size(1)):
            one_path_wrapped_input = [support_images[:,i,:,:,:,:]]  # for passing stem_helper check
            gen_encode = self.g(one_path_wrapped_input)  # [batchsize, feature_dim]
            encoded_images.append(gen_encode)

        if target_labels is None:
            preds_list = []  # return batch of preds

        # produce embeddings for target images
        for i in np.arange(n_target):
            one_path_wrapped_input = [target_images[:,i,:,:,:,:]]
            gen_encode = self.g(one_path_wrapped_input)
            encoded_images.append(gen_encode)
            outputs = torch.stack(encoded_images)  # [n_support+1, batchsize, feature_dim]

            if self.fce:
                outputs, _, __ = self.lstm(outputs)  # [n_support+1, batchsize, feature_dim]

            # get similarity between support set embeddings and target
            similarities = self.dn(support_vecs=outputs[:-1], target_vec=outputs[-1])  # [nsupport, batchsize]
            similarities = similarities.t()  # [batchsize, nsupport]

            # produce predictions for target probabilities
            preds = self.classify(similarities, support_labels_one_hot=support_labels_one_hot)  # [batch_size, sequence_length]

            if target_labels is None:
                preds_list.append(preds)
            else:
                # calculate accuracy and crossentropy loss
                _, indices = preds.max(1)

                if i == 0:
                    accuracy = torch.mean((indices == target_labels[:,i]).float())
                    cross_entropy_loss = F.cross_entropy(preds, target_labels[:,i].long())
                else:
                    accuracy += torch.mean((indices == target_labels[:, i]).float())
                    cross_entropy_loss += F.cross_entropy(preds, target_labels[:, i].long())

            # delete the last target data
            encoded_images.pop()

        if target_labels is None:
            return torch.stack(preds_list)
            
        return accuracy/n_target, cross_entropy_loss/n_target
        

if __name__ == '__main__':
    unittest.main()



