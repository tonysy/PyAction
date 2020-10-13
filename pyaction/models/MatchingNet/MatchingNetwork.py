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
from .DistanceNetwork import CosineDistanceNetwork, EuclideanDistanceNetwork, \
        FrameMaxCosineDistanceNetwork, FrameStraightAlignCosineDistanceNetwork, \
        FrameGreedyAlignCosineDistanceNetwork, FrameSelfContextMeanNetwork, \
        FrameCosineDistanceMeanNetwork, FrameMeanCosineDistanceNetwork, \
        FrameFrechetMeanCosineDistanceNetwork, \
        FrameCosineDistanceSumNetwork, \
        FrameOTAMDistanceNetwork
from .AttentionalClassify import AttentionalClassify
import torch.nn.functional as F
from pyaction.utils.freeze import freeze  # Freeze network function

def update_frame_fuse_method(cfg):
    if hasattr(cfg.FEW_SHOT, "DISTANCE"):
        if str(cfg.FEW_SHOT.DISTANCE).startswith("FRAME"):
            cfg.FRAME_FUSE = "FRAME_CAT"
            if cfg.MODEL.ARCH.startswith("c2d"):
                cfg.MODEL.ARCH = "c2d_nopool"
            elif cfg.MODEL.ARCH.startswith("i3d"):
                cfg.MODEL.ARCH = "i3d_nopool"
            else:
                raise RuntimeError("Error from MarchingNetwork.py: unexpected model arch, please check!")
            return
    cfg.FRAME_FUSE = "FRAME_MEAN"

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

        # Embedding Network
        update_frame_fuse_method(cfg)  # cfg.FRAME_FUSE := FRAME_MEAN/FRAME_CAT
        self.g = ResNetModel(cfg)

        # Freeze embedding
        if hasattr(cfg, "FREEZE_RESNET_EXCEPT_NONLOCAL") and cfg.FREEZE_RESNET_EXCEPT_NONLOCAL:
            freeze(self.g, freeze_bn_stats=cfg.FREEZE_BN_STATS, freeze_nln=False)

        # Full Context Embedding
        if self.fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], vector_dim=cfg.RESNET.FEATURE_DIM)  # self.g.outSize
        
        # Distance Network
        if hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "EUCLIDEAN":
            self.dn = EuclideanDistanceNetwork()
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_MAX_COSINE":
            self.dn = FrameMaxCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_STRAIGHT_ALIGN_COSINE":
            self.dn = FrameStraightAlignCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_GREEDY_ALIGN_COSINE":
            self.dn = FrameGreedyAlignCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_SELF_CONTEXT_MEAN_COSINE":
            self.dn = FrameSelfContextMeanNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_COSINE_MEAN":
            self.dn = FrameCosineDistanceMeanNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_COSINE_SUM":
            self.dn = FrameCosineDistanceSumNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_MEAN_COSINE":
            self.dn = FrameMeanCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_FRECHETMEAN_COSINE":
            self.dn = FrameFrechetMeanCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_OTAM":
            lam = cfg.FEW_SHOT.LAMBDA if hasattr(cfg.FEW_SHOT, "LAMBDA") else None
            ndirection = cfg.FEW_SHOT.NDIR if hasattr(cfg.FEW_SHOT, "NDIR") else None
            self.dn = FrameOTAMDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES, lam=lam, ndirection=ndirection)
        else:  # DEFAULT
            self.dn = CosineDistanceNetwork()

        # Sanity check
        print(self.dn, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Classifier
        self.classify = AttentionalClassify()

        # Learnable temperature
        if hasattr(self.cfg.FEW_SHOT, "LEARN_TEMP") and cfg.FEW_SHOT.LEARN_TEMP:
            self.temperature = nn.Parameter(torch.tensor([1.0]), requires_grad=True)


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

            # Softmax with temporature
            if hasattr(self.cfg.FEW_SHOT, "TEMP"):
                temperature = self.cfg.FEW_SHOT.TEMP
                similarities *= temperature
            elif hasattr(self.cfg.FEW_SHOT, "LEARN_TEMP") and self.cfg.FEW_SHOT.LEARN_TEMP:
                similarities *= self.temperature

            if hasattr(self.cfg, "TEST_DEBUG") and self.cfg.TEST_DEBUG:
                import pdb; pdb.set_trace()

            # produce predictions for target probabilities
            preds = self.classify(similarities, support_labels_one_hot=support_labels_one_hot)  # [batch_size, sequence_length]

            if hasattr(self.cfg, "TEST_DEBUG") and self.cfg.TEST_DEBUG:
                import pdb; pdb.set_trace()

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

                if hasattr(self.cfg, "TEST_DEBUG") and self.cfg.TEST_DEBUG:
                    import pdb; pdb.set_trace()

            # delete the last target data
            encoded_images.pop()

        if target_labels is None:
            return torch.stack(preds_list)
            
        return accuracy/n_target, cross_entropy_loss/n_target
        

if __name__ == '__main__':
    unittest.main()



