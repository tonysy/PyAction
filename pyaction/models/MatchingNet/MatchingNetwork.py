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
        FrameGreedyAlignCosineDistanceNetwork, \
        FrameCosineDistanceMeanNetwork, FrameMeanCosineDistanceNetwork, \
        FrameCosineDistanceSumNetwork, \
        FrameOTAMDistanceNetwork, \
        TemporalGNN, \
        FrameMeanLearnableDistanceNetwork, \
        FrameMeanMeanCosineDistanceNetwork, \
        MHATT_temporal
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

        # The second g
        if hasattr(cfg.FEW_SHOT, "G2") and cfg.FEW_SHOT.G2:
            self.has_g2 = True
            cfg.RESNET.GET_FEATURE=False  ### !!!!!
            self.g2 = ResNetModel(cfg)
            for p in self.g2.parameters():  # do not fine-tune it!!!
                p.requires_grad = False
        else:
            self.has_g2 = False

        # # Freeze embedding
        # if hasattr(cfg, "FREEZE_RESNET_EXCEPT_NONLOCAL") and cfg.FREEZE_RESNET_EXCEPT_NONLOCAL:
        #     freeze(self.g, freeze_bn_stats=cfg.FREEZE_BN_STATS, freeze_nln=False)

        # Full Context Embedding
        if self.fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], vector_dim=cfg.RESNET.FEATURE_DIM)  # self.g.outSize

        # Reduce Feature vector dim
        if hasattr(cfg.FEW_SHOT, "LINEAR"):
            df = cfg.FEW_SHOT.LINEAR
            self.ln = nn.Linear(2048, df)
            print("Reduced dim: ", df, "!!!!!!!!!!!!!!!!")

        # Temporal GNN
        if hasattr(self.cfg.FEW_SHOT, "TGNN") and self.cfg.FEW_SHOT.TGNN:
            cos_scaler = cfg.FEW_SHOT.TGNN_COS_SCALER if hasattr(cfg.FEW_SHOT, "TGNN_COS_SCALER") else None
            self.tgnn = TemporalGNN(nframes=cfg.DATA.NUM_FRAMES, cos_scaler=cos_scaler)

        # Multi-head self-attention: Temporal
        if hasattr(self.cfg, "MHATT_TEMPORAL") and self.cfg.MHATT_TEMPORAL:
            nhead = cfg.MHATT_TEMPORAL.NHEAD
            pre_norm = cfg.MHATT_TEMPORAL.PRE_NORM  # True/False
            self.mhatt_temporal = MHATT_temporal(nframes=cfg.DATA.NUM_FRAMES, nhead=nhead, pre_norm=pre_norm)
            print("Multi-head Temporal self-attention: pre_norm={} !!!!!!!!!!!".format(pre_norm))

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
            if hasattr(cfg.FEW_SHOT, "UNNORM_MEAN") and cfg.FEW_SHOT.UNNORM_MEAN:
                norm = False
            else:
                norm = True
            self.dn = FrameMeanCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES, norm=norm)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_MEAN_MEAN_COSINE":
            self.dn = FrameMeanMeanCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_FRECHETMEAN_COSINE":
            self.dn = FrameFrechetMeanCosineDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_OTAM":
            lam = cfg.FEW_SHOT.LAMBDA if hasattr(cfg.FEW_SHOT, "LAMBDA") else None
            ndirection = cfg.FEW_SHOT.NDIR if hasattr(cfg.FEW_SHOT, "NDIR") else None
            self.dn = FrameOTAMDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES, lam=lam, ndirection=ndirection)
        elif hasattr(cfg.FEW_SHOT, "DISTANCE") and cfg.FEW_SHOT.DISTANCE == "FRAME_MEAN_LEARNABLE":
            self.dn = FrameMeanLearnableDistanceNetwork(nframes=cfg.DATA.NUM_FRAMES)
        else:  # DEFAULT
            self.dn = CosineDistanceNetwork()

        # Sanity check
        print("dn:", self.dn, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # second dn for g2
        if self.has_g2:
            assert self.cfg.FEW_SHOT.DISTANCE2 == "FRAME_COSINE_MEAN"
            self.dn2 = FrameCosineDistanceMeanNetwork(nframes=cfg.DATA.NUM_FRAMES)
            # Sanity check
            print("dn2:", self.dn2, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # activation for g2 logits
            if self.cfg.FEW_SHOT.G2_ACT:
                self.g2_act = nn.Softmax(dim=-1)

        # Classifier
        self.classify = AttentionalClassify()

        # Learnable temperature
        if hasattr(self.cfg.FEW_SHOT, "LEARN_TEMP") and cfg.FEW_SHOT.LEARN_TEMP:
            self.temperature = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        # Relation net
        if hasattr(self.cfg.FEW_SHOT, "MSELOSS") and self.cfg.FEW_SHOT.MSELOSS:
            self.mseloss = nn.MSELoss()
            print("MSE Loss!!!!!!!!!!!!")



    def forward(self, support_images, support_labels_one_hot, target_images, target_labels=None, return_indices=False):
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
        encoded_images = []  # by g
        if self.has_g2:
            encoded_images2 = [] # by g2

        for i in np.arange(support_images.size(1)):
            if self.has_g2:
                one_path_wrapped_input = [support_images[:,i,:,:,:,:].clone()]  # for passing stem_helper check
            else:
                one_path_wrapped_input = [support_images[:,i,:,:,:,:]]  # for passing stem_helper check

            gen_encode = self.g(one_path_wrapped_input)  # [batchsize, feature_dim]
            if self.has_g2:
                one_path_wrapped_input = [support_images[:, i, :, :, :, :]]
                gen_encode2 = self.g2(one_path_wrapped_input)  # [batchsize, feature_dim]
                # assert gen_encode2.shape == (4, 8000), "{}  {}".format(gen_encode2.shape[0],gen_encode2.shape[1])  ###
                # will raise err at the end because num of data not divisible by 4 !!!!!!!!!!!!!!!!!!!!!!
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            ### Reduce Feature vector dim ###
            if hasattr(self.cfg.FEW_SHOT, "LINEAR"):
                bs, _ = gen_encode.shape  # torch.Size([bs, 16384])
                gen_encode = gen_encode.view(bs, self.dn.nframes, -1)
                gen_encode = self.ln(gen_encode)
                gen_encode = gen_encode.view(bs, -1)

            encoded_images.append(gen_encode)
            if self.has_g2:
                encoded_images2.append(gen_encode2)

        if target_labels is None:
            preds_list = []  # return batch of preds

        # For sanity check
        if return_indices:
            list_indices = []

        # produce embeddings for target images
        for i in np.arange(n_target):
            if self.has_g2:
                one_path_wrapped_input = [target_images[:,i,:,:,:,:].clone()]
            else:
                one_path_wrapped_input = [target_images[:,i,:,:,:,:]]
            gen_encode = self.g(one_path_wrapped_input)

            if self.has_g2:
                one_path_wrapped_input = [target_images[:,i,:,:,:,:]]
                gen_encode2 = self.g2(one_path_wrapped_input)

            ### Reduce Feature vector dim ###
            if hasattr(self.cfg.FEW_SHOT, "LINEAR"):
                bs, _ = gen_encode.shape  # torch.Size([bs, 16384])
                gen_encode = gen_encode.view(bs, self.dn.nframes, -1)
                gen_encode = self.ln(gen_encode)
                gen_encode = gen_encode.view(bs, -1)

            encoded_images.append(gen_encode)
            outputs = torch.stack(encoded_images)  # [n_support+1, batchsize, feature_dim]

            if self.has_g2:
                encoded_images2.append(gen_encode2)
                outputs2 = torch.stack(encoded_images2)  # [n_support+1, batchsize, feature_dim]

            if self.fce:
                outputs, _, __ = self.lstm(outputs)  # [n_support+1, batchsize, feature_dim]

            if hasattr(self.cfg.FEW_SHOT, "TGNN") and self.cfg.FEW_SHOT.TGNN:
                assert n_target == 1  # should only pass tgnn network one time!
                for output in outputs:
                    output = self.tgnn(output)

            # Multi-head self-attention: Temporal
            if hasattr(self.cfg, "MHATT_TEMPORAL") and self.cfg.MHATT_TEMPORAL:
                assert n_target == 1  # should only pass tgnn network one time!
                for output in outputs:
                    output = self.mhatt_temporal(output)

            # get similarity between support set embeddings and target
            similarities = self.dn(support_vecs=outputs[:-1], target_vec=outputs[-1])  # [nsupport, batchsize]
            similarities = similarities.t()  # [batchsize, nsupport]

            # get the second similarity
            if self.has_g2:
                # whether to activate the distribution
                if self.cfg.FEW_SHOT.G2_ACT:
                    assert n_target == 1
                    outputs2 = self.g2_act(outputs2)  # softmax at dim -1
                # get similarity between support set embeddings and target
                similarities2 = self.dn2(support_vecs=outputs2[:-1], target_vec=outputs2[-1])  # [nsupport, batchsize]
                similarities2 = similarities2.t()  # [batchsize, nsupport]

                # weighted sum
                similarities += similarities2 * self.cfg.FEW_SHOT.G2_ALPHA

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

                # For sanity check
                if return_indices:
                    list_indices.append(indices)

                if hasattr(self.cfg.FEW_SHOT, "MSELOSS") and self.cfg.FEW_SHOT.MSELOSS:

                    preds = F.sigmoid(preds)

                    # target_labels: Add extra dimension for the one_hot
                    target_labels = torch.unsqueeze(target_labels, -1)  # (bs, ntest, 1)
                    batch_size = target_labels.shape[0]
                    n_samples = target_labels.shape[1]
                    assert target_labels.shape[1] == target_labels.shape[2] == 1

                    target_labels_one_hot = torch.zeros(batch_size, n_samples,
                                                        self.cfg.FEW_SHOT.CLASSES_PER_SET).cuda()  # the last dim as one-hot
                    target_labels_one_hot.scatter_(2, target_labels.cuda(), 1.0)

                    if i == 0:
                        accuracy = torch.mean((indices == target_labels[:, i]).float())
                        loss = self.mseloss(preds, target_labels_one_hot[:, i])
                    else:
                        accuracy += torch.mean((indices == target_labels[:, i]).float())
                        loss += self.mseloss(preds, target_labels_one_hot[:, i])
                else:
                    if i == 0:
                        accuracy = torch.mean((indices == target_labels[:, i]).float())
                        loss = F.cross_entropy(preds, target_labels[:, i].long())
                    else:
                        accuracy += torch.mean((indices == target_labels[:, i]).float())
                        loss += F.cross_entropy(preds, target_labels[:, i].long())

                # if i == 0:
                #     accuracy = torch.mean((indices == target_labels[:,i]).float())
                #     cross_entropy_loss = F.cross_entropy(preds, target_labels[:,i].long())
                # else:
                #     accuracy += torch.mean((indices == target_labels[:, i]).float())
                #     cross_entropy_loss += F.cross_entropy(preds, target_labels[:, i].long())

                if hasattr(self.cfg, "TEST_DEBUG") and self.cfg.TEST_DEBUG:
                    import pdb; pdb.set_trace()

            # delete the last target data
            encoded_images.pop()

        if target_labels is None:
            return torch.stack(preds_list)

        if return_indices:
            tensor_indices = torch.stack(list_indices).transpose(-1, -2)
            assert tensor_indices.shape == target_labels.shape  # bs,ntest
            return accuracy / n_target, loss / n_target, tensor_indices
        else:
            return accuracy/n_target, loss/n_target
        

if __name__ == '__main__':
    unittest.main()



