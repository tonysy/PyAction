import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from pyaction.models.video_model_builder import ResNetModel
from .GNN_nl import GNN_nl


class FewShotGNN(nn.Module):
    def __init__(self, cfg):
        super(FewShotGNN, self).__init__()
        """
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.cfg = cfg
        self.num_classes_per_set = cfg.FEW_SHOT.CLASSES_PER_SET
        self.num_samples_per_class = cfg.FEW_SHOT.SAMPLES_PER_CLASS
        self.g = ResNetModel(cfg)
        self.gnn = GNN_nl(dim_features=[2048+self.num_classes_per_set, 1024, 1024, self.num_classes_per_set])
        

    def forward(self, support_images, support_labels_one_hot, target_images, target_labels=None):
        """
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

            outputs = torch.transpose(outputs, 0, 1)  # [batchsize, n_support+1, feature_dim=2048] # checked

            lshape = support_labels_one_hot.shape
            zero_pad = torch.zeros(lshape[0], 1, lshape[2])
            zero_pad = zero_pad.cuda()
            labels_one_hot = torch.cat((support_labels_one_hot, zero_pad), 1)  # [batchsize, n_support+1, nclass] # checked

            # Concat image feature with one-hot label
            nodes = torch.cat((outputs, labels_one_hot), 2)  # [batchsize, n_support+1, feature_dim + nclass] # checked

            # Get probabilities of current target
            preds = self.gnn(nodes)

            # import pdb; pdb.set_trace()  # see preds.shape

            # produce predictions for target probabilities
            # preds = self.classify(similarities, support_labels_one_hot=support_labels_one_hot)  # [batch_size, sequence_length]

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

