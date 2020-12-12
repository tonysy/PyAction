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
# import unittest

class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, similarities, support_labels_one_hot):
        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [batch_size, sequence_length]
        :param support_labels_one_hot: A tensor with the one hot vectors of the targets for each support set image
                                                                        [batch_size, sequence_length, num_classes]
        :return: Softmax pdf
        """
        
        # similarities = self.softmax(similarities)  # checked.
        preds = torch.bmm(similarities.unsqueeze(1), support_labels_one_hot).squeeze(1)  # [batch_size, sequence_length(i.e. num_classes)]

        # the one-hot encoding essentially acts a permutation on preds(or anything it multiplies)?
        # when support data are sorted by class, the one hot is identity matrix, in which case preds==softmax_similarities here
        # so if support data are always sorted, there is no need to do the multiplication?
        
        return preds

# class AttentionalClassifyTest(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass

#     def test_forward(self):
#         pass

if __name__ == '__main__':
    unittest.main()