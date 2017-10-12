import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class SST_AD_Prop(nn.Module):
    """Implementation of SSTAD proposal module, for use in full SSTAD model.
    Forwards both the SST output proposals and the corresponding hidden state representation.
    """
    def __init__(self, num_proposals=16, num_rnn_layers=1, h_width=512, input_size=500, dropout=0):
        super(SST_AD_Prop, self).__init__() 
        self.rnn = nn.GRU(input_size=input_size, hidden_size=h_width, num_layers=num_rnn_layers, dropout=dropout)
        self.lin_out = nn.Linear(h_width, num_proposals)
        self.nonlin_final = nn.Sigmoid()
        self.num_rnn_layers = num_rnn_layers
        self.h_width = h_width

    def init_hidden_state(self, batch_sz):
        return Variable(torch.zeros(self.num_rnn_layers, batch_sz, self.h_width).cuda())

    def forward(self, inputs):
        batch_sz = inputs.size(0) # should be batch_sz (~200 in old set-up)
        inputs = torch.transpose(inputs,0,1)
        h0 = self.init_hidden_state(batch_sz)
        rnn_output, h_n = self.rnn.forward(inputs, h0)
        # get proposals output (L x N x h_width) ==> (N x L x K)
        output = self.lin_out.forward(rnn_output.view(rnn_output.size(0)*rnn_output.size(1), rnn_output.size(2)))
        lin_out = output.view(rnn_output.size(0), rnn_output.size(1), output.size(1))
        final_out = self.nonlin_final(torch.transpose(lin_out,0,1))
        return final_out, rnn_output


class SST_AD_SegAct(nn.Module):
    """Implementation of SSTAD segment action classification module, for use in full SSTAD model.
    Forwards both the output segment classification and the corresponding hidden state representation.
    ** Note that num_classes = number of classes + 1 for background class.
    """
    def __init__(self, num_classes=201, num_rnn_layers=1, h_width=512, input_size=500, dropout=0, init_range=None,**kwargs):
        super(SST_AD_SegAct, self).__init__() 
        self.rnn = nn.GRU(input_size=input_size, hidden_size=h_width, num_layers=num_rnn_layers, dropout=dropout) #, batch_first=True)
        self.lin_out = nn.Linear(h_width, num_classes)
        self.nonlin_eval = torch.nn.Softmax()
        self.num_rnn_layers = num_rnn_layers
        self.h_width = h_width
        self.init_weights(init_range)

    def init_weights(self, init_range):
        if init_range is None:
            return
        elif isinstance(init_range, float):
            init_range = (init_range,) * 2
        elif len(init_range) != 2:
            raise ValueError('Argument ({}) different than tuple with size {} '
                             'different than 2'.format(type(init_range),
                                                       len(init_range)))

        for i in range(self.num_rnn_layers):
            weight_var = getattr(self.rnn, 'weight_ih_l{}'.format(i))
            weight_var.data.uniform_(-init_range[0], init_range[0])
        self.lin_output.weight.data.uniform_(-init_range[1], init_range[1])
        self.lin_out.bias.data.fill_(0)

    def init_hidden_state(self, batch_sz):
        return Variable(torch.zeros(self.num_rnn_layers, batch_sz, self.h_width).cuda())

    def forward(self, inputs):
        batch_sz = inputs.size(0) # should be batch_sz (~200 in old set-up)
        inputs = torch.transpose(inputs,0,1)
        h0 = self.init_hidden_state(batch_sz)
        rnn_output, h_n = self.rnn.forward(inputs, h0)
        # get "output" after linear layer. 
        output = self.lin_out.forward(rnn_output.view(rnn_output.size(0)*rnn_output.size(1), rnn_output.size(2)))
        L, N = rnn_output.size(0), rnn_output.size(1)
        C = output.size(1)
        assert L*N == output.size(0), "ERROR: mismatch in output tensor dimensions"
        fin_out = output.view(L, N, C) 
        fin_out = torch.transpose(fin_out,0,1) 
        fin_out = fin_out.contiguous().view(N*L, C)
        return fin_out, rnn_output

    def obtain_class_scores(self, output, rnn_output_size, flatten_two_dim=True): 
        # use this during evaluation of segment-level classification.
        L, N = rnn_output_size[0], rnn_output_size[1]
        C = output.size(1)
        assert L*N == output.size(0), "ERROR: mismatch in output tensor dimensions"
        class_scores = self.nonlin_eval(output)
        if flatten_two_dim:
            return class_scores
        return class_scores.view(N, L, C)


class SST_AD_PreRel(nn.Module): # Note: basic pre-release version... will be updating code for actual release. 
    """Note: basic pre-release factored version of SS-TAD. Stay tuned for updated versions coming later :)
    """
    def __init__(self, num_proposals=16, num_classes=21, num_rnn_layers=1, h_width=512, input_size=500, dropout=0):
        super().__init__() 
        self.SST_Prop = SST_AD_Prop(num_proposals, num_rnn_layers, h_width, input_size, dropout)
        self.SST_SegAct = SST_AD_SegAct(num_classes, num_rnn_layers, h_width, input_size, dropout)
        self.num_rnn_layers = num_rnn_layers
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.eval_non_lin = nn.Softmax()
        self.lin_out = nn.Linear(h_width*2, num_proposals * num_classes)
        self.h_width = h_width

    def forward(self, inputs):
        (proposals_out, h_prop), (class_out, h_cls) = \
                (self.SST_Prop.forward(inputs), self.SST_SegAct.forward(inputs))
        h_comb = torch.cat((h_prop,h_cls), 2) 
        self.h_cls_size = h_cls.size()
        self.seq_len = h_comb.size(0)
        self.batch_num = h_comb.size(1)
        output = self.lin_out.forward(h_comb.view(h_comb.size(0)*h_comb.size(1), h_comb.size(2)))
        output = self.convert_to_batch_order(output, self.batch_num, self.seq_len, self.num_proposals, self.num_classes)
        final_joint_cls_scores = self.get_class_scores(output)
        return output, proposals_out, class_out, final_joint_cls_scores

    def get_seg_class_scores(self, class_out):
        return self.SST_SegAct.obtain_class_scores(class_out, self.h_cls_size)

    def convert_to_batch_order(self, output, N, L, K, C):
        output = output.view(L, N, K, C)
        output = torch.transpose(output, 0,1)
        return output.contiguous().view(N*L*K, C)

    def get_class_scores(self, output): 
        class_scores = self.eval_non_lin(output)
        fin_out = class_scores.view(self.batch_num, self.seq_len, self.num_proposals, self.num_classes) 
        return fin_out 

class VisTransformLayer(nn.Module):
    def __init__(self):
        super(VisTransformLayer, self).__init__()
        self.pool_input_size = 8192
        self.input_size = 500
        self.fc6 = nn.Linear(self.pool_input_size, 4096) 
        self.relu6 = nn.ReLU()
        self.dpt6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dpt7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, self.input_size)
    def forward(self, inputs):
        # input_size = inputs.size()
        batch_sz = inputs.size(0)
        seq_len = inputs.size(1)
        # flatten input into 1 vector
        inputs = inputs.view(batch_sz * seq_len, self.pool_input_size) 
        fc6_feats = self.dpt6.forward(self.relu6.forward(self.fc6.forward(inputs)))
        fc7_feats = self.dpt7.forward(self.relu7.forward(self.fc7.forward(fc6_feats)))
        final_feats = self.fc8.forward(fc7_feats)
        return final_feats

