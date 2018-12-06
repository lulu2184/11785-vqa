import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from attention import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, v_dim, q_dim, num_hid):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.v_dim = v_dim // self.head_num
        self.q_dim = q_dim // self.head_num
        self.num_hid = num_hid // self.head_num
        self.atts = []
        for i in range(self.head_num):
            self.atts.append(
                Attention(self.v_dim, self.q_dim, self.num_hid).cuda())
        self.linear = weight_norm(
            nn.Linear(in_features=head_num, out_features=1), dim=None)

    def forward(self, v, q):
        v = v.permute(2, 0, 1)  # [batch, k, v_dim] -> [v_dim, batch, k]
        q = q.permute(1, 0)

        w = None
        for i in range(self.head_num):
            head_v = v[i * self.v_dim: (i + 1) * self.v_dim].permute(1, 2, 0)
            head_q = q[i * self.q_dim:(i + 1) * self.q_dim].permute(1, 0)
            if i == 0:
                w = self.atts[i](head_v, head_q).permute(2, 0, 1)
            else:
                w = torch.cat(
                    (w, self.atts[i](head_v, head_q).permute(2, 0, 1)))

        w = self.linear(w.permute(1, 2, 0))
        return w
