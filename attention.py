import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from nonlinear import NonLinearLayer


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = NonLinearLayer([v_dim + q_dim, num_hid])
        self.linear = weight_norm(
            nn.Linear(in_features=num_hid, out_features=1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, v_dim]
        q: [batch, q_dim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, dim=1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)  # [batch, k, v_dim + q_dim]
        joint_repr = self.nonlinear(vq)  # [batch, k, num_hid]
        logits = self.linear(joint_repr)  # [batch, k, 1]
        return logits
