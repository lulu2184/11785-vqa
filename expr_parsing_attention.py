import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from nonlinear import NonLinearLayer


class ExprParsingAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.1):
        super(ExprParsingAttention, self).__init__()

        self.v_proj = NonLinearLayer([v_dim, num_hid])
        self.q_proj = NonLinearLayer([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, v_dim]
        q: [batch, q_dim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits
