import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.linear1 = weight_norm(nn.Linear(input_dim, hidden_dim), dim=None)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = weight_norm(nn.Linear(hidden_dim, output_dim), dim=None)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        return self.linear2(out)
