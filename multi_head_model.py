import torch.nn as nn

from classifier import SimpleClassifier
from language_model import WordEmbedding, QuestionEmbedding
from multi_head_attention import MultiHeadAttention
from nonlinear import NonLinearLayer


class MultiHeadAttentionModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(MultiHeadAttentionModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(dim=1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_multi_head_attention_model(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300,
                          'data/glove6b_init_300d.npy')
    q_emb = QuestionEmbedding(300, num_hid, 1, False)
    v_att = MultiHeadAttention(2, dataset.v_dim, q_emb.hidden_num, num_hid)
    q_net = NonLinearLayer([num_hid, num_hid])
    v_net = NonLinearLayer([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.answer_candidates_number, 0.5)
    return MultiHeadAttentionModel(
        w_emb, q_emb, v_att, q_net, v_net, classifier)
