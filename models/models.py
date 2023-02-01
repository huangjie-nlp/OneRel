import torch
import torch.nn as nn
from transformers import BertModel


class OneRel(nn.Module):
    def __init__(self, config):
        super(OneRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.relation_linear = nn.Linear(self.config.bert_dim * 3, self.config.num_rel * self.config.tag_size)
        self.project_matrix = nn.Linear(self.config.bert_dim * 2, self.config.bert_dim * 3)
        self.dropout = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_encoded_text(self, input_ids, mask):
        bert_encoded_text = self.bert(input_ids=input_ids, attention_mask=mask)[0]
        return bert_encoded_text

    def get_triple_score(self, bert_encoded_text, train):
        batch_size, seq_len, bert_dim = bert_encoded_text.size()
        # [batch_size, seq_len*seq_len, bert_dim]
        head_rep = bert_encoded_text.unsqueeze(dim=2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len * seq_len, bert_dim)
        tail_rep = bert_encoded_text.repeat(1, seq_len, 1)
        # [batch_size, seq_len*seq_len, bert_dim * 2]
        entity_pair = torch.cat([head_rep, tail_rep], dim=-1)

        # [batch_size, seq_len*seq_len, bert_dim * 3]
        entity_pair = self.project_matrix(entity_pair)
        entity_pair = self.dropout_2(entity_pair)
        entity_pair = self.activation(entity_pair)

        # [batch_size, seq_len*seq_len, num_rel*tag_size]
        matrix_socre = self.relation_linear(entity_pair).reshape(batch_size, seq_len, seq_len, self.config.num_rel, self.config.tag_size)
        if train:
            return matrix_socre.permute(0, 4, 3, 1, 2)
        else:
            return matrix_socre.argmax(dim=-1).permute(0, 3, 1, 2)

    def forward(self, data, train=True):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)

        bert_encoded_text = self.get_encoded_text(input_ids, attention_mask)
        bert_encoded_text = self.dropout(bert_encoded_text)

        matrix_score = self.get_triple_score(bert_encoded_text, train)

        return matrix_score