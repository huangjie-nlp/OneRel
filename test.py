from models.models import OneRel
import torch
from config.config import Config
import numpy as np
import json
from transformers import BertTokenizer

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = Config()

id2label = json.load(open(config.schema_fn, "r", encoding="utf-8"))[1]
id2tag = json.load(open(config.tags, "r", encoding="utf-8"))[1]

tokenizer = BertTokenizer.from_pretrained(config.bert_path)
model = OneRel(config)
model.load_state_dict(torch.load(config.checkpoint, map_location=device))
model.to(device)
model.eval()


def parser(sentence):
    token = ['[CLS]'] + list(sentence)[:510] + ['[SEP]']
    token2id = [tokenizer.convert_tokens_to_ids(token)]
    mask = [[1]*len(token)]
    data = {"input_ids": torch.LongTensor(token2id), "attention_mask": torch.LongTensor(mask)}
    # [num_rel, seq_len, seq_len]
    output = model(data, False).cpu()[0]
    num_rel, seq_lens, seq_lens = output.shape

    relations, heads, tails = np.where(output > 0)

    predict = []
    relation_num = len(relations)
    predict = {"text": sentence, "predict": []}
    if relation_num > 0:
        for r in range(relation_num):
            rel2indx = relations[r]
            h_start_index = heads[r]
            t_start_index = tails[r]
            if output[rel2indx][h_start_index][t_start_index] == id2tag["HB-TB"] and r + 1 < relation_num:
                t_end_index = tails[r + 1]
                if output[rel2indx][h_start_index][t_end_index] == id2tag["HB-TE"]:
                    for h_end_index in range(h_start_index, seq_lens):
                        if output[rel2indx][h_end_index][t_end_index] == id2tag["HE-TE"]:
                            subject_head, subject_tail = h_start_index, h_end_index
                            object_head, object_tail = t_start_index, t_end_index
                            subject = "".join(token[subject_head: subject_tail+1])
                            object = "".join(token[object_head: object_tail+1])
                            relation = id2label[str(rel2indx)]
                            if len(subject) > 0 and len(object) > 0:
                                predict["predict"].append((subject, relation, object))
                            break
    return predict

if __name__ == '__main__':
    while True:
        sentence = input("请输入:")
        predict = parser(sentence)
        print(json.dumps(predict, indent=4, ensure_ascii=False))