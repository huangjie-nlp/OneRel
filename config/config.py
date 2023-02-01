
class Config():
    def __init__(self):
        self.num_rel = 53
        # self.rel_num = 53
        self.train_file = "./dataset/CIE/train_data.json"
        self.dev_file = "./dataset/CIE/dev_data.json"
        self.schema_fn = "./dataset/CIE/schema.json"
        self.bert_path = "./bert-base-chinese"
        self.tags = "./dataset/tag2id.json"
        self.bert_dim = 768
        self.tag_size = 4
        self.batch_size = 4
        self.max_len = 510
        self.learning_rate = 1e-5
        self.epochs = 100
        self.checkpoint = "checkpoint/OneRel_self.pt"
        self.dev_result = "dev_result/dev_result.json"
        self.dropout_prob = 0.1
        self.entity_pair_dropout = 0.2
