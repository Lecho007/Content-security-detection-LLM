import json
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ✅ 自定义攻击Prompt数据集类
class AttackPromptDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        label = self.samples[idx]["label"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ✅ 加载模型与分词器
model_name = "../../models/chinese-macbert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=17)  # ← 修改标签数！

# ✅ 加载数据集
train_dataset = AttackPromptDataset("data/train.jsonl", tokenizer)
eval_dataset = AttackPromptDataset("data/val.jsonl", tokenizer)

"""
output_dir：保存模型的文件夹路径。

num_train_epochs：训练轮数。

per_device_train_batch_size：每张卡训练 batch。

evaluation_strategy="epoch"：每个 epoch 进行一次验证。

load_best_model_at_end：根据验证集结果自动保留最优模型。

save_total_limit=2：最多保留两个 checkpoint。
"""

# ✅ 微调参数
training_args = TrainingArguments(
    output_dir="../../models/chiness-macbert-base_output",  # ← 自定义保存路径
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    eval_steps=200,
    save_strategy="epoch",
    save_steps=200,
    logging_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2,
)

# ✅ 启动Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ✅ 开始训练
trainer.train()

# ✅ 保存模型（可选手动）
trainer.save_model("../../models/chiness-macbert-base_output")
tokenizer.save_pretrained("../../models/chiness-macbert-base_output")
