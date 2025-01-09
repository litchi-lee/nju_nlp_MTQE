import os

import torch
from torch.utils.data import Dataset


def load_data(data_path):
    # 文件路径
    zh_file = os.path.join(data_path, os.path.basename(data_path) + ".zh-en.zh")
    en_file = os.path.join(data_path, os.path.basename(data_path) + ".zh-en.en")
    dtag_file = os.path.join(data_path, os.path.basename(data_path) + ".zh-en.dtag")

    # 读取文件内容
    with open(zh_file, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f.readlines()]
    with open(en_file, 'r', encoding='utf-8') as f:
        tgt_texts = [line.strip() for line in f.readlines()]
    with open(dtag_file, 'r', encoding='utf-8') as f:
        labels = [line.strip().split() for line in f.readlines()]

    # 确保三者行数一致
    assert len(src_texts) == len(tgt_texts) == len(labels), "数据文件行数不匹配！"

    return src_texts, tgt_texts, labels


def load_test_data(data_path):
    # 文件路径
    zh_file = os.path.join(data_path, os.path.basename(data_path) + ".zh-en.zh")
    en_file = os.path.join(data_path, os.path.basename(data_path) + ".zh-en.en")

    # 读取文件内容
    with open(zh_file, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f.readlines()]
    with open(en_file, 'r', encoding='utf-8') as f:
        tgt_texts = [line.strip() for line in f.readlines()]

    # 确保两者行数一致
    assert len(src_texts) == len(tgt_texts), "数据文件行数不匹配！"

    return src_texts, tgt_texts


def process_labels(label_lines):
    LABEL_MAP = {'OK': 0, 'minor': 1, 'major': 2, 'critical': 3}

    return [[LABEL_MAP[label] for label in line] for line in label_lines]


class MTQEDataset(Dataset):
    def __init__(
        self, src_texts, tgt_texts, labels, src_tokenizer, tgt_tokenizer, max_len
    ):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.labels = labels
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]  # 源句子（中文）
        tgt_text = self.tgt_texts[idx]  # 目标句子（英文）
        label = self.labels[idx]  # 标签

        # 使用 BERT 的 tokenizer 进行编码
        src_ids = self.src_tokenizer.encode(
            src_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )
        tgt_ids = self.tgt_tokenizer.encode(
            tgt_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )

        # 扩充 labels 到 max_len，填充部分使用 -100（用于 CrossEntropyLoss 忽略填充）
        label = label[: self.max_len]  # 截断标签（如果标签长度超过 max_len）
        label = label + [-100] * (self.max_len - len(label))  # 填充标签

        # 构造注意力掩码
        src_mask = [
            1 if token_id != self.src_tokenizer.pad_token_id else 0
            for token_id in src_ids
        ]
        tgt_mask = [
            1 if token_id != self.tgt_tokenizer.pad_token_id else 0
            for token_id in tgt_ids
        ]

        return {
            'src_input_ids': torch.tensor(src_ids, dtype=torch.long),
            'src_attention_mask': torch.tensor(src_mask, dtype=torch.long),
            'tgt_input_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'tgt_attention_mask': torch.tensor(tgt_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
        }


class MTQETestDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]  # 源句子（中文）
        tgt_text = self.tgt_texts[idx]  # 目标句子（英文）

        # 使用 BERT 的 tokenizer 进行编码
        src_ids = self.src_tokenizer.encode(
            src_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )
        tgt_ids = self.tgt_tokenizer.encode(
            tgt_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )

        # 构造注意力掩码
        src_mask = [
            1 if token_id != self.src_tokenizer.pad_token_id else 0
            for token_id in src_ids
        ]
        tgt_mask = [
            1 if token_id != self.tgt_tokenizer.pad_token_id else 0
            for token_id in tgt_ids
        ]

        return {
            'src_input_ids': torch.tensor(src_ids, dtype=torch.long),
            'src_attention_mask': torch.tensor(src_mask, dtype=torch.long),
            'tgt_input_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'tgt_attention_mask': torch.tensor(tgt_mask, dtype=torch.long),
            'tgt_len': len(tgt_text.split()),  # 返回原始序列的长度
        }
