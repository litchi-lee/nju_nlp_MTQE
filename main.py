import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import MTQEDataset, load_data, process_labels
from model import MTQEModelWithPretrained
from summary import summary


def collate_fn(batch):
    src_input_ids = [item['src_input_ids'] for item in batch]
    src_attention_mask = [item['src_attention_mask'] for item in batch]
    tgt_input_ids = [item['tgt_input_ids'] for item in batch]
    tgt_attention_mask = [item['tgt_attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    src_input_ids_padded = pad_sequence(
        src_input_ids, batch_first=True, padding_value=0
    )
    src_attention_mask_padded = pad_sequence(
        src_attention_mask, batch_first=True, padding_value=0
    )
    tgt_input_ids_padded = pad_sequence(
        tgt_input_ids, batch_first=True, padding_value=0
    )
    tgt_attention_mask_padded = pad_sequence(
        tgt_attention_mask, batch_first=True, padding_value=0
    )
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=-100
    )  # 使用 CrossEntropyLoss 的 ignore_index

    return {
        'src_input_ids': src_input_ids_padded,
        'src_attention_mask': src_attention_mask_padded,
        'tgt_input_ids': tgt_input_ids_padded,
        'tgt_attention_mask': tgt_attention_mask_padded,
        'labels': labels_padded,
    }


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        src_input_ids = batch['src_input_ids'].to(device).long()
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device).long()
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)
        labels = batch['labels'].to(device).long()  # Shape: (batch_size, tgt_seq_len)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(
            src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
        )  # (batch_size, tgt_seq_len, num_classes)

        # 计算损失
        outputs = outputs.view(
            -1, outputs.shape[-1]
        )  # 展平成 (batch_size * tgt_seq_len, num_classes)
        labels = labels.view(-1)  # 展平成 (batch_size * tgt_seq_len)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in data_loader:
            src_input_ids = batch['src_input_ids'].to(device).long()
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device).long()
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)
            labels = (
                batch['labels'].to(device).long()
            )  # Shape: (batch_size, tgt_seq_len)

            outputs = model(
                src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
            )

            outputs = outputs.view(
                -1, outputs.shape[-1]
            )  # (batch_size * tgt_seq_len, num_classes)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 获取预测的类别
            _, preds = torch.max(outputs, dim=1)

            # 忽略填充部分的预测
            mask = labels != -100  # 只计算有效标签部分
            correct_preds += torch.sum(
                (preds == labels) & mask
            )  # 只计算有效部分的正确预测
            total_preds += mask.sum()  # 计算有效部分的总数

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy


def run_training(args):
    print("加载训练数据...")
    train_src, train_tgt, train_labels = load_data(os.path.join(args.data_dir, "train"))
    valid_src, valid_tgt, valid_labels = load_data(os.path.join(args.data_dir, "valid"))

    print("处理标签...")
    train_labels = process_labels(train_labels)
    valid_labels = process_labels(valid_labels)
    src_tokenizer = BertTokenizer.from_pretrained(args.encoder_model_name)
    tgt_tokenizer = BertTokenizer.from_pretrained(args.decoder_model_name)

    print("创建数据加载器...")
    train_dataset = MTQEDataset(
        train_src, train_tgt, train_labels, src_tokenizer, tgt_tokenizer, args.max_len
    )
    valid_dataset = MTQEDataset(
        valid_src, valid_tgt, valid_labels, src_tokenizer, tgt_tokenizer, args.max_len
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print("初始化模型...")
    model = MTQEModelWithPretrained(
        args.encoder_model_name, args.decoder_model_name, args.num_classes
    )
    model = model.to(args.device)

    # 打印模型架构
    print("模型架构:")
    src_input_ids = torch.zeros((1, args.max_len), dtype=torch.long).to(args.device)
    src_attention_mask = torch.ones((1, args.max_len), dtype=torch.long).to(args.device)
    tgt_input_ids = torch.zeros((1, args.max_len), dtype=torch.long).to(args.device)
    tgt_attention_mask = torch.ones((1, args.max_len), dtype=torch.long).to(args.device)
    summary(model, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_valid_loss = float('inf')
    best_model_path = None

    log_data = {"train_loss": [], "valid_loss": [], "valid_acc": [], "saved_epochs": []}

    for epoch in range(args.epochs):
        print("=" * 50)
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, args.device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, args.device)
        print(
            f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
        )

        # 记录数据
        log_data["train_loss"].append(train_loss)
        log_data["valid_loss"].append(valid_loss)
        log_data["valid_acc"].append(valid_acc.item())

        # 保存验证集上表现最好的模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_path = os.path.join(args.model_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"保存模型到 {best_model_path}")
            log_data["saved_epochs"].append(epoch + 1)  # 记录保存模型的 epoch

        # 保存日志数据到文件
        with open(os.path.join(args.logs_path, "training_log.json"), "w") as f:
            json.dump(log_data, f)

    # 绘制训练损失和准确率图表
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log_data["train_loss"], label="Train Loss")
    plt.plot(log_data["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(log_data["valid_acc"], label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.savefig(os.path.join(args.logs_path, "training_plots.png"))
    plt.show()

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./resource/data", help="数据文件夹根目录"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results",
        help="模型保存路径",
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        default="./logs",
        help="日志文件保存路径",
    )
    parser.add_argument(
        "--encoder_model_name",
        type=str,
        default="./pretrained/bert-base-chinese",
        help="编码器预训练模型路径",
    )
    parser.add_argument(
        "--decoder_model_name",
        type=str,
        default="./pretrained/bert-base-uncased",
        help="解码器预训练模型路径",
    )
    parser.add_argument("--num_classes", type=int, default=4, help="标签类别数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--max_len", type=int, default=256, help="序列最大长度")
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="设备",
    )
    parser.add_argument("--gpu", type=int, default=0, help="使用的GPU")
    args = parser.parse_args()

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)

    model = run_training(args)
