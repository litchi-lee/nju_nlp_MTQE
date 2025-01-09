import argparse
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import MTQETestDataset, load_test_data
from model import MTQEModelWithPretrained


def collate_fn(batch):
    src_input_ids = [item['src_input_ids'] for item in batch]
    src_attention_mask = [item['src_attention_mask'] for item in batch]
    tgt_input_ids = [item['tgt_input_ids'] for item in batch]
    tgt_attention_mask = [item['tgt_attention_mask'] for item in batch]
    tgt_lens = [item['tgt_len'] for item in batch]

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

    return {
        'src_input_ids': src_input_ids_padded,
        'src_attention_mask': src_attention_mask_padded,
        'tgt_input_ids': tgt_input_ids_padded,
        'tgt_attention_mask': tgt_attention_mask_padded,
        'tgt_lens': tgt_lens,
    }


def test(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            src_input_ids = batch['src_input_ids'].to(device).long()
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device).long()
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)
            tgt_lens = batch['tgt_lens']

            outputs = model(
                src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
            )

            _, preds = torch.max(outputs, dim=2)
            preds = preds.cpu().numpy()

            # 根据原始序列长度裁剪预测结果
            for i, length in enumerate(tgt_lens):
                predictions.append(preds[i][:length])

    return predictions


def run_test(args):
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)

    print("加载测试数据...")
    test_src, test_tgt = load_test_data(os.path.join(args.data_dir, "test"))
    src_tokenizer = BertTokenizer.from_pretrained(args.encoder_model_name)
    tgt_tokenizer = BertTokenizer.from_pretrained(args.decoder_model_name)

    print("创建数据加载器...")
    test_dataset = MTQETestDataset(
        test_src, test_tgt, src_tokenizer, tgt_tokenizer, args.max_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print("加载模型...")
    model = MTQEModelWithPretrained(
        args.encoder_model_name, args.decoder_model_name, args.num_classes
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    print("开始测试...")
    predictions = test(model, test_loader, args.device)

    print("保存预测结果...")
    save_predictions(predictions, args.output_file)


def save_predictions(predictions, output_file):
    LABEL_MAP_INV = {0: 'OK', 1: 'minor', 2: 'major', 3: 'critical'}
    TAG_MAP_INV = {0: 'OK', 1: 'BAD', 2: 'BAD', 3: 'BAD'}

    dtag_output_file = os.path.join(output_file, "test.zh-en.dtag")
    tag_output_file = os.path.join(output_file, "test.zh-en.tag")
    score_output_file = os.path.join(output_file, "test.mqm_score")

    with open(dtag_output_file, "w") as f:
        for pred in predictions:
            mapped_pred = [LABEL_MAP_INV[label] for label in pred]
            f.write(" ".join(mapped_pred) + "\n")
    print(f"dtag预测结果已保存到 {dtag_output_file}")

    with open(tag_output_file, "w") as f:
        for pred in predictions:
            mapped_pred = [TAG_MAP_INV[label] for label in pred]
            f.write(" ".join(mapped_pred) + "\n")
    print(f"tag预测结果已保存到 {tag_output_file}")

    mqm_scores = []

    for pred in predictions:
        n = len(pred)
        n_minor = sum(1 for label in pred if label == 1)
        n_major = sum(1 for label in pred if label == 2)
        n_critical = sum(1 for label in pred if label == 3)
        mqm_score = 1 - (n_minor + 5 * n_major + 10 * n_critical) / n
        mqm_scores.append(mqm_score)

    with open(score_output_file, "w") as f:
        for score in mqm_scores:
            f.write(f"{score}\n")
    print(f"mqm_score预测结果已保存到 {score_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./resource/data", help="数据文件夹根目录"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/best_model.pth",
        help="模型路径",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results",
        help="输出文件路径",
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
    parser.add_argument("--max_len", type=int, default=256, help="序列最大长度")
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="设备",
    )
    parser.add_argument("--gpu", type=int, default=0, help="使用的GPU")
    args = parser.parse_args()

    run_test(args)
