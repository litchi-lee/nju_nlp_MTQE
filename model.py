import torch.nn as nn
from transformers import BertModel


class MTQEModelWithPretrained(nn.Module):
    def __init__(self, encoder_model_name, decoder_model_name, num_classes):
        super(MTQEModelWithPretrained, self).__init__()

        # 加载预训练的 BERT 模型作为编码器和解码器
        self.encoder = BertModel.from_pretrained(encoder_model_name)  # BERT encoder
        self.decoder = BertModel.from_pretrained(decoder_model_name)  # BERT decoder

        # 获取预训练模型的隐层维度
        d_model = self.encoder.config.hidden_size

        # 用于逐词分类的全连接层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(
        self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
    ):
        # 编码源句
        encoder_outputs = self.encoder(
            input_ids=src_input_ids, attention_mask=src_attention_mask, return_dict=True
        )
        src_encoded = (
            encoder_outputs.last_hidden_state
        )  # (batch_size, src_seq_len, d_model)

        # 解码目标句，同时使用交互注意力
        decoder_outputs = self.decoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            encoder_hidden_states=src_encoded,
            encoder_attention_mask=src_attention_mask,
            return_dict=True,
        )
        tgt_decoded = (
            decoder_outputs.last_hidden_state
        )  # (batch_size, tgt_seq_len, d_model)

        # 逐词分类
        token_outputs = self.fc(tgt_decoded)  # (batch_size, tgt_seq_len, num_classes)
        return token_outputs


if __name__ == "__main__":
    model = MTQEModelWithPretrained(
        "./pretrained/bert-base-chinese", "./pretrained/bert-base-uncased", 4
    )
    model = model.to("cuda")
    print(model)
