import argparse
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """使用 argparse 解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-2 based Table-to-Text Inference with PyTorch")

    # 路径和模式
    parser.add_argument("--data_path", type=str, required=True, help="Path to the original dataset directory (containing test.json)")
    parser.add_argument("--output_path", type=str, default="../output_inference_torch", help="Path to save inference outputs")
    parser.add_argument("--model_to_load", type=str, required=True, help="Path to a saved model checkpoint directory")
    parser.add_argument("--exp_name", type=str, default="inference_run", help="Name for the experiment output folder")

    # 数据处理参数
    parser.add_argument("--max_input_len", type=int, default=512, help="Max length for model input")
    parser.add_argument("--max_target_len", type=int, default=100, help="Max length for generated text")

    # 推理超参数
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for beam search decoding")

    return parser.parse_args()

class TableTextInferenceDataset(Dataset):
    """为推理准备数据的自定义PyTorch数据集"""
    def __init__(self, json_path, tokenizer, max_input_len):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 仅使用表格作为输入来生成文本
        table_str = " ".join([" ".join(row) for row in item['table_cont']])
        input_text = table_str + self.tokenizer.eos_token
        
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        
        # 同时返回sha1以便后续匹配
        sha1 = item.get("sha1", f"item_{idx}")
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "sha1": sha1}

def run_inference(args, model, tokenizer, dataloader, device):
    """执行推理循环"""
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="Running Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sha1s = batch["sha1"]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_target_len,
                num_beams=args.beam_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 从生成结果中移除输入部分
        output_ids = generated_ids[:, input_ids.shape[1]:]
        
        # 解码生成的文本
        preds_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        for sha, text in zip(sha1s, preds_text):
            # 清理文本
            clean_text = text.replace("\n", " ").strip()
            if not clean_text:
                clean_text = "empty ."
            results.append({"sha1": sha, "text": clean_text})
            
    return results

def save_results(output_path, results):
    """将推理结果保存为多种格式"""
    # 1. 保存为纯文本文件
    plain_text_path = os.path.join(output_path, "inferred_texts.txt")
    with open(plain_text_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(item['text'] + '\n')
    logging.info(f"纯文本结果已保存至: {plain_text_path}")

    # 2. 保存为带哈希的CSV文件
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_path, "hashed_inferred_texts.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"CSV结果已保存至: {csv_path}")

    # 3. 保存为带哈希的文本文件
    hashed_text_path = os.path.join(output_path, "hashed_inferred_texts.txt")
    with open(hashed_text_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"{item['sha1']} {item['text']}\n")
    logging.info(f"带哈希的文本结果已保存至: {hashed_text_path}")

def main():
    args = parse_args()

    # 创建唯一的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f"{args.exp_name}_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"所有输出将保存至: {output_dir}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 加载分词器和模型
    logging.info(f"从 '{args.model_to_load}' 加载模型和分词器...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_to_load)
        model = GPT2LMHeadModel.from_pretrained(args.model_to_load)
        model.to(device)
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        return

    # 加载数据集
    test_json_path = os.path.join(args.data_path, "test.json")
    logging.info(f"加载测试数据从: {test_json_path}")
    test_dataset = TableTextInferenceDataset(test_json_path, tokenizer, args.max_input_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 运行推理
    inference_results = run_inference(args, model, tokenizer, test_dataloader, device)

    # 保存结果
    save_results(output_dir, inference_results)
    
    logging.info("推理完成！")

if __name__ == "__main__":
    main()


'''
```

### 如何运行新的脚本

假设您的目录结构如下：

```
/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/
├── dataset/
│   └── original_data/
│       └── test.json
├── output_gpt_torch/
│   └── gpt2_t2t_model_20250721.../  <-- 训练好的模型目录
│       └── epoch_3/                  <-- 具体的模型检查点
└── gpt_base/
    └── inference_torch.py
```

您可以使用以下命令来运行推理：

```bash
python ./inference_torch.py \
    --data_path ../dataset/original_data \
    --model_to_load ../output_gpt_torch/gpt2_t2t_model_20250721.../epoch_3 \
    --output_path ../output_inference_torch \
    --exp_name gpt2_test_run \
    --batch_size 8 \
    --beam_size 4
```

**命令解析:**

*   `--data_path`: 指向包含原始 `test.json` 的目录。
*   `--model_to_load`: **必需参数**。指向您之前训练并保存的 PyTorch 模型检查点目录（例如，`epoch_3` 目录）。
*   `--output_path`: 指定一个根目录来存放所有推理结果。
*   `--exp_name`: 本次推理运行的名称，脚本会自动创建一个带时间戳的子目录。
*   `--batch_size`, `--beam_size`: 控制推理时的批处理大小和束搜索宽度。

这个新脚本完全摆脱了 TensorFlow 的依赖，采用了现代 PyTorch 项目的编码规范，并且更加高效和易于理解。# filepath: /root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/gpt_base/inference_torch.py

'''