import argparse
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW  # <--- 从这里导入 AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_scheduler # <--- 从这里移除 AdamW
import evaluate
from tqdm.auto import tqdm

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """使用 argparse 解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-2 based Table-to-Text generation with PyTorch")

    # --- 路径和模式 ---
    parser.add_argument("--data_path", type=str, default="../dataset/original_data", help="Path to the dataset directory with .json files")
    parser.add_argument("--output_path", type=str, default="../output_gpt_torch", help="Root directory to save outputs")
    parser.add_argument("--model_save_name", type=str, default="gpt2_t2t_model", help="Subdirectory name for saving the model")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], required=True, help="Run mode: train or test")
    parser.add_argument("--model_to_load", type=str, default=None, help="Path to a saved model checkpoint for testing or resuming training")

    # --- 模型和分词器 ---
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base GPT-2 model from Hugging Face (e.g., gpt2, gpt2-medium)")

    # --- 数据处理参数 ---
    parser.add_argument("--max_input_len", type=int, default=512, help="Max length for model input (table + text)")
    parser.add_argument("--max_target_len", type=int, default=100, help="Max length for generated text during inference")

    # --- 训练超参数 ---
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation to simulate larger batch size")

    # --- 解码参数 ---
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for beam search decoding")

    return parser.parse_args()

class TableTextDataset(Dataset):
    """
    自定义PyTorch数据集。
    从预处理的 .jsonl 文件加载数据。
    """
    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        logging.info(f"Loading data from {jsonl_path}")
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        # GPT-2 需要一个 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 从预处理文件中获取编码后的ID
        table_ids = item['table_ids']
        text_ids = item['text_ids']
        
        # 拼接: [table_ids] <eos> [text_ids] <eos>
        input_ids_list = table_ids + [self.tokenizer.eos_token_id] + text_ids + [self.tokenizer.eos_token_id]
        
        # 创建标签，只在文本部分计算损失
        # 用-100填充非目标部分（表格部分）
        labels_list = [-100] * (len(table_ids) + 1) + text_ids + [self.tokenizer.eos_token_id]

        # 截断或填充到max_len
        input_ids = torch.full((self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        labels = torch.full((self.max_len,), -100, dtype=torch.long) # -100被损失函数忽略

        seq_len = len(input_ids_list)
        if seq_len > self.max_len:
            seq_len = self.max_len
        
        input_ids[:seq_len] = torch.tensor(input_ids_list[:seq_len], dtype=torch.long)
        attention_mask[:seq_len] = 1
        labels[:seq_len] = torch.tensor(labels_list[:seq_len], dtype=torch.long)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def train(args, model, tokenizer, train_dataloader, eval_dataloader, device, output_dir):
    """训练循环"""
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = args.num_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # 支持梯度累积
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1} | Average Training Loss: {avg_loss:.4f}")
        
        # 每个epoch后进行评估
        # logging.info(f"--- Evaluating after Epoch {epoch+1} ---")
        # run_evaluate(args, model, tokenizer, eval_dataloader, device)
        
        # 保存模型检查点
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)
        logging.info(f"Model checkpoint saved to {epoch_output_dir}")

def run_evaluate(args, model, tokenizer, dataloader, device):
    """评估函数，计算BLEU和ROUGE"""
    logging.info("Loading BLEU and ROUGE metrics...")

    # 直接通过名字加载，evaluate库会自动处理
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        # labels现在包含了-100，我们需要原始的input_ids来解码参考文本
        # 我们将从原始的input_ids中提取标签
        labels_with_padding = batch["labels"].clone()

        with torch.no_grad():
            # 生成时只使用表格部分作为输入
            # 找到每个样本中第一个eos_token的位置，作为输入的结束
            # 这标志着表格部分的结束
            input_lengths = [torch.where(ids == tokenizer.eos_token_id)[0][0] + 1 for ids in input_ids]
            
            # 截断输入以只包含表格部分
            prompt_ids_list = [ids[:length] for ids, length in zip(input_ids, input_lengths)]
            
            # 为了批处理，需要将它们填充到相同的长度
            padded_prompts = tokenizer.pad(
                {"input_ids": prompt_ids_list},
                padding=True,
                return_tensors="pt"
            ).to(device)

            generated_ids = model.generate(
                **padded_prompts,
                max_new_tokens=args.max_target_len,
                num_beams=args.beam_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 解码生成的文本（只解码新生成的部分）
        output_ids = generated_ids[:, padded_prompts.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        # 解码参考文本（只解码文本部分）
        labels_with_padding[labels_with_padding == -100] = tokenizer.pad_token_id
        labels_text = tokenizer.batch_decode(labels_with_padding, skip_special_tokens=True)
        
        all_preds.extend(preds)
        all_labels.extend(labels_text)

    # 计算指标
    bleu_score = bleu_metric.compute(predictions=all_preds, references=[[label] for label in all_labels])
    rouge_score = rouge_metric.compute(predictions=all_preds, references=all_labels)

    logging.info(f"BLEU score: {bleu_score['bleu']:.4f}")
    logging.info(f"ROUGE scores: {rouge_score}")
    
    return bleu_score, rouge_score

def main():
    args = parse_args()

    # --- 创建唯一的输出目录 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f"{args.model_save_name}_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"All outputs will be saved to: {output_dir}")

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 加载分词器和模型 ---
    model_path = args.model_to_load if args.mode == 'test' or (args.mode == 'train' and args.model_to_load) else args.model_name
    logging.info(f"Loading model and tokenizer from '{model_path}'...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # --- 根据模式执行操作 ---
    if args.mode == 'train':
        train_dataset = TableTextDataset(os.path.join(args.data_path, 'train.jsonl'), tokenizer, args.max_input_len)
        eval_dataset = TableTextDataset(os.path.join(args.data_path, 'valid.jsonl'), tokenizer, args.max_input_len)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)
        
        train(args, model, tokenizer, train_dataloader, eval_dataloader, device, output_dir)

    elif args.mode == 'test':
        if not args.model_to_load:
            raise ValueError("Must provide --model_to_load for testing.")
        
        test_dataset = TableTextDataset(os.path.join(args.data_path, 'test.jsonl'), tokenizer, args.max_input_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

        logging.info("--- Running Final Test Evaluation ---")
        run_evaluate(args, model, tokenizer, test_dataloader, device)


if __name__ == "__main__":
    main()

'''
### 如何运行新的脚本

1.  **训练新模型**:
    ```bash
    python ./main_torch.py \
        --mode train \
        --data_path ../dataset/original_data \
        --output_path ../output_gpt_torch \
        --model_save_name gpt2_logic2text \
        --model_name gpt2 \
        --num_epochs 3 \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 5e-5
    ```

2.  **从检查点恢复训练**:
    假设训练中断，最好的模型保存在 `../output_gpt_torch/gpt2_logic2text_.../epoch_1`。
    ```bash
    python ./main_torch.py \
        --mode train \
        --model_to_load ../output_gpt_torch/gpt2_logic2text_.../epoch_1 \
        --data_path ../dataset/original_data \
        --output_path ../output_gpt_torch \
        --model_save_name gpt2_logic2text_resume \
        --num_epochs 2 # 再训练2个epoch
    ```

3.  **测试已训练好的模型**:
    假设训练完成，最好的模型保存在 `../output_gpt_torch/gpt2_logic2text_.../epoch_3`。
    ```bash
    python ./main_torch.py \
        --mode test \
        --model_to_load ../output_gpt_torch/gpt2_logic2text_.../epoch_3 \
        --data_path ../dataset/original_data \
        --beam_size 4 \
        --eval_batch_size 8
    ```# filepath: /root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/gpt_base/main_torch.py
'''