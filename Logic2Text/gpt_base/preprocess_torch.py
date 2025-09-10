import os
import json
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer


# 假设这个自定义解析器仍然需要
# 注意：您需要确保这个导入路径在您的PYTHONPATH中，或者调整它
from gpt_base.table2logic.lf_parser import get_cased_values

def linearize_table(table_content: list) -> str:
    """
    将表格内容线性化为一个字符串。
    原始逻辑： " row 0 : a ; b ; c row 1 : d ; e ; f "
    """
    res = []
    for i, row in enumerate(table_content):
        row_str = f"row {i} : " + " ; ".join(row)
        res.append(row_str)
    return " ".join(res)

def build_values_extra(cased_values: dict, value_cases_in_extra: str) -> list:
    """根据指定的 case 从解析出的值中构建一个列表。"""
    values_extra = []
    if not value_cases_in_extra:
        return []
    for case in value_cases_in_extra.split(";"):
        if case in cased_values:
            values_extra.extend(cased_values[case])
    return values_extra

def preprocess_data(args):
    """
    主预处理函数。
    读取原始JSON数据，使用Hugging Face Tokenizer进行编码，
    并将处理后的数据保存为JSON Lines格式。
    """
    print(f"从 '{args.tokenizer_path}' 加载 GPT-2 分词器...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        print(f"错误：无法加载分词器。请确保路径 '{args.tokenizer_path}' "
              f"包含有效的 GPT-2 分词器文件 (vocab.json, merges.txt)。\n{e}")
        return

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录 '{args.output_dir}' 已准备就绪。")

    for split in ["train", "valid", "test"]:
        input_file = os.path.join(args.input_dir, f"{split}.json")
        output_file = os.path.join(args.output_dir, f"{split}.jsonl")

        if not os.path.exists(input_file):
            print(f"警告：未找到输入文件 '{input_file}'，跳过该部分。")
            continue

        print(f"正在处理 '{input_file}'...")
        
        processed_samples = []
        with open(input_file, 'r', encoding='utf-8') as f_in:
            original_data = json.load(f_in)

            for sample in tqdm(original_data, desc=f"Processing {split}"):
                # 提取和转换文本字段
                text = sample.get("sent", "")
                topic = sample.get("topic", "")
                logic_interpret = sample.get("interpret", "")
                logic_str = sample.get("logic_str", "")
                header = " ; ".join(sample.get("table_header", []))
                table = linearize_table(sample.get("table_cont", []))

                # 处理额外的逻辑值
                cased_values = get_cased_values(logic_str)
                values_extra = build_values_extra(cased_values, args.value_cases)
                values_extra_str = " ; ".join(values_extra)

                # 使用分词器编码所有字段
                processed_sample = {
                    "text": text,
                    "text_ids": tokenizer.encode(text),
                    "topic_ids": tokenizer.encode(topic),
                    "logic_interpret_ids": tokenizer.encode(logic_interpret),
                    "logic_str_ids": tokenizer.encode(logic_str),
                    "header_ids": tokenizer.encode(header),
                    "table_ids": tokenizer.encode(table),
                    "extra_values_ids": tokenizer.encode(values_extra_str)
                }
                processed_samples.append(processed_sample)

        # 将处理后的样本写入JSON Lines文件
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for sample in processed_samples:
                f_out.write(json.dumps(sample) + '\n')
        
        print(f"已成功将处理后的数据写入 '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="使用Hugging Face Tokenizer为Logic2Text进行数据预处理。")
    parser.add_argument(
        "input_dir",
        type=str,
        help="包含原始 train.json, valid.json, test.json 的目录。"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="用于保存处理后的 .jsonl 文件的目录。"
    )
    parser.add_argument(
        "tokenizer_path",
        type=str,
        help="包含 GPT-2 分词器文件（例如，从Hugging Face下载的'gpt2'模型目录）的路径。"
    )
    parser.add_argument(
        "--value_cases",
        type=str,
        default="case1a;case1b;case2;case3",
        help="需要从逻辑形式中提取的值类型，用分号分隔。"
    )
    
    args = parser.parse_args()
    preprocess_data(args)
    print("预处理完成！")

if __name__ == '__main__':
    main()
