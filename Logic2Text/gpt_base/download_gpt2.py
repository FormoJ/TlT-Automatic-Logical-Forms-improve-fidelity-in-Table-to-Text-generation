import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_gpt2_model():
    """
    下载原始的 GPT-2 预训练模型和分词器，并保存到指定目录。
    """
    # 指定模型名称和保存路径
    model_name = "gpt2"
    save_directory = "/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/gpt_models/"

    # 确保保存目录存在
    os.makedirs(save_directory, exist_ok=True)
    print(f"保存目录 '{save_directory}' 已准备就绪。")

    # 下载并加载分词器
    print(f"正在从 Hugging Face Hub 下载 '{model_name}' 的分词器...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("分词器下载成功。")
    except Exception as e:
        print(f"下载分词器时出错: {e}")
        return

    # 下载并加载模型
    print(f"正在从 Hugging Face Hub 下载 '{model_name}' 的预训练模型...")
    print("这可能需要一些时间，具体取决于您的网络连接。")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print("模型下载成功。")
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return

    # 保存模型和分词器到本地目录
    print(f"正在将模型和分词器保存到 '{save_directory}'...")
    try:
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print("模型和分词器已成功保存！")
        print(f"文件列表: {os.listdir(save_directory)}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    download_gpt2_model()