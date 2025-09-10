# 将文件路径的点号/斜杠语法改为模块的点号语法
export PYTHONPATH=$PYTHONPATH:./src


----------------------------------------------------------------
python ./preprocess.py ../dataset/ ../gpt_models/

重写为：
python -m gpt_base.preprocess_torch \
    ./dataset/original_data/ \
    ./dataset/processed_data_torch/ \
    ./gpt_models/
-------------------------------------------------------------------
**训练新模型**:
    ```bash
    python -m gpt_base.main_torch \
        --mode train \
        --data_path ./dataset/original_data \
        --output_path ./output_gpt_torch \
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
-----------------------------------------------------------------------------------------------------------

inference.py --mode test --saved_model_path ../output_gpt/03_fix_l2t_20220519180620/saved_model/loads/14/


##任务进展
2025/07/21
运行了以下命令：
**训练新模型**:
    ```bash
    python -m gpt_base.main_torch \
        --mode train \
        --data_path ./dataset/original_data \
        --output_path ./output_gpt_torch \
        --model_save_name gpt2_logic2text \
        --model_name gpt2 \
        --num_epochs 3 \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 5e-5
    ```

-----------------------------------------------------------------------------------
IRNet 是一个用于语义分析的神经网络模型，其核心思想是将一个复杂的预测任务分解为两个更简单的阶段。
IRNet 的工作流程如下：
编码器 (Encoder)：使用一个 Transformer Encoder 来编码输入信息，这些信息包括自然语言问题（topics）以及数据库的结构信息（tables, column_names, values）。
两阶段解码器 (Two-Stage Decoder)：
第一阶段：生成“骨架”(Sketch Generation)：使用一个名为 sketch_decoder_lstm 的 LSTM 解码器，先生成目标逻辑形式（Logical Form）的高层结构或骨架。这个骨架定义了查询的整体结构，但不包含具体的列名或值。
第二阶段：填充细节 (Leaf-Node Prediction)：在生成骨架后，使用另一个名为 lf_decoder_lstm 的 LSTM 解码器和指针网络 (Pointer Networks)，来填充骨架中的叶子节点。这些叶子节点通常是数据库中的具体列（C）、值（V）或索引（I）。
IRNet 是一个采用“先生成骨架，再填充细节”策略的序列到序列（Seq2Seq）模型，专门用于处理与结构化数据（如数据库表）相关的自然语言理解和生成任务。这种分解方法旨在降低直接生成复杂逻辑形式的难度。

table2logic:
训练table2logic模型：
python ./src/main.py --batch_size 8 --beam_size 8 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --cuda --include_oov_token --rejection_sampling
需要用到数据集：/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Table2Logic/data/Logic2Text/original_data_fix

模型推理：
./src/main_inference.py --beam_size 2048 --value_cases_in_extra "case1a;case1b;case2;case3" --masked_cases "" --model_to_load_path "experiments/exp_1a-1b-2-3_RS/best_model.pt" --cuda --include_oov_token --rejection_sampling
需要用到数据集：/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Table2Logic/data/Logic2Text/original_data_fix/test.json
生成推理文件：/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Table2Logic/inferences
/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Table2Logic/inferences/exp_2/test.text

logic2text:
进入 Logic2Text 项目，将生成的 test.json 和 test.text 文件复制到 dataset/original_data 中并运行
python ./gpt_base/preprocess.py ../dataset/ ../gpt_models/

文件：/root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/gpt_base/preprocess_torch.py
作用：
存储分词后的数据：脚本读取原始的 .json 文件，使用 GPT2Tokenizer 将文本字段（如 sent, topic, logic_str 等）转换成模型能够理解的数字ID序列（例如 text_ids, topic_ids），然后将这些ID序列存入 .jsonl 文件。
高效的数据加载：.jsonl (JSON Lines) 格式的特点是每一行都是一个独立的、完整的JSON对象。这使得在模型训练时可以逐行读取和处理数据，而不需要一次性将整个巨大的数据集加载到内存中，从而提高了效率并减少了内存消耗。

训练GPT2model：
python /root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/gpt_base/main_torch.py \
    --mode train \
    --data_path /root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/dataset/processed_data_torch \
    --output_path ../output_gpt_torch \
    --model_save_name gpt2_logic2text_run1 \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5


进行推理：

python ./inference_torch.py \
    --data_path ../dataset/original_data \
    --model_to_load ../output_gpt_torch/gpt2_logic2text_run1_20250823_200305/epoch_1 \
    --output_path ../output_inference_torch \
    --exp_name my_first_inference \
    --batch_size 8 \
    --beam_size 4

生成的文本将保存在 Logic2Text 项目中的 output_inference 文件夹中。
python /root/TlT-Automatic-Logical-Forms-improve-fidelity-in-Table-to-Text-generation/Logic2Text/gpt_base/inference_torch.py

item_0 the majority of the drivers in the 1995 marlboro mclaren mercedes mclaren mp4 / 10b mercedes v10 were from the united states .
item_1 most of the players in the 2008-09 olympics were from montreal scott deibert .
item_2 the average number of points scored per game in the 1995-96 season was 2.75 .
item_3 most of the games were played on november 3 , 1963 .

要训练

结果不确定



