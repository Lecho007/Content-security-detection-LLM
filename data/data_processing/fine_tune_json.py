import pandas as pd
import json

"""
    将原始JSON或CSV数（含 type 和 prompt）转换为微调用的 JSONL格式。
"""

def convert_to_jsonl(input_file, output_file):
    # 读取CSV或JSON数据
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.json_normalize(data)  # 如果是嵌套JSON，可能需要展开
    else:
        print("不支持的文件格式！")
        return

    # 目标字段：转换为模型需要的格式
    jsonl_data = []

    # 根据数据的具体结构进行转换
    for index, row in df.iterrows():
        # 提取 `type` 和 `prompt`
        # prompt = row['prompt']  # 适应数据的具体列名
        # attack_type = row['type']  # 适应数据的具体列名
        prompt = row['问题']  # 适应数据的具体列名
        attack_type = row['违规类型']  # 适应数据的具体列名

        # 根据攻击类型，构建 instruction 和 output 格式
        instruction = f"请生成一个{attack_type}类型的攻击prompt"
        output = prompt  # 输出就是原始的 prompt

        # 创建 JSON 对象
        jsonl_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    # 将数据保存为 JSONL 格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"转换完成，保存至 {output_file}")


# 示例：调用函数，传入输入文件和输出文件路径
input_file = 'jade_benchmark_medium_zh_expand.csv'  # 或者 'your_input_data.json'
output_file = 'jade_benchmark_medium_zh_expand_converted_data.jsonl'

convert_to_jsonl(input_file, output_file)
