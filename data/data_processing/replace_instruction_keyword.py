import json

"""将 JSONL 文件中所有 "instruction" 字段里出现的 “侵害权益” 替换为 “侵犯权益”"""

input_path = "test_data.jsonl"               # ← 原始 JSONL 文件路径
output_path = "test_data.jsonl"  # ← 输出替换后的文件路径

# 读取并处理每一行
with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        if "instruction" in data and isinstance(data["instruction"], str):
            data["instruction"] = data["instruction"].replace("侵害权益", "侵犯权益")
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ 替换完成，输出文件保存在：{output_path}")
