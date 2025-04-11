import json

# instruction关键词 -> 标签映射（请根据你的数据实际情况自定义）
label_map = {
    "违法犯罪": 0,
    "恐怖主义": 1,
    "歧视偏见": 2,
    "侵犯权益": 3,
    "反面诱导": 4,
    "目标劫持": 5,
    "Prompt泄漏": 6,
    "不安全的指令主题": 7,
    "赋予角色后发指令": 8,
    "带有不安全观点的询问": 9,
    "偏见歧视": 10,
    "脏话侮辱": 11,
    "心理健康": 12,
    "身体伤害": 13,
    "财产隐私": 14,
    "道德伦理": 15,
    "正常类型": 16
}

def extract_label(instruction):
    for keyword, label in label_map.items():
        if keyword in instruction:
            return label
    return -1  # 未找到标签

def convert_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line.strip())
            label = extract_label(item["instruction"])
            if label == -1:
                print(f"⚠️ 无法识别标签，跳过：{item['instruction']}")
                continue
            new_item = {
                "text": item["output"],
                "label": label
            }
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

# 修改文件路径运行
if __name__ == "__main__":
    convert_dataset("test_data.jsonl", "test_data.jsonl")
