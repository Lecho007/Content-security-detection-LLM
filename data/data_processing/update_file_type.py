import json
import csv

# 类型映射字典：英文 -> 中文
type_mapping = {
    "Insult": "脏话侮辱",
    "Unfairness_And_Discrimination": "偏见歧视",
    "Crimes_And_Illegal_Activities": "违法犯罪",
    "Physical_Harm": "身体伤害",
    "Mental_Health": "心理健康",
    "Privacy_And_Property": "财产隐私",
    "Ethics_And_Morality": "道德伦理"
}

# 读取 JSON 文件
with open('typical_safety_scenarios.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 输出 CSV 文件名
output_file = '../Safety-Prompts-main/typical_safety_scenarios.csv'

# 写入 CSV
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'prompt', 'type'])  # 写入表头

    idx = 1  # id 计数器
    for type_key, entries in data.items():
        for item in entries:
            prompt = item.get('prompt', '')
            raw_type = item.get('type', type_key)
            mapped_type = type_mapping.get(raw_type, raw_type)  # 如果没找到就保留原样
            writer.writerow([idx, prompt, mapped_type])
            idx += 1

print(f"✅ 已保存 CSV 文件：{output_file}")
