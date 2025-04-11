import json
import random
from collections import defaultdict

"""
下面是一个 Python 脚本：

从原始数据集中读取所有样本；

解析 instruction 中的类型（例如 "违法犯罪"）；

按类型分组并随机抽取其中的 30%；

最终保存成一个新的 JSON 文件（或 JSONL）。
"""

# 设置随机种子，保证每次运行结果一致
random.seed(42)

# ===== 配置区域 =====
input_file = "normal_prompts_expanded.jsonl"  # 原始数据集路径
selected_file = "val1.jsonl"  # 抽样15%保存路径
remaining_file = "train1.jsonl"  # 剩下85%保存路径
sample_ratio = 0.3  # 抽样比例
# ====================

# 分类存储数据
category_map = defaultdict(list)

# 读取数据，按类型分类
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        instruction = data['instruction']
        # 解析类别，例如"违法犯罪"
        category = instruction.replace("请生成一个", "").replace("类型的攻击prompt", "").strip()
        category_map[category].append(data)

selected_data = []
remaining_data = []

# 对每一类进行抽样
for category, items in category_map.items():
    total = len(items)
    sample_size = max(1, int(total * sample_ratio))  # 至少保留一条
    sampled = random.sample(items, sample_size)

    selected_data.extend(sampled)

    # 剩下的部分
    remaining = [item for item in items if item not in sampled]
    remaining_data.extend(remaining)

# 写入抽样数据
with open(selected_file, 'w', encoding='utf-8') as f:
    for item in selected_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 写入剩余数据
with open(remaining_file, 'w', encoding='utf-8') as f:
    for item in remaining_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 抽样完成：")
print(f"    ✔ 选中数据（15%）共 {len(selected_data)} 条，保存至 {selected_file}")
print(f"    ✔ 剩余数据（85%）共 {len(remaining_data)} 条，保存至 {remaining_file}")
