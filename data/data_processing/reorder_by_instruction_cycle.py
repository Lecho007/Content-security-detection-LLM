import json
from collections import defaultdict
from itertools import cycle

"""将数据集按 instruction 字段的类型轮流排列"""

input_path = "train.jsonl"
output_path = "train1.jsonl"

# 1. 分类存储所有数据
category_data = defaultdict(list)
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        category = item["instruction"]
        category_data[category].append(item)

# 2. 构造循环排序
category_keys = list(category_data.keys())
data_iters = {cat: iter(lst) for cat, lst in category_data.items()}
final_data = []

while True:
    added = False
    for cat in category_keys:
        try:
            final_data.append(next(data_iters[cat]))
            added = True
        except StopIteration:
            continue
    if not added:
        break

# 3. 写出新文件
with open(output_path, "w", encoding="utf-8") as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"已完成重排，共 {len(final_data)} 条数据。保存至：{output_path}")
