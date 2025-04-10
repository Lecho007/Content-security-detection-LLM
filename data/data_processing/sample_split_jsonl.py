import random

"""按组随机采样并拆分 JSONL 数据集"""

# ✅ 可调参数
input_file = ""               # 输入文件
sampled_file = ""         # 随机采样输出
remaining_file = ""     # 未采样输出

group_size = 10       # 每组多少行
sample_count = 2      # 每组采样多少行（可改成2、3...）

# 读取非空行
with open(input_file, "r", encoding="utf-8") as infile:
    lines = [line for line in infile if line.strip()]

# 分组
groups = [lines[i:i+group_size] for i in range(0, len(lines), group_size)]

sampled = []
remaining = []

for group in groups:
    if group:
        k = min(sample_count, len(group))  # 防止 group 太小
        chosen = random.sample(group, k)
        sampled.extend(chosen)
        for c in chosen:
            group.remove(c)
        remaining.extend(group)

# 写入文件
with open(sampled_file, "w", encoding="utf-8") as s_out:
    s_out.writelines(sampled)

with open(remaining_file, "w", encoding="utf-8") as r_out:
    r_out.writelines(remaining)

print(f"采样完成：每 {group_size} 行中抽取 {sample_count} 条")
print(f"→ 抽中 {len(sampled)} 条写入 {sampled_file}")
print(f"→ 剩余 {len(remaining)} 条写入 {remaining_file}")
