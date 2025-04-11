# 要合并的文件列表
files_to_merge = [
    "val.jsonl",
    "val1.jsonl",
]

# 输出文件名
output_file = "val_normal.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for file in files_to_merge:
        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if line:  # 跳过空行
                    outfile.write(line + "\n")

print(f"合并完成，共合并 {len(files_to_merge)} 个文件，输出文件为 {output_file}")
