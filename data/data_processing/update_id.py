import pandas as pd
# 更新CSV文件的数据id
# 读取CSV文件
df = pd.read_csv('../jade-db/jade-db-v2.0/jade_benchmark_medium_zh_expand_1.csv')

# 重新排序原始ID列
df['ID'] = range(1, len(df) + 1)

# 保存新的CSV文件，保持原格式
df.to_csv('jade_benchmark_medium_zh_expand_1.csv', index=False, encoding='utf-8')

print("ID排序完成，结果已保存为 jade_benchmark_medium_zh_expand_1.csv")

