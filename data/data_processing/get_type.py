import pandas as pd

"""
获取数据中的类型然后去重
"""

def deduplicate_violation_types(input_file, output_file):
    # 读取CSV数据
    df = pd.read_csv(input_file)

    # 提取并去重 `违规类型` 列
    unique_violation_types = df['type'].drop_duplicates().reset_index(drop=True)

    # 将去重后的数据保存到新的 CSV 文件
    unique_violation_types.to_csv(output_file, index=False, header=["违规类型"])

    print(f"去重后的违规类型已保存至 {output_file}")


# 示例：调用函数，传入输入文件和输出文件路径
input_file = 'typical_safety_scenarios.csv'  # 请替换为你的CSV文件路径
output_file = 'violation_types_4.csv'  # 输出文件路径

deduplicate_violation_types(input_file, output_file)
