import pandas as pd

# 原始 Excel 文件路径
input_excel = '/root/autodl-tmp/data/stroy/stroy_example.xlsx'
# 目标 CSV 文件路径
output_csv = '/root/autodl-tmp/data/stroy/stroy_example.csv'

try:
    # 1. 读取 Excel 文件
    df = pd.read_excel(input_excel)
    
    # 2. 修正列名拼写错误 (stroy -> story)
    if 'stroy' in df.columns:
        print("发现列名拼写错误 'stroy'，正在修正为 'story'...")
        df.rename(columns={'stroy': 'story'}, inplace=True)
    
    # 3. 确保包含必要的列
    required_cols = ['ID', 'story', 'text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"错误：转换后的数据缺少列: {col}")

    # 4. 保存为标准的 CSV 文件 (UTF-8 编码，不带索引)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"成功！已将 Excel 转换为 CSV，并保存至: {output_csv}")
    print("前几行数据预览：")
    print(df.head())

except Exception as e:
    print(f"转换失败: {e}")
    # 如果提示缺少 openpyxl，请运行 pip install openpyxl