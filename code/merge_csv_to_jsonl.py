import pandas as pd
import ast  # 关键库：用于把 "[1, 2]" 这种字符串变成真正的列表 [1, 2]

def merge_csv_to_jsonl(csv_emotion, csv_audio_token, output_jsonl):
    print("1. 正在读取 CSV 文件...")
    
    # --- 关键修改点在这里 ---
    # 使用 converters 参数，读取时直接把字符串转为列表
    df_token = pd.read_csv(
        csv_audio_token, 
        converters={'audio_token': ast.literal_eval} 
    )
    # -----------------------
    
    df_emotion = pd.read_csv(csv_emotion)

    print("2. 正在合并数据...")
    # 提取需要的列
    df_emotion_subset = df_emotion[['text', 'V_pred', 'A_pred']]

    # 根据 ID 合并
    merged_df = pd.merge(df_emotion_subset, df_token, on='text', how='inner')

    print("3. 构建目标格式...")
    output_df = pd.DataFrame()

    # ID 转为整数
    output_df['key'] = merged_df['ID'].astype(int)

    # 文本内容
    output_df['source_text'] = merged_df['text']
    output_df['target_text'] = merged_df['text']

    # Emotion 组合成列表
    output_df['emotion'] = merged_df[['V_pred', 'A_pred']].values.tolist()

    # Audio Token (因为读取时已经转了，这里直接赋值就是列表)
    output_df['answer_cosyvoice_speech_token'] = merged_df['audio_token']

    # 导出文件名
    
    print(f"4. 正在写入 {output_jsonl} ...")
    
    # 写入 JSONL
    output_df.to_json(output_jsonl, orient='records', lines=True, force_ascii=False)

    print("完成！请检查结果。")
    
    # 打印一条数据验证类型
    first_token = output_df['answer_cosyvoice_speech_token'].iloc[0]
    print(f"验证数据类型: 第一条 token 的类型是 {type(first_token)} (应该是 <class 'list'>)")

if __name__ == "__main__":
    csv_emotion = '/root/autodl-tmp/data/MsceneSpeech/data/MsceneSpeech_train_emotion.csv'
    csv_audio_token = '/root/autodl-tmp/data/MsceneSpeech/data/MsceneSpeech_with_audio_token1.csv'
    output_jsonl = '/root/autodl-tmp/data/MsceneSpeech/data/MsceneSpeech_train_11111.jsonl'

    merge_csv_to_jsonl(csv_emotion, csv_audio_token, output_jsonl)