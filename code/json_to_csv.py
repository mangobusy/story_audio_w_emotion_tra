import json
import csv
import re
import os

# 1. 配置输入和输出文件名
root_path = "/root/autodl-tmp/data/MsceneSpeech/data"
input_file = root_path + '/MsceneSpeech_with_audio_token1.jsonl'  # 请修改为你的 jsonl 文件实际路径
output_file = root_path + '/MsceneSpeech_with_audio_token1.csv'    # 输出的 CSV 文件名

def process_mscene_data(input_path, output_path):
    # 用于存储故事数据：结构为 { "故事名": [ (序号, 文本), ... ] }
    stories_data = {}
    
    # 正则表达式解释：
    # MsceneSpeech_(train/test/dev)_story_  -> 匹配前缀和 dataset split
    # (.*?)                                 -> Group 1: 捕获故事名称 (非贪婪匹配)
    # _([^_]+)                              -> Group 2: 捕获说话人 (例如: 旁白, 拨云观星)
    # _(\d+)                                -> Group 3: 捕获句子序号
    # \.mp3                                 -> 匹配后缀
    pattern = re.compile(r"MsceneSpeech_(?:train|test|dev)_story_(.*?)_([^_]+)_(\d+)\.mp3")

    print("正在读取并解析数据...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                key = entry.get('key', '')
                text = entry.get('source_text', '') # 获取文本内容
                audio_token = entry.get('answer_cosyvoice_speech_token', '') # 获取音频 token
                
                # 过滤：只处理包含 _story_ 的音频
                if '_story_' not in key:
                    continue

                # 解析文件名
                match = pattern.search(key)
                if match:
                    story_name = match.group(1) # 例如：故事讲述文本二
                    # speaker = match.group(2)  # 例如：拨云观星 (目前逻辑不需要用到说话人，只要按序号排即可)
                    sentence_idx = int(match.group(3)) # 例如：333
                    
                    # 将数据存入字典
                    if story_name not in stories_data:
                        stories_data[story_name] = []
                    
                    stories_data[story_name].append((sentence_idx, text, audio_token))
                else:
                    print(f"警告: 无法解析的文件名格式 -> {key}")

            except json.JSONDecodeError:
                print(f"跳过无效的 JSON 行: {line[:50]}...")

    print(f"共找到 {len(stories_data)} 个不同的故事。正在排序并写入 CSV...")

    # 准备写入 CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'story', 'text','audio_token']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        global_id = 1 # 全局 ID 从 1 开始
        
        # 对故事名称进行排序，确保生成的 story1, story2 顺序是固定的
        sorted_story_names = sorted(stories_data.keys())

        for i, story_name in enumerate(sorted_story_names):
            # 生成 story ID (例如 story1, story2...)
            current_story_id = f"story{i+1}"
            
            # 获取该故事的所有句子
            sentences = stories_data[story_name]
            
            # 【关键步骤】按句子序号从小到大排序
            sentences.sort(key=lambda x: x[0])
            
            # 写入每一行
            for sent_idx, text, audio_token in sentences:
                writer.writerow({
                    'ID': global_id,
                    'story': current_story_id,
                    'text': text,
                    'audio_token': audio_token
                })
                global_id += 1

    print(f"处理完成！结果已保存至: {output_path}")
    print(f" - 总行数 (ID): {global_id - 1}")
    print(f" - 故事数量: {len(sorted_story_names)}")

# 运行函数
if __name__ == "__main__":
    # 检查输入文件是否存在
    if os.path.exists(input_file):
        process_mscene_data(input_file, output_file)
    else:
        print(f"错误: 找不到文件 {input_file}，请修改脚本中的 input_file 变量。")