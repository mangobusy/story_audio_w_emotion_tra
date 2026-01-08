import json
import os
import torch
import torchaudio
import onnxruntime
import numpy as np
import whisper  # 新增：用于特征提取
from tqdm import tqdm
# ==================== 配置区域 ====================
# 1. 模型路径 (请修改为你上传的 onnx 文件路径)
ONNX_MODEL_PATH = "/root/autodl-tmp/CosyVoice/ckp/Cosyvoice-300M/models--FunAudioLLM--CosyVoice-300M/snapshots/f3ba236933d576582badded489545704c9b54799/speech_tokenizer_v1.onnx" 

# 2. 数据集根目录 (MsceneSpeech_train.jsonl 所在的文件夹)
# 假设结构:
# /root/autodl-tmp/MsceneSpeech/
#    ├── MsceneSpeech_train.jsonl
#    └── MsceneSpeech_audio/
DATA_ROOT = "/root/autodl-tmp/data/MsceneSpeech/"

# 3. 输入输出文件
INPUT_JSONL = os.path.join(DATA_ROOT, "data/MsceneSpeech_with_audio_token2.jsonl")
OUTPUT_JSONL = os.path.join(DATA_ROOT, "data/MsceneSpeech_with_audio_token1.jsonl")
# =================================================

# =================================================

def load_onnx_session(device_type='cuda'):
    print(f"正在加载 ONNX 模型: {ONNX_MODEL_PATH}")
    if device_type == 'cuda' and torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    try:
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        return session
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

# === 新增：模糊查找函数 (解决中文乱码/找不到文件问题) ===
def find_real_file_path(expected_path):
    # 1. 如果路径直接存在，直接返回
    if os.path.exists(expected_path):
        return expected_path
    
    # 2. 尝试旧的 #U 编码查找
    def encode_chinese(path_str):
        new_path = ""
        for char in path_str:
            if '\u4e00' <= char <= '\u9fa5':
                new_path += f"#U{hex(ord(char))[2:]}"
            else:
                new_path += char
        return new_path
    
    encoded_path = encode_chinese(expected_path)
    if os.path.exists(encoded_path):
        return encoded_path

    # 3. 智能模糊匹配 (通过文件ID查找)
    # 假设结构是 MsceneSpeech_audio/filename.mp3
    directory, filename = os.path.split(expected_path)
    
    # 如果目录(MsceneSpeech_audio)存在，进去找
    if os.path.exists(directory):
        try:
            # 提取文件后缀 ID (例如 "174.mp3")
            # 逻辑：取最后一段作为唯一标识
            if '_' in filename:
                file_id_suffix = filename.split('_')[-1] # 拿到 "174.mp3"
                
                # 遍历目录下所有文件
                for f in os.listdir(directory):
                    if f.endswith(file_id_suffix):
                        return os.path.join(directory, f)
        except:
            pass
            
    return None

# === 修正：使用 Whisper 提取 128维特征 (解决 INVALID_ARGUMENT 报错) ===
def extract_token_whisper_128(session, audio_path, device):
    # 1. 加载音频
    wav, sr = torchaudio.load(audio_path)
    
    # 2. 重采样
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    # 3. 单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # 4. 特征提取
    wav = wav.to(device)
    # log_mel_spectrogram 输出: (128, Time)
    mel_tensor = whisper.log_mel_spectrogram(wav[0], n_mels=128)
    
    # 5. 调整维度为 [1, 128, Time] (对应模型要求的 Expected: 128 at index 1)
    mel_numpy = mel_tensor.unsqueeze(0).cpu().numpy()
    
    # 6. 准备长度输入
    feat_len = mel_numpy.shape[2]
    length_numpy = np.array([feat_len], dtype=np.int32)
    
    # 7. 构造 ONNX 输入
    input_name_speech = session.get_inputs()[0].name 
    input_name_length = session.get_inputs()[1].name 
    
    inputs = {
        input_name_speech: mel_numpy,
        input_name_length: length_numpy
    }
    
    # 8. 推理
    outputs = session.run(None, inputs)
    codes = np.squeeze(outputs[0])
    
    if len(codes.shape) == 1:
        return codes.tolist()
    else:
        return codes[0].tolist() if len(codes) > 0 else []

def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"错误: 找不到输入文件 {INPUT_JSONL}")
        return

    # 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = load_onnx_session()
    if session is None: return

    print(f"开始处理...")
    print(f"输入: {INPUT_JSONL}")
    print(f"输出: {OUTPUT_JSONL}")
    
    success_count = 0
    fail_count = 0
    
    # 读取所有行
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 写入模式
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines):
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                
                key = data.get('key')
                if not key: continue
                
                # 拼接原始完整路径
                original_full_path = os.path.join(DATA_ROOT, key)
                
                # === 1. 找文件 (核心修改) ===
                real_path = find_real_file_path(original_full_path)
                
                if not real_path:
                    # 如果找不到，尝试手动构建可能的 #U 路径再试一次（作为双重保险）
                    dir_part, file_part = os.path.split(original_full_path)
                    if os.path.exists(dir_part):
                         # 最后的尝试：直接列出目录打印出来看看（调试用，可注释）
                         # print(f"找不到文件: {key}")
                         pass
                    fail_count += 1
                    continue
                
                # === 2. 提特征 (核心修改) ===
                tokens = extract_token_whisper_128(session, real_path, device)
                
                data['answer_cosyvoice_speech_token'] = tokens
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                success_count += 1
                
            except Exception as e:
                # print(f"处理出错: {e}") 
                fail_count += 1
                continue

    print(f"\n任务完成！成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    main()