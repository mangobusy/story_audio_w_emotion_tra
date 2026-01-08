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
ONNX_MODEL_PATH = "/root/autodl-tmp/CosyVoice/ckp/CosyVoice2-0.5B/speech_tokenizer_v2.onnx" 

# 2. 数据集根目录 (MsceneSpeech_train.jsonl 所在的文件夹)
# 假设结构:
# /root/autodl-tmp/MsceneSpeech/
#    ├── MsceneSpeech_train.jsonl
#    └── MsceneSpeech_audio/
DATA_ROOT = "/root/autodl-tmp/data/MsceneSpeech/"

# 3. 输入输出文件
INPUT_JSONL = os.path.join(DATA_ROOT, "data/MsceneSpeech_train.jsonl")
OUTPUT_JSONL = os.path.join(DATA_ROOT, "data/MsceneSpeech_with_audio_token.jsonl")
# =================================================

def encode_chinese_filename(path_str):
    new_path = ""
    for char in path_str:
        if '\u4e00' <= char <= '\u9fa5':
            code = hex(ord(char))[2:]
            new_path += f"#U{code}"
        else:
            new_path += char
    return new_path

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

def extract_token_onnx(session, audio_path, device):
    # 1. 加载音频
    wav, sr = torchaudio.load(audio_path)
    
    # 2. 强制重采样到 16000Hz
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    # 3. 单声道处理
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # wav shape is (1, Samples)
    
    # === 关键修改开始：提取 Whisper Mel 特征 ===
    # whisper.log_mel_spectrogram 需要输入 1D Tensor，且在这个设备上
    wav = wav.to(device)
    
    # 计算 Mel 频谱，n_mels=128 (对应报错里的 Expected: 128)
    # 输入 shape: (Samples,) -> 输出 shape: (128, Time_frames)
    mel_tensor = whisper.log_mel_spectrogram(wav[0], n_mels=128)
    
    # 转回 CPU 并增加 batch 维度
    # shape: (1, 128, Time_frames)
    mel_numpy = mel_tensor.unsqueeze(0).cpu().numpy()
    # === 关键修改结束 ===

    # 4. 准备长度数据
    # 这里使用的是频谱图的时间长度 (Time_frames)，不是原始音频采样点数
    feat_len = mel_numpy.shape[2]
    length_numpy = np.array([feat_len], dtype=np.int32)
    
    # 5. 获取输入层名称
    input_name_speech = session.get_inputs()[0].name # 'feats'
    input_name_length = session.get_inputs()[1].name # 'feats_length'
    
    # 6. 构造输入
    inputs = {
        input_name_speech: mel_numpy,
        input_name_length: length_numpy
    }
    
    # 7. 推理
    outputs = session.run(None, inputs)
    
    # 8. 处理输出
    codes = outputs[0]
    codes = np.squeeze(codes)
    
    if len(codes.shape) == 1:
        semantic_token = codes.tolist()
    else:
        semantic_token = codes[0].tolist()
    return semantic_token

def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"错误: 找不到输入文件 {INPUT_JSONL}")
        return

    session = load_onnx_session()
    if session is None: return

    # 准备 device 用于 whisper 特征提取 (GPU更快)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"开始处理，结果将写入: {OUTPUT_JSONL}")
    
    total_lines = sum(1 for _ in open(INPUT_JSONL, 'r', encoding='utf-8'))
    success_count = 0
    fail_count = 0

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines):
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                
                relative_path = data.get('key')
                if not relative_path: continue
                
                audio_full_path = os.path.join(DATA_ROOT, relative_path)
                
                if not os.path.exists(audio_full_path):
                    encoded_relative_path = encode_chinese_filename(relative_path)
                    encoded_full_path = os.path.join(DATA_ROOT, encoded_relative_path)
                    if os.path.exists(encoded_full_path):
                        audio_full_path = encoded_full_path
                    else:
                        fail_count += 1
                        continue
                
                # 传入 device 参数
                tokens = extract_token_onnx(session, audio_full_path, device)
                data['answer_cosyvoice_speech_token'] = tokens
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                success_count += 1
                
            except Exception as e:
                print(f"\n处理出错 ({relative_path}): {e}")
                fail_count += 1
                continue

    print(f"\n任务完成！成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    main()