
from huggingface_hub import snapshot_download
'''
# 直接将仓库文件下载到本地目录
local_dir = "/root/autodl-tmp/data/"
snapshot_download(
    repo_id="mythicinfinity/libritts_r",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 确保下载的是实体文件而不是链接
)

print(f"下载完成，数据保存在 {local_dir}")
'''

# # !pip install openai-whisper

# import os
# import torch
# import whisper

# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# model = whisper.load_model(
#     "large-v3",
#     device=device,
#     download_root="/root/autodl-tmp/EmoVoice/checkpoint"
# )


snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/root/autodl-tmp/CosyVoice/ckp/CosyVoice2-0.5B')
# snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='/root/autodl-tmp/CosyVoice/ckp/CosyVoice-ttsfrd')
