import sounddevice as sd
import numpy as np
import io
import wave
from faster_whisper import WhisperModel, BatchedInferencePipeline

# 设置录音参数
duration = 5  # 录音时长（秒）
fs = 44100    # 采样率

# 录制音频
print("等待语音输入中...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()  # 等待录制结束

# 使用 BytesIO 创建一个内存中的文件
audio_buffer = io.BytesIO()

# 保存录制的音频到内存中的 BytesIO 对象
with wave.open(audio_buffer, 'wb') as wf:
    wf.setnchannels(1)  # 单声道,2为立体声
    wf.setsampwidth(2)  # 采样宽度为2字节（16位）
    wf.setframerate(fs) # 采样率，每秒钟取样的次数，通常使用赫兹（Hz）作为单位
    wf.writeframes((recording * 32767).astype(np.int16).tobytes())  # 转换为 int16

# 将指针移动到 BytesIO 的开始位置
audio_buffer.seek(0)

# 初始化模型
model = WhisperModel("medium", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

# 使用录制的音频进行转录
segments, info = batched_model.transcribe(
    audio_buffer,  # 直接使用 BytesIO 对象
    batch_size=16,
    initial_prompt="这是一个对话。"
)

# 打印转录的文本
for segment in segments:
    print(segment.text)