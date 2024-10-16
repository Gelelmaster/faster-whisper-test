import sounddevice as sd
import numpy as np
import io
import wave
from faster_whisper import WhisperModel, BatchedInferencePipeline

def fasterwhisper_recognize():
    # 设置录音参数
    fs = 44100  # 采样率

    # 初始化音频缓冲区
    audio_buffer = []
    silence_duration = 3  # 静默时长（秒）
    silence_threshold = 0.02  # 静默阈值（增大该值可避免误检测）
    silent_samples = 0  # 静默样本计数

    print("等待语音输入中...")

    # 开始录制音频
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
        while True:
            # 从输入流中读取音频
            data = stream.read(int(fs * 0.1))[0]  # 每次读取0.1秒的音频
            audio_buffer.append(data)  # 将数据添加到音频缓冲区

            # 检查是否存在有效声音
            if np.max(np.abs(data)) < silence_threshold:
                silent_samples += 1  # 增加静默样本计数
            else:
                silent_samples = 0  # 重置静默样本计数

            # 如果静默时长超过设定值，则停止录音
            if silent_samples > (silence_duration / 0.1):  # 计算静默样本数量
                break

    # 将音频缓冲区转换为 NumPy 数组
    audio_data = np.concatenate(audio_buffer)

    # 检查录音数据是否包含有效声音
    if np.max(np.abs(audio_data)) < silence_threshold:
        return None  # 返回 None 表示没有有效输入

    # 使用 BytesIO 创建一个内存中的文件
    audio_bytes = io.BytesIO()

    # 保存录制的音频到内存中的 BytesIO 对象
    with wave.open(audio_bytes, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 采样宽度为2字节（16位）
        wf.setframerate(fs) # 采样率
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())  # 转换为 int16

    # 将指针移动到 BytesIO 的开始位置
    audio_bytes.seek(0)

    # 初始化模型
    model = WhisperModel("medium", device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)

    # 使用录制的音频进行转录
    segments, info = batched_model.transcribe(
        audio_bytes,  # 直接使用 BytesIO 对象
        batch_size=16,
        initial_prompt="这是一个对话，内容包含问答形式。"
    )

    # 拼接转录的文本
    transcribed_text = "".join([segment.text for segment in segments])

    # 返回转录的文本
    return transcribed_text

def main():
    try:
        while True:
            # 调用识别函数并获取转录文本
            result = fasterwhisper_recognize()
            # 打印转录结果
            if result:  # 只有当 result 不是 None 时才打印
                print("识别结果:", result)

    except KeyboardInterrupt:
        print("程序已退出。")

# 调用主函数
if __name__ == "__main__":
    main()
