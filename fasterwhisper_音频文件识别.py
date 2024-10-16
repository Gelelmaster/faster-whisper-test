# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")


# # 转录英语
# from faster_whisper import WhisperModel

# model_size = "distil-large-v3"

# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# segments, info = model.transcribe(r"d:\Desktop\project\faster-whisper\output.wav", beam_size=5, language="en", condition_on_previous_text=False)

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("medium", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

segments, info = batched_model.transcribe(
    r"d:\Desktop\project\faster-whisper\output.wav", 
    batch_size=16,
    initial_prompt = "这是一个对话。")

for segment in segments:
    print(segment.text)
