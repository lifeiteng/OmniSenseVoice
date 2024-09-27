import time

import torch
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

from omnisense import OmniSenseVoiceSmall, OmniTranscription

model_dir = "iic/SenseVoiceSmall"
model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False, device_id=1)

# inference
wav_or_scp = ["tests/data/example.wav"]
for textnorm in ["woitn", "withitn"]:
    print(f"\n====== Text normalization: {textnorm} ======")
    start_time = time.time()
    res = model(wav_or_scp, language="auto", textnorm=textnorm)
    print(f"Time cost: {time.time() - start_time:.2f}s")

    print(res)
    print([rich_transcription_postprocess(i) for i in res])


model_dir = "iic/SenseVoiceSmall"
model = OmniSenseVoiceSmall(model_dir, quantize=False, device_id=0 if torch.cuda.is_available() else -1)


# inference
wav_or_scp = ["tests/data/Laughter.wav", "tests/data/Cry.wav", "tests/data/Applause.wav"]
for textnorm in ["woitn", "withitn"]:
    print(f"\n====== Text normalization: {textnorm} ======")
    start_time = time.time()
    res = model.transcribe(wav_or_scp, language="auto", textnorm=textnorm, timestamps=True)
    print(f"Time cost: {time.time() - start_time:.2f}s")
    print(res)


wav_or_scp = ["tests/data/example.wav"]
for textnorm in ["woitn", "withitn"]:
    print(f"\n====== Text normalization: {textnorm} ======")
    start_time = time.time()
    res = model.transcribe(wav_or_scp, language="auto", textnorm=textnorm, timestamps=True)
    print(f"Time cost: {time.time() - start_time:.2f}s")
    print(res)

