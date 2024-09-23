# Omni SenseVoice

Wrapper of [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), optimized the inference speed and extended the ability to get timestamps.

## Benchmark

| Optimize       | WER ⬇️  | RTF ⬇️ | Speed Up |
| -----          |-----   | ----- |  ----- |
| baseline(onnx) | 1.17%  | 0.1200 |  1x   |
| torch          | 1.23%  | 0.0046 | 26x   |


```
# LibriTTS
DIR=benchmark/data
lhotse download libritts -p dev-clean benchmark/dataLibriTTS
lhotse prepare libritts -p dev-clean benchmark/data/LibriTTS/LibriTTS benchmark/data/manifests/libritts

lhotse cut simple --force-eager -r benchmark/data/manifests/libritts/libritts_recordings_dev-clean.jsonl.gz \
    -s benchmark/data/manifests/libritts/libritts_supervisions_dev-clean.jsonl.gz \
    benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl

omnisense benchmark -s -d --num-workers 2 --device-id 0 --batch-size 10 -
-textnorm woitn --language en benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl
# Audio time: 2109.1703s Compute time: 9.6817s RTF: 0.0046 WER: 1.23%

omnisense benchmark -s --num-workers 4 --device-id 0 --batch-size 16 --textnorm woitn --language en benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl
# Audio time: 32292.7525s Compute time: 67.5100s RTF: 0.0021 WER: 1.81%
```

## Install
```
pip install .
```

## Contributing
#### step1: set code Formatting
```
pip install pre-commit==3.6.0
pre-commit install
```

#### step2: make a Pull Request
😊😊
