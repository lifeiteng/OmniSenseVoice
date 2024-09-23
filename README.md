# Omni SenseVoice

Wrapper of [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), optimized the inference speed and extended the ability to get timestamps.

## Benchmark

| Optimize       | WER ⬇️ | RTF ⬇️ |
| -----          |-----   | ----- |
| --batch-size 10| 1.17% | 0.12 |



```
# LibriTTS
DIR=benchmark/data
lhotse download libritts -p dev-clean benchmark/dataLibriTTS
lhotse prepare libritts -p dev-clean benchmark/data/LibriTTS/LibriTTS benchmark/data/manifests/libritts

lhotse cut simple --force-eager -r benchmark/data/manifests/libritts/libritts_recordings_dev-clean.jsonl.gz \
    -s benchmark/data/manifests/libritts/libritts_supervisions_dev-clean.jsonl.gz \
    benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl

omnisense benchmark -s -d --num-workers 1 --device-id 1 --batch-size 10 --textnorm woitn --language en benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl

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
