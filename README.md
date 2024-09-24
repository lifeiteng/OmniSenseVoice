# Omni SenseVoice 🚀

## The Ultimate Speech Recognition Solution
Built on [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), Omni SenseVoice is optimized for lightning-fast inference and precise timestamps—giving you a smarter, faster way to handle audio transcription!

## Install
```
pip install .
```

## Usage
```
omnisense transcribe [OPTIONS] AUDIO_PATH
```
Key Options:
* `--language`: Automatically detect the language or specify (`auto, zh, en, yue, ja, ko`).
* `--textnorm`: Choose whether to apply inverse text normalization (`withitn for inverse normalized` or `woitn for raw`).
* `--device-id`: Run on a specific GPU (default: -1 for CPU).
* `--quantize`: Use a quantized model for faster processing.
* `--help`: Display detailed help information.

## Benchmark
`omnisense benchmark -s -d --num-workers 2 --device-id 0 --batch-size 10 --textnorm woitn --language en benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl`

| Optimize       | WER ⬇️  | RTF ⬇️ | Speed Up 🔥 |
| -----          |-----   | ----- |  ----- |
| baseline(onnx) | 4.47%  | 0.1200 |  1x   |
| torch          | 5.02%  | 0.0022 | 50x   |

* With Omni SenseVoice, experience up to 50x faster processing without sacrificing accuracy.


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

omnisense benchmark -s --num-workers 4 --device-id 0 --batch-size 16 --textnorm woitn --language en benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl
```

## Contributing 🙌
#### Step 1: Code Formatting
Set up pre-commit hooks:
```
pip install pre-commit==3.6.0
pre-commit install
```

#### Step 2: Pull Request
Submit your awesome improvements through a PR. 😊
