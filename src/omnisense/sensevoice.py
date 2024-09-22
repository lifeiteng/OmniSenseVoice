#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import os.path
from pathlib import Path
from typing import List, Tuple, Union

import librosa
import numpy as np
import torch
from funasr_onnx.utils.frontend import WavFrontend
from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr_onnx.utils.utils import OrtInferSession, get_logger, read_yaml
from torch.utils.data import DataLoader, Dataset

logging = get_logger()


class OmniSenseVoiceSmall:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        **kwargs,
    ):

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download

            model_dir = snapshot_download(model_dir, cache_dir=cache_dir)

        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_file):
            print(".onnx does not exist, begin to export onnx")
            from funasr import AutoModel

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        config["frontend_conf"]["cmvn_file"] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)

        self.blank_id = 0
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, List[np.ndarray]],
        language: str = "auto",
        textnorm: str = "woitn",
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> List[str]:
        sampling_rate = self.frontend.opts.frame_opts.samp_freq

        if isinstance(audio, List):
            if isinstance(audio[0], str):
                audios = [librosa.load(path, sr=sampling_rate)[0] for path in audio]
            else:
                audios = audio
        else:
            pass

        dataset = NumpyDataset(audios)

        def collate_fn(batch):
            batch_size = len(batch)
            feats, feats_len = self.extract_feat(batch)
            return batch_size, feats, feats_len

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
        )

        results = [None] * len(audios)
        for batch in dataloader:
            batch_size, feats, feats_len = batch
            ctc_logits, encoder_out_lens = self.ort_infer(
                [
                    feats,
                    feats_len,
                    np.array([self.lid_dict[language]] * batch_size, dtype=np.int32),
                    np.array([self.textnorm_dict[textnorm]] * batch_size, dtype=np.int32),
                ]
            )
            encoder_out_lens = encoder_out_lens.tolist()

            ctc_logits = torch.from_numpy(ctc_logits).to(device="cpu")
            ctc_maxids = ctc_logits.argmax(dim=-1)

            for b in range(batch_size):
                yseq = ctc_maxids[b, : encoder_out_lens[b]]
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                results.append(self.tokenizer.decode(token_int))

        return results

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats


class NumpyDataset(Dataset):
    def __init__(self, segments: List[np.ndarray]):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]
