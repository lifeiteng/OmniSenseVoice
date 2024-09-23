#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os.path
import re
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from funasr_onnx.utils.frontend import WavFrontend
from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr_onnx.utils.utils import read_yaml
from lhotse.audio import Recording
from lhotse.cut import Cut, MultiCut
from lhotse.utils import Pathlike
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import SenseVoiceSmall


# modified from Lhotse AlignmentItem
class OmniTranscription(NamedTuple):
    """
    This class contains an alignment item, for example a word, along with its
    start time (w.r.t. the start of recording) and duration. It can potentially
    be used to store other kinds of alignment items, such as subwords, pdfid's etc.
    """

    language: str
    emotion: str
    event: str
    textnorm: str

    text: Optional[str] = None
    # start: Optional[float] = None
    # duration: Optional[float] = None

    # # Score is an optional aligner-specific measure of confidence.
    # # A simple measure can be an average probability of "symbol" across
    # # frames covered by the AlignmentItem.
    # score: Optional[float] = None

    @property
    def end(self) -> float:
        return round(self.start + self.duration, ndigits=8)

    @classmethod
    def parse(cls, input_string: str):
        """
        Parse from SenseVoice output.
        """
        # '<|nospeech|><|EMO_UNKNOWN|><|Laughter|><|woitn|>'
        # '<|en|><|NEUTRAL|><|Speech|><|withitn|>As you can see...'
        pattern = r"<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>(.*)?"
        match = re.match(pattern, input_string)
        if match:
            language, emotion, event, textnorm, text = match.groups()
            return cls(language, emotion, event, textnorm, text)
        else:
            raise ValueError(f"Cannot parse the input string: {input_string}")


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
        cache_dir: str = None,
    ):
        self.model, kwargs = SenseVoiceSmall.from_pretrained(model_dir, quantize=quantize)
        del kwargs

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download

            model_dir = snapshot_download(model_dir, cache_dir=cache_dir)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        config["frontend_conf"]["cmvn_file"] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.sampling_rate = self.frontend.opts.frame_opts.samp_freq

        self.device = "cpu"
        if device_id != "-1":
            assert torch.cuda.is_available(), "CUDA is not available"
            self.device = f"cuda:{device_id}"
        self.model.to(self.device)

        self.blank_id = 0
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, List[np.ndarray], List[Cut]],
        language: str = "auto",
        textnorm: str = "woitn",
        sort_by_duration: bool = True,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> List[OmniTranscription]:
        if isinstance(audio, List):
            indexs = list(range(len(audio)))
            if sort_by_duration:
                if isinstance(audio[0], Cut):
                    audios = sorted(zip(indexs, audio), key=lambda x: x[1].duration, reverse=False)
                elif isinstance(audio[0], str):
                    recordings = [Recording.from_file(i) for i in audio]
                    cuts = [
                        MultiCut(id="", recording=recording, start=0, duration=recording.duration, channel=0)
                        for recording in recordings
                    ]
                    audios = sorted(zip(indexs, cuts), key=lambda x: x[1].duration, reverse=False)
                elif isinstance(audio[0], np.ndarray):
                    audios = sorted(zip(indexs, audio), key=lambda x: x[1].shape[0], reverse=False)
                else:
                    raise ValueError(f"Unsupported audio type {type(audio[0])}")
            else:
                audios = list(enumerate(audio))
        else:
            audios = [(0, audio)]

        dataset = NumpyDataset(audios, sampling_rate=self.sampling_rate)

        def collate_fn(batch, device=self.device):
            batch_size = len(batch)
            feats, feats_len = self.extract_feat([item[1] for item in batch])
            return (
                batch_size,
                [item[0] for item in batch],
                torch.from_numpy(feats).to(device),
                torch.from_numpy(feats_len).to(device),
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        results = [None] * len(audios)
        for batch in tqdm(dataloader, desc="Transcribing", total=len(audios) // batch_size):
            batch_size, indexs, feats, feats_len = batch

            ctc_logits, encoder_out_lens = self.model.inference(
                feats,
                feats_len,
                language,
                textnorm,
            )
            encoder_out_lens = encoder_out_lens.cpu().numpy().tolist()
            ctc_maxids = ctc_logits.argmax(dim=-1)

            for b, index in enumerate(indexs):
                yseq = ctc_maxids[b, : encoder_out_lens[b]]
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                results[index] = self.tokenizer.decode(token_int)

        return [OmniTranscription.parse(i) for i in results]

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
    def __init__(self, segments: List[Tuple[int, Any]], sampling_rate: int):
        self.segments = segments
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        if isinstance(segment[1], np.ndarray):
            audio = segment
        elif isinstance(segment[1], Pathlike):
            audio = (segment[0], librosa.load(segment, sr=self.sampling_rate, mono=True)[0])
        elif isinstance(segment[1], Cut):
            audio = (segment[0], segment[1].resample(self.sampling_rate).load_audio()[0])
        else:
            raise ValueError(f"Unsupported audio type {type(segment)}")

        assert audio[1].ndim == 1, f"Only support mono audio, but got {audio[1].ndim} channels"
        return audio
