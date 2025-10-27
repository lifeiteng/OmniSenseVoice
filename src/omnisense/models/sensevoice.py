#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

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
from lhotse.supervision import AlignmentItem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .k2_utils import ctc_greedy_search
from .model import SenseVoiceSmall

PATTERN = r"<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>(.*)?"


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

    words: Optional[List[AlignmentItem]] = None

    @property
    def end(self) -> float:
        return round(self.start + self.duration, ndigits=8)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "emotion": self.emotion,
            "event": self.event,
            "textnorm": self.textnorm,
            "text": self.text,
            **({"words": [w.serialize() for w in self.words]} if self.words is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: dict):
        words = data.get("words", [])
        return OmniTranscription(
            language=data["language"],
            emotion=data["emotion"],
            event=data["event"],
            textnorm=data["textnorm"],
            text=data.get("text"),
            words=[AlignmentItem.deserialize(w) for w in words] if words else None,
        )

    @classmethod
    def parse(cls, input_string: str):
        """
        Parse from SenseVoice output.
        """
        # '<|nospeech|><|EMO_UNKNOWN|><|Laughter|><|woitn|>'
        # '<|en|><|NEUTRAL|><|Speech|><|withitn|>As you can see...'

        match = re.match(PATTERN, input_string)
        if match:
            language, emotion, event, textnorm, text = match.groups()
            return cls(language, emotion, event, textnorm, text or "")
        else:
            # <|ja|><|EMO_UNKNOWN|><|Speech|>
            pattern = r"<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>"
            match = re.match(pattern, input_string)
            if match:
                language, emotion, event = match.groups()
                return cls(language, emotion, event, "")
            else:
                return cls("", "", "", "", input_string)


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
        device: Optional[str] = None,
        quantize: bool = False,
    ):
        device_id = int(device_id)
        if device:
            self.device = str(device)
        else:
            self.device = "cpu"
            if device_id != -1 and torch.cuda.is_available():
                self.device = f"cuda:{device_id}"

        model, kwargs = SenseVoiceSmall.from_pretrained(model_dir, quantize=quantize, device=self.device)

        config_file = kwargs["config"]
        config = read_yaml(config_file)

        self.tokenizer = SentencepiecesTokenizer(bpemodel=kwargs["tokenizer_conf"]["bpemodel"])
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.frontend.opts.frame_opts.dither = 0
        self.sampling_rate = self.frontend.opts.frame_opts.samp_freq

        model.eval()
        self.model = model.to(self.device)

        self.blank_id = 0

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, List[np.ndarray], List[Cut]],
        language: Union[str, List[str]] = "auto",
        textnorm: str = "woitn",
        sort_by_duration: bool = True,
        batch_size: int = 4,
        timestamps: bool = False,
        num_workers: int = 0,
        progressbar: bool = True,
    ) -> List[OmniTranscription]:
        if isinstance(audio, List):
            if not isinstance(language, List):
                languages = [language] * len(audio)
            else:
                languages = language
                assert len(languages) == len(
                    audio
                ), "The length of audio and language must be the same, or language must be a single string."

            indexs = list(range(len(audio)))
            if sort_by_duration:
                if isinstance(audio[0], Cut):
                    audios = sorted(zip(indexs, zip(audio, languages)), key=lambda x: x[1][0].duration, reverse=False)
                elif isinstance(audio[0], str):
                    recordings = [Recording.from_file(i) for i in audio]
                    cuts = [
                        MultiCut(id="", recording=recording, start=0, duration=recording.duration, channel=0)
                        for recording in recordings
                    ]
                    audios = sorted(zip(indexs, zip(cuts, languages)), key=lambda x: x[1][0].duration, reverse=False)
                elif isinstance(audio[0], np.ndarray):
                    audios = sorted(zip(indexs, zip(audio, languages)), key=lambda x: x[1][0].shape[0], reverse=False)
                else:
                    raise ValueError(f"Unsupported audio type {type(audio[0])}")
            else:
                audios = list(enumerate(zip(audio, languages)))
        else:
            audios = [(0, (audio, language[0] if isinstance(language, list) else language))]

        dataset = NumpyDataset(audios, sampling_rate=self.sampling_rate)

        def collate_fn(batch, device=self.device):
            batch_size = len(batch)
            feats, feats_len = self.extract_feat([item[1][0] for item in batch])
            return (
                batch_size,
                [item[0] for item in batch],
                [item[1][1] for item in batch],
                feats,
                feats_len,
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
        for batch in (
            tqdm(dataloader, desc="Transcribing", total=len(audios) // batch_size) if progressbar else dataloader
        ):
            batch_size, indexs, languages, feats, feats_len = batch

            ctc_logits, encoder_out_lens = self.model.inference(
                torch.from_numpy(feats).to(self.device),
                torch.from_numpy(feats_len).to(self.device),
                languages,
                textnorm,
            )
            if timestamps:
                # decode first 4 frames
                ctc_maxids = ctc_logits[:, :4].argmax(dim=-1)
                for b, index in enumerate(indexs):
                    yseq = ctc_maxids[b, :4]
                    token_int = yseq.tolist()
                    results[index] = OmniTranscription.parse(self.tokenizer.decode(token_int))

                utt_time_pairs, utt_words = ctc_greedy_search(
                    ctc_logits[:, 4:],
                    encoder_out_lens - 4,
                    sp=self.tokenizer.sp,
                    subsampling_factor=6,
                    frame_shift_ms=self.frontend.opts.frame_opts.frame_shift_ms,
                )
                for k, (result, time_pairs, words) in enumerate(
                    zip([results[index] for index in indexs], utt_time_pairs, utt_words)
                ):
                    results[indexs[k]] = result._replace(
                        words=[
                            AlignmentItem(symbol=word, start=pair[0], duration=round(pair[1] - pair[0], ndigits=4))
                            for (pair, word) in zip(time_pairs, words)
                        ],
                        text=" ".join(words),
                    )
            else:
                encoder_out_lens = encoder_out_lens.cpu().numpy().tolist()

                ctc_maxids = ctc_logits.argmax(dim=-1)

                for b, index in enumerate(indexs):
                    yseq = ctc_maxids[b, : encoder_out_lens[b]]
                    yseq = torch.unique_consecutive(yseq, dim=-1)
                    mask = yseq != self.blank_id
                    token_int = yseq[mask].tolist()
                    results[index] = self.tokenizer.decode(token_int)

        if timestamps:
            return results

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
        segment = self.segments[idx]  # (index, (audio, language))

        if isinstance(segment[1][0], np.ndarray):
            audio = segment
        elif isinstance(segment[1][0], str) or isinstance(segment[1][0], Path):
            audio = (segment[0], (librosa.load(segment[1][0], sr=self.sampling_rate, mono=True)[0], segment[1][1]))
        elif isinstance(segment[1][0], Cut):
            audio = (segment[0], (segment[1][0].resample(self.sampling_rate).load_audio()[0], segment[1][1]))
        else:
            raise ValueError(f"Unsupported audio type {type(segment[1][0])}")

        if audio[1][0].shape[0] <= 1000:
            audio = (audio[0], (np.pad(audio[1][0], (0, 1000 - audio[1][0].shape[0])), audio[1][1]))

        assert audio[1][0].ndim == 1, f"Only support mono audio, but got {audio[1][0].ndim} channels"
        return audio
