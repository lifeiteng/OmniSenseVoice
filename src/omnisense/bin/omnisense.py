import re
import time

import click
from kaldialign import edit_distance
from lhotse import load_manifest
from lhotse.utils import Pathlike

from omnisense.models import OmniSenseVoiceSmall, OmniTranscription

from .cli_base import cli


@cli.command()
@click.argument(
    "audio_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--language",
    type=click.Choice(["auto", "zh", "en", "yue", "ja", "ko"]),
    default="auto",
    help="Language of the audio file.",
)
@click.option(
    "--textnorm",
    type=click.Choice(["withitn", "woitn"]),
    default="withitn",
    help="Text normalization.",
)
@click.option(
    "-g",
    "--device-id",
    type=int,
    default=-1,
    help="GPU ID to run the model(defualt: -1 use cpu).",
)
@click.option(
    "--quantize",
    is_flag=True,
    help="Use quantized model.",
)
@click.option(
    "-t",
    "--timestamps",
    is_flag=True,
    help="Return word level timestamps."
)
def transcribe(
    audio_path: Pathlike,
    language: str,
    textnorm: str,
    device_id: int,
    quantize: bool,
    timestamps: bool
):
    omnisense = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=quantize, device_id=device_id)
    result = omnisense.transcribe(audio_path, language=language, textnorm=textnorm,
                                  batch_size=8,
                                  timestamps=timestamps)
    print(result[0].text)


@cli.command()
@click.argument(
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--language",
    type=click.Choice(["auto", "zh", "en", "yue", "ja", "ko"]),
    default="auto",
    help="Language of the audio file.",
)
@click.option(
    "--textnorm",
    type=click.Choice(["withitn", "woitn"]),
    default="woitn",
    help="Text normalization.",
)
@click.option(
    "--device-id",
    type=int,
    default=-1,
    help="Device ID to run the model.",
)
@click.option(
    "-s",
    "--sort-by-duration",
    is_flag=True,
    help="Sort cuts by duration before processing.",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Batch size.",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers.",
)
@click.option(
    "--quantize",
    is_flag=True,
    help="Use quantized model.",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="debug.",
)
def benchmark(
    manifest_path: Pathlike,
    language: str,
    textnorm: str,
    device_id: int,
    sort_by_duration: bool,
    batch_size: int = 4,
    num_workers: int = 0,
    quantize: bool = False,
    debug: bool = False,
):
    return _benchmark(
        manifest_path, language, textnorm, device_id, sort_by_duration, batch_size, num_workers, quantize, debug
    )


def _benchmark(
    manifest_path: Pathlike,
    language: str,
    textnorm: str,
    device_id: int,
    sort_by_duration: bool,
    batch_size: int = 4,
    num_workers: int = 0,
    quantize: bool = False,
    debug: bool = False,
):
    cuts = load_manifest(manifest_path)

    if debug:
        cuts = cuts.sort_by_duration() if sort_by_duration else cuts
        cuts = cuts[:100]

    omnisense = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=quantize, device_id=device_id)

    begin = time.time()
    results = omnisense.transcribe(
        [cut.recording.sources[0].source for cut in cuts],
        language=language,
        textnorm=textnorm,
        sort_by_duration=sort_by_duration,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    compute_time = time.time() - begin

    def _clean_punctuations(text):
        return re.sub(r"[^\w\s]", "", text).split()

    wers = []
    for result, cut in zip(results, cuts):
        cut_text = _clean_punctuations(cut.supervisions[0].text.lower())
        wer = edit_distance(cut_text, _clean_punctuations(result.text.lower()))["total"] * 100 / len(cut_text)
        wers.append(wer)

    audio_time = sum(cut.duration for cut in cuts)
    print(
        f"Audio time: {audio_time:.4f}s Compute time: {compute_time:.4f}s RTF: {compute_time / audio_time:.4f} WER: {sum(wers)/len(wers):.2f}%"  # noqa
    )


if __name__ == "__main__":
    _benchmark("benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl.gz", "auto", "woitn", device_id=0)
