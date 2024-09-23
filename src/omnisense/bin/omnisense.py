import click
from lhotse.utils import Pathlike

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
def transcribe(
    audio_path: Pathlike,
    language: str,
    textnorm: str,
):
    pass


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
    default="withitn",
    help="Text normalization.",
)
def benchmark(
    manifest_path: Pathlike,
    language: str,
    textnorm: str,
):
    pass
