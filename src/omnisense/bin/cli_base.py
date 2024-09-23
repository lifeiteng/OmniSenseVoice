import logging

import click


@click.group()
def cli():
    """
    The shell entry point to OmniSenseVoice.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
