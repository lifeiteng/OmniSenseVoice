from omnisense.bin.omnisense import _benchmark

if __name__ == "__main__":

    _benchmark(
        "benchmark/data/manifests/libritts/libritts_cuts_dev-clean.jsonl",
        "auto",
        "woitn",
        device_id=0,
        sort_by_duration=True,
        debug=True,
    )
