"""Build the STT test sets from FLEURS (google/fleurs, parquet conversion branch).

Only the first ROW_LIMIT rows of each test parquet are streamed over HTTP range reads, so the
462 MB pl_pl and 402 MB en_us files are never downloaded whole. Output:
  bench/data/stt/{pl,en}/wer/*.wav        16 kHz mono PCM16 utterances with reference text
  bench/data/stt/{pl,en}/refs.tsv         utt_id, file, duration_s, ref, raw_ref, fleurs_id, row
  bench/data/stt/{pl,en}/latency/*.wav    real speech cut at silences to 2-5 s (no reference)
  bench/data/stt/{pl,en}/latency.tsv      clip id, file, duration_s, source utterance, cut target
"""
import csv
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import HfFileSystem

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stt")
SOURCES = {
    "pl": ("datasets/google/fleurs@refs/convert/parquet/pl_pl/test/0000.parquet", 44),
    "en": ("datasets/google/fleurs@refs/convert/parquet/en_us/test/0000.parquet", 36),
}
ROW_LIMIT = 150
CUT_TARGETS = (2.0, 3.0, 4.0, 5.0)
SR = 16000


def read_meta(path):
    with HfFileSystem().open(path, "rb") as handle:
        table = pq.read_table(handle, columns=["id", "num_samples", "transcription", "raw_transcription"], pre_buffer=False)
    return table.to_pylist()


def read_audio_rows(path, wanted_rows):
    last = max(wanted_rows)
    out, seen = {}, 0
    with HfFileSystem().open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle, pre_buffer=False)
        for batch in parquet.iter_batches(batch_size=8, columns=["id", "audio"]):
            for row in batch.to_pylist():
                if seen in wanted_rows:
                    out[seen] = (row["id"], row["audio"]["bytes"])
                seen += 1
            if seen > last:
                break
    return out


def decode_16k(payload):
    data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SR:
        mono = soxr.resample(mono, rate, SR, quality="HQ")
    return np.clip(mono, -1.0, 1.0)


def write_wav(path, samples):
    sf.write(path, samples, SR, subtype="PCM_16")
    return round(len(samples) / SR, 3)


def pick_utterances(rows, count):
    seen, picked = set(), []
    for index, row in enumerate(rows[:ROW_LIMIT]):
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        picked.append(index)
        if len(picked) == count:
            break
    return picked


def pause_midpoints(samples, min_pause_ms=60, rel_level=0.1):
    frame = SR // 100
    count = len(samples) // frame
    rms = np.sqrt((samples[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
    quiet = rms < rel_level * np.percentile(rms, 90)
    mids, i = [], 0
    while i < count:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < count and quiet[j]:
            j += 1
        if (j - i) * 10 >= min_pause_ms:
            mids.append(int((i + j) / 2 * frame))
        i = j
    return mids


def build_latency_clips(lang, wer_audio, out_dir):
    lat_dir = os.path.join(out_dir, "latency")
    os.makedirs(lat_dir, exist_ok=True)
    rows = []
    for utt_index, (file_name, samples) in enumerate(wer_audio):
        used = []
        for cut in pause_midpoints(samples):
            if not (2.0 * SR <= cut <= 5.0 * SR):
                continue
            if any(abs(cut - u) < 0.3 * SR for u in used):
                continue
            used.append(cut)
            target = min(CUT_TARGETS, key=lambda t: abs(cut / SR - t))
            name = f"{lang}_lat_{utt_index:03d}_{cut / SR:.2f}s.wav"
            seconds = write_wav(os.path.join(lat_dir, name), samples[:cut])
            rows.append({"clip": f"latency/{name}", "duration_s": seconds,
                         "source": file_name, "cut_target_s": target})
    with open(os.path.join(out_dir, "latency.tsv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip", "duration_s", "source", "cut_target_s"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def rebuild_latency_only():
    for lang in SOURCES:
        out_dir = os.path.join(ROOT, lang)
        with open(os.path.join(out_dir, "refs.tsv"), encoding="utf-8") as handle:
            refs = list(csv.DictReader(handle, delimiter="\t"))
        wer_audio = [(os.path.basename(r["file"]), sf.read(os.path.join(out_dir, r["file"]), dtype="float32")[0])
                     for r in refs]
        clips = build_latency_clips(lang, wer_audio, out_dir)
        print(f"{lang}: {len(clips)} latency clips rebuilt", file=sys.stderr)


def main():
    if "--latency-only" in sys.argv:
        return rebuild_latency_only()
    for lang, (path, count) in SOURCES.items():
        out_dir = os.path.join(ROOT, lang)
        wer_dir = os.path.join(out_dir, "wer")
        os.makedirs(wer_dir, exist_ok=True)
        meta = read_meta(path)
        picked = pick_utterances(meta, count)
        audio = read_audio_rows(path, set(picked))
        refs, wer_audio = [], []
        for row_index in picked:
            fleurs_id, payload = audio[row_index]
            row = meta[row_index]
            samples = decode_16k(payload)
            name = f"{lang}_{len(refs):03d}_fleurs{fleurs_id}.wav"
            seconds = write_wav(os.path.join(wer_dir, name), samples)
            refs.append({"utt_id": f"{lang}_{len(refs):03d}", "file": f"wer/{name}", "duration_s": seconds,
                         "ref": row["transcription"], "raw_ref": row["raw_transcription"],
                         "fleurs_id": fleurs_id, "row": row_index})
            wer_audio.append((name, samples))
        with open(os.path.join(out_dir, "refs.tsv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(refs[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(refs)
        clips = build_latency_clips(lang, wer_audio, out_dir)
        durs = np.array([r["duration_s"] for r in refs])
        print(f"{lang}: {len(refs)} WER utts, duration median {np.median(durs):.2f}s "
              f"[{durs.min():.2f}-{durs.max():.2f}], {int((durs < 5).sum())} under 5 s; "
              f"{len(clips)} latency clips", file=sys.stderr)


if __name__ == "__main__":
    main()
