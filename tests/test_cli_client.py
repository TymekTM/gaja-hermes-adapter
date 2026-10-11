import asyncio
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import soundfile as sf
from clients.cli_wav import prepare_audio_pcm16_chunks, run_client
from clients.mock_server import MockServer
import websockets


@pytest.fixture
def sample_wav(tmp_path):
    wav_path = str(tmp_path / "test.wav")
    sr = 16000
    duration = 0.5  # 0.5s
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(wav_path, samples.astype(np.float32), sr)
    return wav_path


def test_prepare_audio_pcm16_chunks(sample_wav):
    chunks = prepare_audio_pcm16_chunks(sample_wav, chunk_duration_ms=40.0)
    # 0.5s / 0.04s = 12.5 -> ~13 chunks
    assert len(chunks) in (12, 13)
    # Each chunk should be 640 samples * 2 bytes = 1280 bytes
    for chunk in chunks[:-1]:
        assert len(chunk) == 1280


@pytest.mark.asyncio
async def test_cli_wav_against_mock_server(sample_wav, tmp_path):
    port = 8991
    mock = MockServer(host="127.0.0.1", port=port, web_dir="web")

    server = await websockets.serve(
        mock.handle_ws,
        host="127.0.0.1",
        port=port,
        process_request=mock.process_request,
    )

    out_reply = str(tmp_path / "reply.wav")

    class Args:
        url = f"ws://127.0.0.1:{port}/ws"
        lang = "pl"
        text = None
        input_wav = sample_wav
        realtime = False
        out = out_reply
        timeout = 5.0

    try:
        ret = await run_client(Args())
        assert ret == 0
        assert os.path.exists(out_reply)
        data, sr = sf.read(out_reply)
        assert sr in (22050, 24000)
        assert len(data) > 0
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_cli_text_against_mock_server():
    port = 8992
    mock = MockServer(host="127.0.0.1", port=port, web_dir="web")

    server = await websockets.serve(
        mock.handle_ws,
        host="127.0.0.1",
        port=port,
        process_request=mock.process_request,
    )

    class Args:
        url = f"ws://127.0.0.1:{port}/ws"
        lang = "pl"
        text = "testowa wiadomość"
        input_wav = None
        realtime = False
        out = None
        timeout = 5.0

    try:
        ret = await run_client(Args())
        assert ret == 0
    finally:
        server.close()
        await server.wait_closed()
