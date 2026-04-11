import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import synapse as syn
from synapse.api.datatype_pb2 import BroadbandFrame
from synapse.client.taps import Tap


def build_config(
    peripheral_id: int, n_channels: int, fs_hz: int, udp_port: int
) -> syn.Config:
    """
    BroadbandSource(peripheral_id) configured for n_channels electrodes.
    Channels 0..n_channels-1, electrode_id=i, reference_id=i (no referencing).
    bit_width=16, gain=20.0, no filtering at source.
    udp_port reserved for future StreamOut transport; unused with Tap API.
    """
    channels = [
        syn.Channel(id=i, electrode_id=i, reference_id=i)
        for i in range(n_channels)
    ]
    broadband = syn.BroadbandSource(
        peripheral_id=peripheral_id,
        sample_rate_hz=fs_hz,
        bit_width=16,
        gain=20.0,
        signal=syn.SignalConfig(
            electrode=syn.ElectrodeConfig(
                channels=channels,
                low_cutoff_hz=0.0,
                high_cutoff_hz=fs_hz / 2.0,
            )
        ),
    )
    config = syn.Config()
    config.nodes = []
    config.connections = []
    config.add_node(broadband)
    return config


def capture(
    uri: str,
    peripheral_id: int,
    duration_s: float,
    n_channels: int,
    fs_hz: int,
    udp_port: int,
) -> tuple[np.ndarray, float]:
    """
    Connect, configure, start, stream for duration_s, stop.
    Returns (neural (C, T), actual_fs).
    """
    config = build_config(peripheral_id, n_channels, fs_hz, udp_port)
    device = syn.Device(uri)
    device.configure(config)
    device.start()

    tap = Tap(uri)
    available = tap.list_taps()
    tap_name = available[0].name if available else "broadband"
    for t in available:
        if "broadband" in t.name.lower():
            tap_name = t.name
            break
    tap.connect(tap_name)

    frames: list[np.ndarray] = []
    t0 = time.monotonic()
    for raw_msg in tap.stream():
        frame = BroadbandFrame()
        frame.ParseFromString(raw_msg)
        samples = np.array(list(frame.frame_data), dtype=np.float32)
        if len(samples) >= n_channels:
            n_t = len(samples) // n_channels
            chunk = samples[: n_t * n_channels].reshape(n_t, n_channels).T
            frames.append(chunk)
        if time.monotonic() - t0 >= duration_s:
            break

    device.stop()
    tap.disconnect()

    if frames:
        neural = np.concatenate(frames, axis=1)
    else:
        neural = np.zeros((n_channels, 0), dtype=np.float32)

    elapsed = time.monotonic() - t0
    actual_fs = neural.shape[1] / elapsed if elapsed > 0 else float(fs_hz)
    return neural, actual_fs


def analyze(x: np.ndarray, fs_hz: float) -> dict:
    """
    Per-channel RMS, median RMS, zero-crossing rate, threshold crossings,
    spectral centroid, pooled ISI histogram.
    """
    C, T = x.shape
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    median_rms = float(np.median(rms))

    # Zero-crossing rate per channel (crossings per second)
    sign_diff = np.diff(np.sign(x), axis=1)
    zcr = np.sum(sign_diff != 0, axis=1) / (T / fs_hz) if T > 1 else np.zeros(C)

    # Threshold crossings at -3*RMS and pooled ISI
    thresholds = -3.0 * rms
    crossings_count = np.zeros(C, dtype=int)
    all_isis: list[float] = []
    for c in range(C):
        times: list[int] = []
        for t in range(1, T):
            if x[c, t] < thresholds[c] and x[c, t - 1] >= thresholds[c]:
                times.append(t)
        crossings_count[c] = len(times)
        if len(times) > 1:
            isis = np.diff(times) / fs_hz
            all_isis.extend(isis.tolist())

    # Spectral centroid per channel
    freqs = np.fft.rfftfreq(T, d=1.0 / fs_hz) if T > 0 else np.array([0.0])
    centroids = np.zeros(C)
    for c in range(C):
        spectrum = np.abs(np.fft.rfft(x[c]))
        power = spectrum ** 2
        total = power.sum()
        if total > 0:
            centroids[c] = float(np.sum(freqs * power) / total)

    # ISI histogram (log bins 1us .. 1s)
    bins = np.logspace(-6, 0, 50)
    hist_counts, hist_edges = np.histogram(all_isis, bins=bins) if all_isis else (
        np.zeros(len(bins) - 1, dtype=int), bins,
    )

    return {
        "per_channel_rms": rms.tolist(),
        "median_rms": median_rms,
        "zero_crossing_rate": zcr.tolist(),
        "threshold_crossings": crossings_count.tolist(),
        "spectral_centroid": centroids.tolist(),
        "isi_hist_bins": hist_edges.tolist(),
        "isi_hist_counts": hist_counts.tolist(),
    }


def plot(x: np.ndarray, fs_hz: float, stats: dict, out_png: Path) -> None:
    """3-panel figure: raw trace, ISI histogram, PSD."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    C, T = x.shape
    n1s = min(int(fs_hz), T)
    t_axis = np.arange(n1s) / fs_hz
    axes[0].plot(t_axis, x[0, :n1s], linewidth=0.3)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw trace ch 0 (1 s)")

    bins = np.array(stats["isi_hist_bins"])
    counts = np.array(stats["isi_hist_counts"])
    axes[1].bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("ISI (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("ISI histogram (pooled)")

    if T > 0:
        freqs = np.fft.rfftfreq(T, d=1.0 / fs_hz)
        spectrum = np.abs(np.fft.rfft(x[0]))
        psd_db = 20.0 * np.log10(spectrum + 1e-12)
        axes[2].plot(freqs, psd_db, linewidth=0.5)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power (dB)")
    axes[2].set_title("PSD ch 0")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Inspect encoder broadband data")
    p.add_argument("--uri", required=True)
    p.add_argument("--peripheral-id", type=int, default=104)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--n-channels", type=int, default=32)
    p.add_argument("--fs-hz", type=int, default=30000)
    p.add_argument("--udp-port", type=int, default=50038)
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/inspect"))
    args = p.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{ts}.npz"
    png_path = args.out_dir / f"{ts}.png"
    json_path = args.out_dir / f"{ts}.json"

    x, actual_fs = capture(
        args.uri, args.peripheral_id, args.duration,
        args.n_channels, args.fs_hz, args.udp_port,
    )
    stats = analyze(x, actual_fs)
    np.savez(
        npz_path, neural=x.astype(np.float32),
        fs=actual_fs, peripheral_id=args.peripheral_id,
    )
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    plot(x, actual_fs, stats, png_path)
    print(
        f"INSPECT saved={npz_path} shape={x.shape} fs={actual_fs:.1f} "
        f"median_rms={stats['median_rms']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
