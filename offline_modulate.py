#!/usr/bin/env python3
"""
Offline ambient sound modulator.

Takes a recording of a real acoustic environment and produces modulated
versions across a grid of modulation rates and depths, so you can listen
through them and find out what the effect actually sounds like.

This is step 1: no microphone, no real-time audio, nothing to configure.
Record 30 seconds on your phone somewhere with real ambient sound, run this,
put on isolating earphones, and listen.

Usage
-----
    pip install numpy soundfile

    # default grid: 4/10/20/40 Hz x 0.3/0.6/1.0 depth
    python offline_modulate.py cafe.wav

    # custom grid
    python offline_modulate.py cafe.wav --mod-hz 6 10 --depth 0.5 1.0

    # one file that sweeps the rate from 2 to 60 Hz, good for finding
    # the range where the effect is most salient
    python offline_modulate.py cafe.wav --sweep

If soundfile refuses to open your recording (phone recordings are often .m4a),
convert it first:
    ffmpeg -i recording.m4a recording.wav
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf


def envelope(phase, depth, duty=0.5, ramp_frac=0.05, waveform="square"):
    """Modulation envelope in [1 - depth, 1].

    phase      array of phase values in [0, 1)
    depth      0 = untouched, 1 = full gating (silence in the off phase)
    duty       fraction of the cycle in the "on" state (square only)
    ramp_frac  raised-cosine ramp as a fraction of the cycle. Never set this
               to 0. Hard edges produce audible clicks and spectral splatter,
               and listeners report hearing clicks rather than gating.
    """
    if waveform == "sine":
        g = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))
    else:
        r = max(ramp_frac, 1e-6)
        g = np.zeros_like(phase)
        rise = phase < r
        on = (phase >= r) & (phase < duty)
        fall = (phase >= duty) & (phase < duty + r)
        g[rise] = 0.5 * (1.0 - np.cos(np.pi * phase[rise] / r))
        g[on] = 1.0
        g[fall] = 0.5 * (1.0 + np.cos(np.pi * (phase[fall] - duty) / r))

    return (1.0 - depth) + depth * g


def apply_envelope(x, env):
    """Multiply audio by an envelope, broadcasting across channels."""
    if x.ndim == 1:
        return x * env
    return x * env[:, None]


def normalize(y, headroom_db=1.0):
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        return y
    target = 10 ** (-headroom_db / 20.0)
    return y * (target / peak)


def constant_rate(n, fs, mod_hz):
    return (np.arange(n) * mod_hz / fs) % 1.0


def swept_rate(n, fs, f0, f1):
    """Phase for a rate that sweeps logarithmically from f0 to f1."""
    t = np.arange(n) / fs
    dur = n / fs
    inst = f0 * (f1 / f0) ** (t / dur)
    phase = np.cumsum(inst) / fs
    return phase % 1.0, inst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="ambient recording (wav preferred)")
    ap.add_argument("--outdir", default=None,
                    help="output folder (default: <infile>_modulated)")
    ap.add_argument("--mod-hz", type=float, nargs="+",
                    default=[4, 10, 20, 40],
                    help="modulation rates in Hz")
    ap.add_argument("--depth", type=float, nargs="+",
                    default=[0.3, 0.6, 1.0],
                    help="modulation depths, 0 to 1")
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--ramp", type=float, default=0.05,
                    help="raised-cosine ramp as a fraction of the cycle")
    ap.add_argument("--waveform", default="square",
                    choices=["square", "sine"])
    ap.add_argument("--sweep", action="store_true",
                    help="also write a file sweeping the rate 2 to 60 Hz")
    ap.add_argument("--sweep-range", type=float, nargs=2, default=[2.0, 60.0])
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep original levels instead of peak-normalizing")

    args = ap.parse_args()

    try:
        x, fs = sf.read(args.infile, always_2d=False, dtype="float64")
    except Exception as e:
        print(f"could not read {args.infile}: {e}", file=sys.stderr)
        print("if this is a phone recording, convert it first:", file=sys.stderr)
        print(f"    ffmpeg -i {args.infile} recording.wav", file=sys.stderr)
        sys.exit(1)

    n = x.shape[0]
    dur = n / fs
    ch = 1 if x.ndim == 1 else x.shape[1]
    stem = os.path.splitext(os.path.basename(args.infile))[0]
    outdir = args.outdir or f"{stem}_modulated"
    os.makedirs(outdir, exist_ok=True)

    print(f"input   : {args.infile}")
    print(f"          {dur:.1f} s, {fs} Hz, {ch} channel(s)")
    print(f"output  : {outdir}/")
    print()

    # unmodulated reference, so you always have an A/B comparison
    ref = normalize(x.copy()) if not args.no_normalize else x
    ref_path = os.path.join(outdir, f"{stem}_00_reference.wav")
    sf.write(ref_path, ref, fs)
    print(f"  wrote  {os.path.basename(ref_path)}   (unmodulated reference)")

    for mod_hz in args.mod_hz:
        cycles = dur * mod_hz
        if cycles < 5:
            print(f"  skip   {mod_hz} Hz: recording too short "
                  f"({cycles:.1f} cycles)")
            continue
        ph = constant_rate(n, fs, mod_hz)
        for depth in args.depth:
            env = envelope(ph, depth, args.duty, args.ramp, args.waveform)
            y = apply_envelope(x, env)
            if not args.no_normalize:
                y = normalize(y)
            name = f"{stem}_{mod_hz:g}Hz_depth{depth:g}.wav"
            sf.write(os.path.join(outdir, name), y, fs)
            print(f"  wrote  {name}")

    if args.sweep:
        f0, f1 = args.sweep_range
        ph, inst = swept_rate(n, fs, f0, f1)
        depth = max(args.depth)
        env = envelope(ph, depth, args.duty, args.ramp, args.waveform)
        y = apply_envelope(x, env)
        if not args.no_normalize:
            y = normalize(y)
        name = f"{stem}_sweep_{f0:g}to{f1:g}Hz_depth{depth:g}.wav"
        sf.write(os.path.join(outdir, name), y, fs)
        print(f"  wrote  {name}")
        print()
        print(f"  the sweep goes from {f0:g} Hz at the start to {f1:g} Hz "
              f"at the end, logarithmically.")
        print(f"  at {dur:.0f} s total, the rate is roughly:")
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            idx = min(int(frac * (n - 1)), n - 1)
            print(f"      {frac*dur:5.1f} s  ->  {inst[idx]:5.1f} Hz")

    print()
    print("listen through them with well isolating in-ear headphones.")
    print("for each one, note: can you tell it apart from the reference?")
    print("does it feel like the environment is pulsing, or just like a")
    print("different timbre? is it tolerable for more than a minute?")


if __name__ == "__main__":
    main()
