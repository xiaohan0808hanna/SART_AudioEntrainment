#!/usr/bin/env python3
"""
Real-time hear-through gating.

Built-in microphone picks up the room, the modulation envelope is applied on the
output side, and the result plays through your earphones. With AirPods ANC on,
the real room is suppressed and what you hear is the gated version.

Input and output are opened as SEPARATE streams with a ring buffer between them.
This matters here: the built-in mic runs on the Mac's clock and Bluetooth
earphones run on their own, so a single duplex stream would fight the drift and
glitch. Separate streams just let the buffer absorb it.

Three conditions:
    modulated    rhythmic gating (the experimental condition)
    attenuated   constant attenuation, energy-matched to `modulated`, no rhythm
                 (the control condition: isolates rhythm from "it got quieter")
    passthrough  no processing at all (for calibration)

Usage
-----
    pip install sounddevice numpy soundfile

    python3 realtime_gate.py --list-devices
    python3 realtime_gate.py --calibrate
    python3 realtime_gate.py --mod-hz 10 --depth 0.3
    python3 realtime_gate.py --mod-hz 10 --depth 0.3 --record demo.wav

Read the setup checklist printed at startup before trusting anything you hear.

From another script (e.g. sart_experiment.py):

    from realtime_gate import RealtimeGate
    gate = RealtimeGate(mod_hz=16, depth=0.5)
    gate.start()
    ...
    gate.stop()
"""

import argparse
import queue
import sys
import threading

import numpy as np
import sounddevice as sd

CHECKLIST = """
setup checklist  (get these wrong and the results are meaningless)

  macOS input
    Sound > Input: pick the MacBook microphone, NOT the AirPods.
      Selecting AirPods as input drops the whole link into 16 kHz mono
      telephony mode and the output quality collapses.
    Control Center > Mic Mode: Standard.
      Voice Isolation strips out steady background noise. That noise IS
      your signal. It would delete the thing you are trying to modulate.
    Sound > Input: turn off ambient noise reduction if the option is there.

  AirPods
    Set to Noise Cancellation. Not Transparency, not Adaptive, not Off.
      Transparency passes the real room through unmodulated and fills in
      every gap you just created.
    Turn off Conversation Awareness (it ducks volume when you speak).
    Turn off Personalized Spatial Audio and head tracking.

  volume
    Calibrate once with --calibrate, then do not touch the volume again.
"""


def envelope(phase, depth, duty=0.5, ramp_frac=0.05, waveform="square"):
    """Modulation envelope in [1 - depth, 1]."""
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


def matched_constant(depth, duty, ramp_frac, waveform):
    """Constant gain with the same RMS as the modulated envelope.

    Used for the control condition, so that `attenuated` removes the same
    amount of energy as `modulated` but carries no rhythm.
    """
    ph = np.linspace(0, 1, 20000, endpoint=False)
    env = envelope(ph, depth, duty, ramp_frac, waveform)
    return float(np.sqrt(np.mean(env ** 2)))


class Params:
    def __init__(self, **kw):
        self.lock = threading.Lock()
        for k, v in kw.items():
            setattr(self, k, v)

    def snapshot(self):
        with self.lock:
            return (self.mode, self.mod_hz, self.depth, self.duty,
                    self.ramp_frac, self.gain, self.waveform, self.const)

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)
            if {"depth", "duty", "ramp_frac", "waveform"} & set(kw):
                self.const = matched_constant(self.depth, self.duty,
                                              self.ramp_frac, self.waveform)


class RealtimeGate:
    """Mic -> modulation envelope -> earphones, on separate in/out streams."""

    def __init__(self, mode="modulated", mod_hz=10.0, depth=0.3, duty=0.5,
                 ramp_frac=0.05, gain=1.0, waveform="square", fs=48000,
                 blocksize=256, in_device=None, out_device=None,
                 record=False):
        self.params = Params(mode=mode, mod_hz=mod_hz, depth=depth,
                             duty=duty, ramp_frac=ramp_frac, gain=gain,
                             waveform=waveform,
                             const=matched_constant(depth, duty, ramp_frac,
                                                    waveform))
        self.fs = fs
        self.blocksize = blocksize
        self.in_device = in_device
        self.out_device = out_device
        self.ring = queue.Queue(maxsize=64)
        self.phase = 0.0
        self.peak = 0.0
        self.input_peak = 0.0
        self.clipped = 0
        self.starved = 0
        self.out_sumsq = 0.0
        self.out_samples = 0
        self.recorded = [] if record else None
        self.rec_lock = threading.Lock()
        self.istream = None
        self.ostream = None

    def _in_cb(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        x = indata[:, 0].copy()
        if len(x):
            self.input_peak = max(self.input_peak, float(np.max(np.abs(x))))
        try:
            self.ring.put_nowait(x)
        except queue.Full:
            pass  # output is behind; drop rather than build latency

    def _out_cb(self, outdata, frames, time_info, status):
        mode, mod_hz, depth, duty, ramp_frac, gain, waveform, const = \
            self.params.snapshot()
        fs = self.fs

        try:
            x = self.ring.get_nowait()
        except queue.Empty:
            self.starved += 1
            outdata.fill(0.0)
            return

        if len(x) < frames:
            x = np.pad(x, (0, frames - len(x)))
        elif len(x) > frames:
            x = x[:frames]

        if mode == "passthrough":
            y = x * gain
        elif mode == "attenuated":
            y = x * const * gain
        else:
            ph = (self.phase + np.arange(frames) * mod_hz / fs) % 1.0
            y = x * envelope(ph, depth, duty, ramp_frac, waveform) * gain
            self.phase = (self.phase + frames * mod_hz / fs) % 1.0

        pk = float(np.max(np.abs(y))) if frames else 0.0
        self.peak = max(self.peak, pk)
        if pk >= 0.999:
            self.clipped += 1
        y = np.clip(y, -1.0, 1.0)
        self.out_sumsq += float(np.sum(y * y))
        self.out_samples += len(y)

        outdata[:, 0] = y
        if outdata.shape[1] > 1:
            outdata[:, 1] = y

        if self.recorded is not None:
            with self.rec_lock:
                self.recorded.append(y.copy())

    def start(self):
        if self.istream is not None:
            return
        self.istream = sd.InputStream(samplerate=self.fs,
                                      blocksize=self.blocksize, channels=1,
                                      dtype="float32", device=self.in_device,
                                      callback=self._in_cb)
        self.ostream = sd.OutputStream(samplerate=self.fs,
                                       blocksize=self.blocksize, channels=2,
                                       dtype="float32", device=self.out_device,
                                       callback=self._out_cb)
        self.istream.start()
        self.ostream.start()

    def stop(self):
        for stream in (self.istream, self.ostream):
            if stream is not None:
                stream.stop()
                stream.close()
        self.istream = None
        self.ostream = None

    @property
    def output_rms(self):
        """RMS of everything played so far (0.0 before any output)."""
        if not self.out_samples:
            return 0.0
        return float(np.sqrt(self.out_sumsq / self.out_samples))

    def device_names(self):
        return (sd.query_devices(self.istream.device)["name"],
                sd.query_devices(self.ostream.device)["name"])

    def save_recording(self, path):
        """Write what was played to a wav file. Returns seconds written."""
        if not self.recorded:
            return 0.0
        import soundfile as sf
        with self.rec_lock:
            y = np.concatenate(self.recorded)
        sf.write(path, y, self.fs)
        return len(y) / self.fs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="passthrough only, for matching level to the open ear")

    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--blocksize", type=int, default=256)
    ap.add_argument("--in-device", default=None)
    ap.add_argument("--out-device", default=None)

    ap.add_argument("--mode", default="modulated",
                    choices=["modulated", "attenuated", "passthrough"])
    ap.add_argument("--mod-hz", type=float, default=10.0)
    ap.add_argument("--depth", type=float, default=0.3)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--ramp", type=float, default=0.05)
    ap.add_argument("--waveform", default="square", choices=["square", "sine"])
    ap.add_argument("--gain", type=float, default=1.0,
                    help="set this once with --calibrate, then leave it alone")
    ap.add_argument("--record", default=None,
                    help="write what was played to this wav file")

    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    for attr in ("in_device", "out_device"):
        v = getattr(args, attr)
        if v is not None and v.isdigit():
            setattr(args, attr, int(v))

    if args.calibrate:
        args.mode = "passthrough"

    gate = RealtimeGate(mode=args.mode, mod_hz=args.mod_hz, depth=args.depth,
                        duty=args.duty, ramp_frac=args.ramp, gain=args.gain,
                        waveform=args.waveform, fs=args.fs,
                        blocksize=args.blocksize, in_device=args.in_device,
                        out_device=args.out_device,
                        record=bool(args.record))
    p = gate.params

    print(CHECKLIST)

    gate.start()
    try:
        in_name, out_name = gate.device_names()
        print(f"in  : {in_name}")
        print(f"out : {out_name}")
        print(f"{args.fs} Hz, blocksize {args.blocksize}")
        print()
        if args.calibrate:
            print("CALIBRATION MODE")
            print("wear ONE earbud, leave the other ear open.")
            print("adjust with  g <value>  until both ears sound equally loud.")
            print("then note the number and pass it as --gain from now on.")
        else:
            print(f"mode {p.mode}   {p.mod_hz:g} Hz   depth {p.depth:g}")
        print()
        print("commands:  m modulated|attenuated|passthrough   f <hz>   "
              "d <0-1>   g <gain>")
        print("           duty <0-1>   ramp <0-0.5>   w square|sine   "
              "s (status)   q")
        print()

        while True:
            try:
                line = input("> ").strip().split()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            cmd = line[0]
            try:
                if cmd == "q":
                    break
                elif cmd == "m":
                    if line[1] not in ("modulated", "attenuated", "passthrough"):
                        print("  unknown mode")
                        continue
                    p.set(mode=line[1])
                elif cmd == "f":
                    p.set(mod_hz=float(line[1]))
                elif cmd == "d":
                    p.set(depth=float(line[1]))
                elif cmd == "g":
                    p.set(gain=float(line[1]))
                elif cmd == "duty":
                    p.set(duty=float(line[1]))
                elif cmd == "ramp":
                    p.set(ramp_frac=float(line[1]))
                elif cmd == "w":
                    p.set(waveform=line[1])
                elif cmd == "s":
                    mode, hz, depth, duty, ramp, gain, wf, const = p.snapshot()
                    print(f"  mode {mode}  {hz:g} Hz  depth {depth:g}  "
                          f"duty {duty:g}  ramp {ramp:g}  {wf}  gain {gain:g}")
                    print(f"  matched constant for control condition: {const:.4f}")
                    print(f"  peak since last check {gate.peak:.3f}   "
                          f"clipped blocks {gate.clipped}   "
                          f"starved blocks {gate.starved}")
                    if gate.starved > 50:
                        print("  many starved blocks: try a larger --blocksize")
                    gate.peak = 0.0
                else:
                    print("  unknown command")
            except (IndexError, ValueError) as e:
                print(f"  bad argument: {e}")
    finally:
        gate.stop()

    if args.record:
        secs = gate.save_recording(args.record)
        print(f"wrote {args.record}  ({secs:.1f} s)")


if __name__ == "__main__":
    main()
