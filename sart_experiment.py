#!/usr/bin/env python3
"""
SART (Sustained Attention to Response Task) with Error Awareness
Ambient Sound Modulation Pilot Study

Design: 4 sound sources x unmodulated / modulated = 8 conditions
    1 = no_headphone  (no headphone, no sound: the room heard as is)
    2 = realtime_gate (ANC headphone + live room sound from the built-in
                       mic, modulated in real time by realtime_gate.py)
    3 = fan           (ANC headphone + fan.wav, unmodulated)
    4 = fan_mod       (ANC headphone + fan_mod.wav, modulated)
    5 = rain          (ANC headphone + rain.wav, unmodulated)
    6 = rain_mod      (ANC headphone + rain_mod.wav, modulated)
    7 = cafe          (ANC headphone + cafe.wav, unmodulated)
    8 = cafe_mod      (ANC headphone + cafe_mod.wav, modulated)
- Target digit: 3 (no-go), all others: go
- Go response: F key
- No-go response: J key
- Response / correction window: the whole trial, from digit onset until the
  next digit appears (TRIAL_DURATION = 1150 ms)
- 8 blocks × ~260 trials (~5 min each), one per condition
- Block order entered manually by experimenter at startup (a permutation of 1–8)

Data columns:
  trial          trial number (1-indexed, continuous across blocks)
  block          block number (1–8)
  sound_cond     condition code (1–8, see above)
  cond_label     condition name, e.g. fan_mod (codes changed between design
                 versions, so analyses should rely on this column)
  digit          digit shown (0–9)
  is_nogo        1 if target (3), 0 otherwise
  correct_key    0=F (go), 1=J (nogo)
  first_key      0=F, 1=J, -1=no response
  first_rt       RT of first keypress in ms (-1 if none)
  n_keypresses   number of F/J presses before the next digit appeared
  correction     1 if there were exactly 2 presses, 0 otherwise
  correction_key key of the 2nd press: 0=F, 1=J, -1 if correction == 0
  correction_rt  RT of the 2nd press in ms (-1 if correction == 0)
  accuracy       1=correct, 0=error
  awareness      0=unaware correct, 1=aware error,
                 2=unaware error, 3=aware correct (unnecessary correction),
                 -1=more than 2 presses (correction not scored)

Audio files needed (in stimuli/ next to this script):
  fan.wav, rain.wav, cafe.wav              unmodulated originals
  fan_mod.wav, rain_mod.wav, cafe_mod.wav  modulated with offline_modulate.py
                                           (16 Hz square wave, depth 0.5)
realtime_gate.py must sit next to this script (used by the realtime_gate
condition). All audio goes through sounddevice/soundfile (bundled with
PsychoPy), not psychopy.sound, so the device list can be rescanned per block.
Output CSVs (plus a per-session _audio.txt log) are written to data/.
"""

import math
import os
import random
import csv
import subprocess
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from psychopy import visual, core, event, data, gui

from realtime_gate import RealtimeGate, matched_constant

# ─── Experiment parameters ───────────────────────────────────────────────────
DIGIT_DURATION    = 0.250   # seconds
BLANK_DURATION    = 0.900   # seconds (ISI)
TRIAL_DURATION    = DIGIT_DURATION + BLANK_DURATION  # 1150 ms
CORRECTION_WINDOW = TRIAL_DURATION  # keys count until the next digit appears
BLOCK_DURATION    = 5 * 60  # 5 minutes per block in seconds
TARGET_DIGIT      = 3
NOGO_PROB         = 0.10    # 10% no-go trials
GO_KEY            = 'f'
NOGO_KEY          = 'j'
QUIT_KEY          = 'escape'

# Condition labels (1–8)
COND_LABELS = {
    1: 'no_headphone',
    2: 'realtime_gate',
    3: 'fan',
    4: 'fan_mod',
    5: 'rain',
    6: 'rain_mod',
    7: 'cafe',
    8: 'cafe_mod',
}
N_BLOCKS = len(COND_LABELS)

# Headphone instructions per condition
HEADPHONE_INSTRUCTIONS = {
    1: 'Please REMOVE the headphone (if wearing one).',
    2: 'Please PUT ON the headphone with noise cancellation ON.',
    3: 'Please PUT ON the headphone with noise cancellation ON.',
    4: 'Please PUT ON the headphone with noise cancellation ON.',
    5: 'Please PUT ON the headphone with noise cancellation ON.',
    6: 'Please PUT ON the headphone with noise cancellation ON.',
    7: 'Please PUT ON the headphone with noise cancellation ON.',
    8: 'Please PUT ON the headphone with noise cancellation ON.',
}

# Folders (resolved relative to this script, not the working directory)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STIMULI_DIR = os.path.join(BASE_DIR, 'stimuli')
DATA_DIR    = os.path.join(BASE_DIR, 'data')

# Real-time gating (condition 2): built-in mic -> modulation -> headphones.
# Defaults match the offline-modulated WAVs (16 Hz square wave, depth 0.5).
GATE_COND       = 2
GATE_MOD_HZ     = 16.0
GATE_DEPTH      = 0.5
GATE_GAIN       = 1.0    # set once with: python3 realtime_gate.py --calibrate
GATE_IN_DEVICE  = None   # None = auto-pick the built-in Mac mic (see below)
GATE_OUT_DEVICE = None   # None = system default output (the headphones)

# Loudness matching: at session start the room is measured through the mic
# for ROOM_MEASURE_DURATION s, and the WAVs are scaled toward the RMS the
# realtime_gate block will play (room RMS x envelope RMS x GATE_GAIN).
# The two versions of a sound (e.g. fan / fan_mod) always get the same RMS:
# the pair's target is lowered until both files can reach it without clipping,
# and, for files in NO_BOOST, without being made louder than they already are.
# Set False to play WAVs unscaled.
MATCH_WAV_TO_GATE     = True
ROOM_MEASURE_DURATION = 5    # seconds
# NO_BOOST files may be turned down, never up: rain.wav already peaks at full
# scale (1.0), so any boost would clip, and rain_mod.wav is paired with it.
# Rain is loud enough for the (quiet) rooms used in this study anyway.
NO_BOOST              = ('rain.wav', 'rain_mod.wav')

# Each block opens with SOUND_INTRO_DURATION s of sound only (or silence in the
# no-sound condition); the trials start as soon as it ends.
SOUND_INTRO_DURATION = 20    # seconds

# Audio files (conditions 3–8)
AUDIO_FILES = {
    3: os.path.join(STIMULI_DIR, 'fan.wav'),
    4: os.path.join(STIMULI_DIR, 'fan_mod.wav'),
    5: os.path.join(STIMULI_DIR, 'rain.wav'),
    6: os.path.join(STIMULI_DIR, 'rain_mod.wav'),
    7: os.path.join(STIMULI_DIR, 'cafe.wav'),
    8: os.path.join(STIMULI_DIR, 'cafe_mod.wav'),
}

# ─── Relaunch through Terminal if needed ─────────────────────────────────────
# macOS grants microphone access to the app that launched us. PsychoPy 2021.2.3
# can never be granted it, so when started from PsychoPy's Run button (no
# TERM_PROGRAM set), reopen via run_experiment.command in Terminal and exit.
if not os.environ.get('TERM_PROGRAM'):
    launcher = os.path.join(BASE_DIR, 'run_experiment.command')
    subprocess.Popen(['open', '-a', 'Terminal', launcher])
    print('Relaunching in Terminal (needed for microphone access)...')
    core.quit()

# ─── Participant info dialog ──────────────────────────────────────────────────
exp_info = {
    'participant_id': '',
    'block_order (e.g. 1,2,3,4,5,6,7,8)': '1,2,3,4,5,6,7,8',
}
dlg = gui.DlgFromDict(
    exp_info,
    title='SART Pilot',
    order=['participant_id', 'block_order (e.g. 1,2,3,4,5,6,7,8)']
)
if not dlg.OK:
    core.quit()

pid = exp_info['participant_id']
order_str = exp_info['block_order (e.g. 1,2,3,4,5,6,7,8)']
try:
    condition_order = [int(x.strip()) for x in order_str.split(',')]
    # Every condition exactly once
    assert sorted(condition_order) == sorted(COND_LABELS)
except Exception:
    print(f"ERROR: block_order must use each of 1-{N_BLOCKS} exactly once, "
          f"comma-separated (e.g. 3,8,1,6,2,5,7,4)")
    core.quit()

# ─── Output file ─────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(DATA_DIR, exist_ok=True)
output_file = os.path.join(DATA_DIR, f'sart_{pid}_{timestamp}.csv')
audio_log_file = os.path.join(DATA_DIR, f'sart_{pid}_{timestamp}_audio.txt')

fieldnames = [
    'trial', 'block', 'sound_cond', 'cond_label', 'digit', 'is_nogo',
    'correct_key', 'first_key', 'first_rt', 'n_keypresses', 'correction',
    'correction_key', 'correction_rt', 'accuracy', 'awareness'
]

# ─── Window setup ────────────────────────────────────────────────────────────
win = visual.Window(
    fullscr=True,
    color='black',
    units='height'
)
win.mouseVisible = False

# ─── Stimuli ─────────────────────────────────────────────────────────────────
digit_stim = visual.TextStim(
    win, text='', height=0.15, color='white', bold=True
)
fix_stim = visual.TextStim(
    win, text='+', height=0.08, color='gray'
)
msg_stim = visual.TextStim(
    win, text='', height=0.05, color='white', wrapWidth=1.5
)

# ─── Audio ────────────────────────────────────────────────────────────────────
# Output device names that mean "the laptop is playing out loud"
SPEAKER_KEYWORDS = ('Speakers', 'Built-in Output')

# True: before each headphone block, wait until the audio output is something
# other than the laptop speakers. False: play on whatever the current output
# is. Note that the realtime_gate block on laptop speakers feeds the speakers
# back into the mic and howls.
REQUIRE_HEADPHONE_OUTPUT = False

def rescan_audio_devices():
    """Re-enumerate audio devices.

    PortAudio scans devices only once, at startup. AirPods taken off for a
    no-headphone block come back as a new device that a stale list can't see,
    so audio would fall back to the laptop speakers. Only call this while no
    stream is open.
    """
    sd._terminate()
    sd._initialize()

def find_builtin_mic():
    """Index of the Mac's built-in microphone, or None (= system default).

    The system default input is often the AirPods, which drops the Bluetooth
    link into low-quality telephony mode, so never rely on it.
    """
    keywords = ('MacBook', 'iMac', 'Built-in', '内建')
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and any(k in dev['name'] for k in keywords):
            return idx
    print('WARNING: built-in microphone not found, using system default input')
    return None

def launching_app():
    """Which app started us; macOS grants mic permission to this app."""
    term = os.environ.get('TERM_PROGRAM', '')
    if term == 'Apple_Terminal':
        return 'Terminal'
    if term == 'vscode':
        return 'Visual Studio Code'
    if term:
        return term
    return 'PsychoPy (or another app)'

def wait_for_headphones():
    """Rescan devices until the default output is not the laptop speakers.

    With REQUIRE_HEADPHONE_OUTPUT off, only rescans. Returns the output name.
    """
    while True:
        rescan_audio_devices()
        out_name = sd.query_devices(kind='output')['name']
        if not REQUIRE_HEADPHONE_OUTPUT:
            return out_name
        if not any(k in out_name for k in SPEAKER_KEYWORDS):
            return out_name
        msg_stim.text = (
            'The headphones are not connected as the audio output.\n\n'
            'Please put the headphones on (noise cancellation ON)\n'
            'and wait a few seconds for them to connect.\n\n'
            'Press SPACE to try again, or ESC to quit.'
        )
        msg_stim.draw()
        win.flip()
        if QUIT_KEY in event.waitKeys(keyList=['space', QUIT_KEY]):
            win.close()
            core.quit()

def write_audio_log(block_num, condition, fields):
    with open(audio_log_file, 'a', encoding='utf-8') as log:
        log.write(f'block={block_num} cond={COND_LABELS[condition]} {fields}\n')

class LoopPlayer:
    """Loop a WAV file on the current default output device.

    With target_rms, the file is scaled to that RMS (full scale = 1.0), but
    never so far that its peak would clip.
    """

    def __init__(self, path, target_rms=None):
        data, fs = sf.read(path, dtype='float32', always_2d=True)
        self.file_rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        self.gain = 1.0
        if target_rms and self.file_rms > 0:
            self.gain = min(target_rms / self.file_rms, 0.99 / peak)
        # Keep int16 in memory: each file is 6 min of stereo audio
        self.data = np.round(data * self.gain * 32767).astype(np.int16)
        del data
        self.pos = 0
        self.stream = sd.OutputStream(samplerate=fs,
                                      channels=self.data.shape[1],
                                      dtype='int16', callback=self._callback)

    def _callback(self, outdata, frames, time_info, status):
        n = len(self.data)
        written = 0
        while written < frames:
            k = min(frames - written, n - self.pos)
            outdata[written:written + k] = self.data[self.pos:self.pos + k]
            written += k
            self.pos = (self.pos + k) % n

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

def start_audio(condition, block_num):
    """Start the audio for a condition. Returns a function that stops it."""
    if condition != GATE_COND and condition not in AUDIO_FILES:
        return lambda: None  # no-headphone block

    out_name = wait_for_headphones()

    if condition == GATE_COND:
        in_device = GATE_IN_DEVICE
        if in_device is None:
            in_device = find_builtin_mic()
        gate = RealtimeGate(mod_hz=GATE_MOD_HZ, depth=GATE_DEPTH,
                            gain=GATE_GAIN, in_device=in_device,
                            out_device=GATE_OUT_DEVICE)
        gate.start()
        in_name, out_name = gate.device_names()
        # PsychoPy's console is ASCII-only; device names may contain e.g. Chinese
        print(ascii(f"realtime_gate: in = {in_name}, out = {out_name}"))

        def write_gate_log(note=''):
            write_audio_log(block_num, condition,
                f'launched_from="{launching_app()}" '
                f'in="{in_name}" out="{out_name}" '
                f'mod_hz={GATE_MOD_HZ} depth={GATE_DEPTH} gain={GATE_GAIN} '
                f'input_peak={gate.input_peak:.4f} '
                f'output_peak={gate.peak:.4f} '
                f'output_rms={gate.output_rms:.5f} '
                f'clipped_blocks={gate.clipped} '
                f'starved_blocks={gate.starved}{note}')

        # A mic without permission delivers pure zeros instead of an error
        core.wait(1.0)
        if gate.input_peak == 0.0:
            msg_stim.text = (
                'WARNING: the microphone is not picking up any sound.\n\n'
                f'Launched from: {launching_app()}\n'
                'Check System Settings > Privacy & Security > Microphone\n'
                'and make sure this app is allowed.\n\n'
                'Please contact the researcher.\n\n'
                'Press SPACE to continue anyway, or ESC to quit.'
            )
            msg_stim.draw()
            win.flip()
            if QUIT_KEY in event.waitKeys(keyList=['space', QUIT_KEY]):
                gate.stop()
                write_gate_log(' quit_at_mic_warning')
                win.close()
                core.quit()

        def stop():
            gate.stop()
            write_gate_log()
        return stop

    path = AUDIO_FILES[condition]
    if not os.path.exists(path):
        print(ascii(f"WARNING: {path} not found. Block will run without audio."))
        write_audio_log(block_num, condition, f'out="{out_name}" file_missing')
        return lambda: None
    player = LoopPlayer(path, target_rms=wav_targets.get(condition))
    player.start()
    print(ascii(f"{os.path.basename(path)}: out = {out_name}"))
    write_audio_log(block_num, condition,
                    f'out="{out_name}" file={os.path.basename(path)} '
                    f'file_rms={player.file_rms:.5f} '
                    f'gain_db={20 * math.log10(player.gain):+.1f} '
                    f'played_rms={player.file_rms * player.gain:.5f}')
    return player.stop

def file_level(path):
    """(RMS, peak) of a WAV, read in chunks to keep memory low."""
    sumsq, n, peak = 0.0, 0, 0.0
    for chunk in sf.blocks(path, blocksize=1 << 20, dtype='float32', always_2d=True):
        sumsq += float(np.sum(chunk.astype(np.float64) ** 2))
        n += chunk.size
        peak = max(peak, float(np.max(np.abs(chunk))))
    return math.sqrt(sumsq / n), peak

def measure_room_rms():
    """Record the room through the built-in mic. Returns RMS (0.0 = no input)."""
    rescan_audio_devices()
    mic = GATE_IN_DEVICE if GATE_IN_DEVICE is not None else find_builtin_mic()
    fs = 48000
    rec = sd.rec(int(ROOM_MEASURE_DURATION * fs), samplerate=fs, channels=1,
                 dtype='float32', device=mic)
    clock = core.Clock()
    while clock.getTime() < ROOM_MEASURE_DURATION:
        left = math.ceil(ROOM_MEASURE_DURATION - clock.getTime())
        msg_stim.text = (
            'Measuring the room sound.\n\n'
            'Please stay quiet and do not touch the keyboard.\n\n'
            f'{left} s'
        )
        msg_stim.draw()
        win.flip()
    sd.wait()
    return float(np.sqrt(np.mean(rec ** 2)))

def timed_screen(make_text, duration, stop_audio):
    """Show make_text(seconds_left) for duration s; ESC quits."""
    clock = core.Clock()
    while clock.getTime() < duration:
        msg_stim.text = make_text(math.ceil(duration - clock.getTime()))
        msg_stim.draw()
        win.flip()
        if event.getKeys(keyList=[QUIT_KEY]):
            stop_audio()
            win.close()
            core.quit()

# ─── Helper functions ─────────────────────────────────────────────────────────
def show_message(text, wait_key=True, duration=None):
    msg_stim.text = text
    msg_stim.draw()
    win.flip()
    if wait_key:
        if QUIT_KEY in event.waitKeys(keyList=['space', QUIT_KEY]):
            win.close()
            core.quit()
    elif duration:
        core.wait(duration)

def generate_digit_sequence(n_trials):
    """Generate digit sequence with ~10% target (3). No consecutive targets."""
    digits = []
    non_targets = [d for d in range(10) if d != TARGET_DIGIT]
    for _ in range(n_trials):
        if digits and digits[-1] == TARGET_DIGIT:
            digits.append(random.choice(non_targets))
        elif random.random() < NOGO_PROB:
            digits.append(TARGET_DIGIT)
        else:
            digits.append(random.choice(non_targets))
    return digits

def compute_awareness(accuracy, corrected, n_keypresses):
    """
    0 = unaware correct   (correct, no correction)
    1 = aware error       (error, corrected within window)
    2 = unaware error     (error, not corrected)
    3 = aware correct     (correct, but unnecessarily corrected)
   -1 = not scored        (more than 2 presses: no single correction to score)
    """
    if n_keypresses > 2:
        return -1
    if accuracy == 1 and corrected == 0:
        return 0
    elif accuracy == 0 and corrected == 1:
        return 1
    elif accuracy == 0 and corrected == 0:
        return 2
    else:  # accuracy == 1 and corrected == 1
        return 3

def run_block(block_num, condition, digits, writer, trial_offset):
    """Run one block of SART trials. Returns number of trials run."""

    stop_audio = start_audio(condition, block_num)

    # Sound-only phase with a countdown; the first trial follows directly
    if condition == GATE_COND or condition in AUDIO_FILES:
        intro = ('Sound phase\n\n'
                 'The background sound for this block is now playing.\n'
                 'Just listen. No response is needed.\n\n'
                 'The task starts in {} s')
    else:
        intro = ('Quiet phase\n\n'
                 'Please sit quietly. No response is needed.\n\n'
                 'The task starts in {} s')
    timed_screen(intro.format, SOUND_INTRO_DURATION, stop_audio)

    for i, digit in enumerate(digits):
        trial_num = trial_offset + i + 1
        is_nogo = 1 if digit == TARGET_DIGIT else 0
        correct_key_label = 1 if is_nogo else 0  # 0=left, 1=right

        # ── Show digit ──
        digit_stim.text = str(digit)
        digit_stim.draw()
        win.flip()
        trial_clock = core.Clock()
        event.clearEvents()

        first_key     = -1
        first_rt      = -1
        presses       = []   # (key code, rt in ms) for every F/J press

        # ── Collect responses within CORRECTION_WINDOW ──
        while trial_clock.getTime() < CORRECTION_WINDOW:
            if trial_clock.getTime() >= DIGIT_DURATION:
                fix_stim.draw()
                win.flip()

            keys = event.getKeys(
                keyList=[GO_KEY, NOGO_KEY, QUIT_KEY],
                timeStamped=trial_clock
            )
            for key, t in keys:
                if key == QUIT_KEY:
                    stop_audio()
                    win.close()
                    core.quit()

                presses.append((0 if key == GO_KEY else 1, round(t * 1000)))

        if presses:
            first_key, first_rt = presses[0]
        n_keypresses = len(presses)

        # Only a clean 2nd press counts as a correction; with 3+ presses
        # there is no single correction to score, so record none.
        if n_keypresses == 2:
            correction = 1
            correction_key, correction_rt = presses[1]
        else:
            correction = 0
            correction_key, correction_rt = -1, -1

        # ── Wait out remaining ITI ──
        remaining = TRIAL_DURATION - trial_clock.getTime()
        if remaining > 0:
            fix_stim.draw()
            win.flip()
            core.wait(remaining, hogCPUperiod=0)

        # ── Compute accuracy ──
        # Every trial needs a key press: F on go, J on no-go. No response is an error.
        accuracy = 1 if first_key == correct_key_label else 0

        awareness = compute_awareness(accuracy, correction, n_keypresses)

        # ── Write row ──
        writer.writerow({
            'trial':         trial_num,
            'block':         block_num,
            'sound_cond':    condition,
            'cond_label':    COND_LABELS[condition],
            'digit':         digit,
            'is_nogo':       is_nogo,
            'correct_key':   correct_key_label,
            'first_key':     first_key,
            'first_rt':      first_rt,
            'n_keypresses':  n_keypresses,
            'correction':    correction,
            'correction_key': correction_key,
            'correction_rt': correction_rt,
            'accuracy':      accuracy,
            'awareness':     awareness,
        })

        # ── Escape during ITI ──
        if event.getKeys(keyList=[QUIT_KEY]):
            stop_audio()
            win.close()
            core.quit()

    stop_audio()

# ─── Instructions ─────────────────────────────────────────────────────────────
instructions = f"""
SART Task

You will see a series of digits (0–9) appear on the screen, one at a time.

  For every digit EXCEPT {TARGET_DIGIT}:  press the F key  (left hand)
  For the digit {TARGET_DIGIT}:           press the J key  (right hand)

Respond as quickly as you can while staying accurate.

If you realize you pressed the wrong key, you can press the correct key
to correct yourself — but only before the next digit appears.

The task will run in {N_BLOCKS} blocks of about 5 minutes each.
You will have a short break between blocks.

Press SPACE to begin the practice.
"""

practice_end = f"""
Practice complete.

Remember:
  All digits except {TARGET_DIGIT}  ->  F key  (left hand)
  Digit {TARGET_DIGIT}              ->  J key  (right hand)

You can correct errors before the next digit appears.

Press SPACE to continue.
"""

# ─── Main experiment ──────────────────────────────────────────────────────────
n_trials_per_block = int(BLOCK_DURATION / TRIAL_DURATION)  # ~260

show_message(instructions)

# ── Room measurement for loudness matching ──
wav_targets = {}   # condition -> RMS to play its WAV at
if MATCH_WAV_TO_GATE:
    show_message('Before the practice, the computer will measure the sound of\n'
                 f'the room for {ROOM_MEASURE_DURATION} seconds.\n\n'
                 'Please stay quiet and do not touch the keyboard.\n\n'
                 'Press SPACE to start the measurement.')
    room_rms = measure_room_rms()
    env_rms = matched_constant(GATE_DEPTH, 0.5, 0.05, 'square')
    gate_rms = room_rms * env_rms * GATE_GAIN
    log_lines = [f'room_measurement room_rms={room_rms:.5f} gate_rms={gate_rms:.5f}'
                 f'{"" if room_rms > 0 else " mic_silent_wavs_unscaled"}']
    if room_rms > 0:
        # Group conditions by sound: fan.wav and fan_mod.wav form the pair 'fan'
        pairs = {}
        for cond, path in AUDIO_FILES.items():
            if os.path.exists(path):
                stem = os.path.basename(path)[:-len('.wav')]
                pairs.setdefault(stem.replace('_mod', ''), []).append((cond, path))
        for sound_name, members in pairs.items():
            target = gate_rms
            for cond, path in members:
                rms, peak = file_level(path)
                target = min(target, rms * 0.99 / peak)          # no clipping
                if os.path.basename(path) in NO_BOOST:
                    target = min(target, rms)                    # no boost
            for cond, _ in members:
                wav_targets[cond] = target
            log_lines.append(f'pair={sound_name} target_rms={target:.5f}'
                             f'{" lowered_below_gate" if target < gate_rms else ""}')
    with open(audio_log_file, 'a', encoding='utf-8') as log:
        log.write('\n'.join(log_lines) + '\n')

# ── Practice block (no audio, no data saved) ──
practice_digits = generate_digit_sequence(20)
show_message(f'Practice: {len(practice_digits)} trials, no audio.\n\nPress SPACE to begin.', wait_key=True)

for digit in practice_digits:
    digit_stim.text = str(digit)
    digit_stim.draw()
    win.flip()
    t_clock = core.Clock()
    event.clearEvents()
    while t_clock.getTime() < CORRECTION_WINDOW:
        if t_clock.getTime() >= DIGIT_DURATION:
            fix_stim.draw()
            win.flip()
        keys = event.getKeys(keyList=[GO_KEY, NOGO_KEY, QUIT_KEY])
        if QUIT_KEY in keys:
            win.close()
            core.quit()
    remaining = TRIAL_DURATION - t_clock.getTime()
    if remaining > 0:
        fix_stim.draw()
        win.flip()
        core.wait(remaining, hogCPUperiod=0)

show_message(practice_end)

# ── Main blocks ──
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    trial_offset = 0

    for block_idx, condition in enumerate(condition_order):
        block_num = block_idx + 1
        cond_name = COND_LABELS[condition]
        headphone_msg = HEADPHONE_INSTRUCTIONS[condition]

        # Break / setup screen between blocks
        if block_idx == 0:
            break_text = (
                f'Block 1 of {N_BLOCKS}\n\n'
                f'{headphone_msg}\n\n'
                f'Press SPACE when ready.'
            )
        else:
            break_text = (
                f'Block {block_num} of {N_BLOCKS}\n\n'
                f'Take a short break.\n\n'
                f'{headphone_msg}\n\n'
                f'Press SPACE when ready.'
            )

        show_message(break_text, wait_key=True)

        digits = generate_digit_sequence(n_trials_per_block)
        run_block(block_num, condition, digits, writer, trial_offset)
        trial_offset += len(digits)

show_message('Task complete! Thank you.\n\nPress SPACE to exit.', wait_key=True)
win.close()
core.quit()
