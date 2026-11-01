# SART with Ambient-Sound Modulation

A Sustained Attention to Response Task (SART) run under eight background-sound
conditions. Participants press F for every digit except 3 and J for 3, and can
correct a wrong press until the next digit appears. Each of the eight blocks
(260 trials, about 5 minutes) uses one sound condition:

| Code | Condition |
|---|---|
| 1 | No headphones (real room) |
| 2 | Live room sound, modulated in real time |
| 3 / 4 | Fan, unmodulated / modulated |
| 5 / 6 | Rain, unmodulated / modulated |
| 7 / 8 | Café, unmodulated / modulated |

Modulation is a 16 Hz square wave with depth 0.5. Loudness is matched to a
5-second room measurement taken at the start of the session.

## Requirements

- macOS laptop with a built-in microphone
- PsychoPy 2021.2.3 (standalone app)
- Noise-cancelling headphones (tested with AirPods Pro)

## Running

| File | Purpose |
|---|---|
| `run_experiment.command` | Full session: practice, then 8 blocks in the order entered |
| `run_practice.command` | Demo version: any block order, block length and sound-phase length |

Double-click a launcher. It starts the program from Terminal, which needs
microphone permission. Data are written to `data/` (`data/practice/` for the
demo). The researcher guide (Word, in Box) covers setup and the session
procedure in detail.

## Repository layout

| Path | Contents |
|---|---|
| `sart_experiment.py` | Main experiment |
| `sart_practice.py` | Demo/practice version |
| `realtime_gate.py` | Real-time modulation of the microphone signal (condition 2) |
| `offline_modulate.py` | Creates the modulated sound files |
| `stimuli/` | The six sound files (6 min each; played from the start and looped if a block is longer) |
