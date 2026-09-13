# Modern Pitch Tuner fork changelog

## 1.1.0-modern

Based on upstream PitchKit 1.0.1.

### Correctness

- Added explicit NOTE, CHORD and compatibility AUTO processing modes. Production tuner/chord screens no longer need to guess monophonic vs polyphonic input.
- Reworked classic chord transition state so a weak or pending new candidate does not keep reporting the previous chord as live audio.
- Added a shared neural `ChordStabilizer` so Crema/ChordNet transitions suppress stale labels until a new chord is confirmed.
- Added detector confidence and backend identity to `TuningResult.Chord`.
- Silence now resets chord state after sustained quiet rather than leaking stale recognition state.

### Performance

- Bounded YIN lag analysis to the selected instrument frequency range instead of scanning every lag up to half the buffer.
- Reused YIN and preprocessing scratch buffers.
- Added pooled microphone float buffers and recycling through the DSP pipeline.
- Reduced note-mode capture window to 4096 samples.
- Removed per-template chord `HashSet` allocation by precomputing pitch masks.
- Reduced classic chord scoring allocation churn.

### Lifecycle

- Microphone collection now runs only while the host lifecycle is `RESUMED`.
- `AudioRecord` is stopped and released when collection is cancelled below `RESUMED`.
- Engine stop is restartable; permanent close is reserved for disposal.

### Harmony backends

- Added pluggable `ChordRecognizer` API.
- Added ChordNet 2E1D ONNX runtime integration.
- Added Crema 0.2.0 ONNX runtime integration with HCQT frontend and multi-head harmony decoder.
- Added deterministic tests for stale chord transitions and bounded YIN pitch accuracy.

### Maintenance

The `modern-pitch-tuner` branch is the maintained library source. The consuming app should pin it as a Git submodule rather than editing a second copy.
