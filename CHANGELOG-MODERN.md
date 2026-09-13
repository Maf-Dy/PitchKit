# Modern Pitch Tuner fork changelog

## 1.1.0-modern

Based on upstream PitchKit 1.0.1.

### Correctness

- Added explicit NOTE, CHORD and compatibility AUTO processing modes. Production tuner/chord screens no longer need to guess monophonic vs polyphonic input.
- Reworked classic chord transition state so a weak or pending new candidate does not keep reporting the previous chord as live audio.
- Removed three-frame classic chroma averaging. A new chord is evaluated from the current audio frame and temporal stability comes from explicit candidate confirmation instead of mixing old harmony into new harmony.
- Added a shared neural `ChordStabilizer` so Crema/ChordNet transitions suppress stale labels until a new chord is confirmed.
- Added detector confidence and backend identity to `TuningResult.Chord`.
- Added debug `PitchKitChord` traces showing raw neural/decoder prediction, confidence and the stabilized chord actually emitted. This distinguishes model/decoder mistakes from wrapper/state mistakes during real microphone tests.
- Silence now resets chord state after sustained quiet rather than leaking stale recognition state.

### Performance

- Bounded YIN lag analysis to the selected instrument frequency range instead of scanning every lag up to half the buffer.
- Reused YIN and preprocessing scratch buffers.
- Added pooled microphone float buffers and recycling through the DSP pipeline.
- Reduced note-mode capture window to 4096 samples.
- Removed per-template chord `HashSet` allocation by precomputing pitch masks.
- Removed classic pitch-class `MutableList`, `Pair` and sorting churn by using reusable fixed arrays.
- Reused large classic FFT work arrays per DSP thread instead of allocating real/imaginary/magnitude arrays every chord frame.
- Explicit CHORD mode performs one classic FFT path; the compatibility AUTO mode may still do extra routing work and is not used by Modern Pitch Tuner.

### Lifecycle

- Microphone collection now runs only while the host lifecycle is `RESUMED`.
- `AudioRecord` is stopped and released when collection is cancelled below `RESUMED`.
- Engine stop is restartable; permanent close is reserved for disposal.

### Harmony backends

- Added pluggable `ChordRecognizer` API.
- Added ChordNet 2E1D ONNX runtime integration.
- Added Crema 0.2.0 ONNX runtime integration with HCQT frontend and multi-head harmony decoder.
- Restored deterministic tests for note mapping, stereo-to-mono conversion, classic synthetic guitar chords, ChordNet vocabulary/post-processing and Crema harmony decoding.
- Added regression tests specifically for stale chord transitions and bounded YIN pitch accuracy.

### Deliberately not claimed as solved yet

- The minimum neural confidence cutoff is not being raised arbitrarily. Confidence is now observable, so it should be calibrated against recorded/labelled guitar tests before changing the acceptance threshold.
- Crema HCQT generation is still the dominant cost in neural chord mode and remains a separate performance target.
- Repeated strums of the same chord are not separate chord identities. A future practice/training mode still needs onset/strum detection if it must count `C` -> `C` -> `C` as three events.

### Maintenance

The `modern-pitch-tuner` branch is the maintained library source. The consuming app pins an exact commit from this branch as a Git submodule rather than editing a second copy.
