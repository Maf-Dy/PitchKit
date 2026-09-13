# Modern Pitch Tuner fork changelog

## 1.1.0-modern

Based on upstream PitchKit 1.0.1.

### Correctness

- Added explicit NOTE, CHORD and compatibility AUTO processing modes. Production tuner/chord screens no longer need to guess monophonic vs polyphonic input.
- Added explicit live chord-engine selection: AUTO, Crema, ChordNet, or Classic DSP. AUTO now tries Crema, then ChordNet, then Classic DSP; a Crema load failure no longer skips the ChordNet fallback.
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
- Added shared offline-song timeline models and repeated-section grouping.
- Added Crema whole-song analysis with sequence-level Viterbi decoding.
- Ported the LV-Chordia / LV Song large-vocabulary five-model ensemble as an offline comparison backend. It is retained for comparison because its prior real-song result was not accurate enough for the app's needs.
- Added an experimental BTC whole-song backend based on the newer ChordMini continual-learning BTC checkpoint. The Android path uses 22.05 kHz audio, 144-bin log-CQT, overlapping 108-frame windows, logit aggregation, Gaussian smoothing, categorical smoothing, and minimum segment-duration cleanup.
- BTC export metadata records the exact source commit, checkpoint Git blob SHA, generated ONNX SHA-256, and checkpoint normalization mean/std so the Android frontend cannot silently drift from the exported model.
- Restored deterministic tests for note mapping, stereo-to-mono conversion, classic synthetic guitar chords, ChordNet vocabulary/post-processing and Crema harmony decoding.
- Added regression tests specifically for stale chord transitions and bounded YIN pitch accuracy.
- Added tests for BTC export metadata validation and shared song-section grouping.

### Deliberately not claimed as solved yet

- The minimum neural confidence cutoff is not being raised arbitrarily. Confidence is now observable, so it should be calibrated against recorded/labelled guitar tests before changing the acceptance threshold.
- Crema HCQT generation is still the dominant cost in neural chord mode and remains a separate performance target.
- Repeated strums of the same chord are not separate chord identities. A future practice/training mode still needs onset/strum detection if it must count `C` -> `C` -> `C` as three events.
- BTC is a new candidate, not a claimed accuracy win. Its ONNX export/runtime path still needs local model export plus the same problematic song used to reject LV Song before BTC can be accepted as the production offline analyzer.
- LV Song remains available only as an offline comparison backend; its previous whole-song result was reported inaccurate and is not treated as the preferred analyzer.

### Maintenance

The `modern-pitch-tuner` branch is the maintained library source. The consuming app pins an exact commit from this branch as a Git submodule rather than editing a second copy.
