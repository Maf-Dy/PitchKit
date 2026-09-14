# Modern Pitch Tuner fork changelog

## 1.1.0-modern

Based on upstream PitchKit 1.0.1.

### Correctness

- Added explicit NOTE, CHORD and compatibility AUTO processing modes. Production tuner/chord screens no longer need to guess monophonic vs polyphonic input.
- Added explicit live chord-engine selection: Auto, Crema, ChordNet, BTC Experimental, or Classic DSP.
- Auto currently prefers ChordNet, then Crema, then Classic DSP. BTC Live is experimental and is never selected automatically.
- Added a shared temporal chord-gesture accumulator for ChordNet, Crema and BTC Live. Notes from a normal arpeggio remain active long enough to form one harmony while simultaneous stable chords can resolve earlier.
- The gesture accumulator has a bounded formation window and resets its decision timing when a new pitch arrives after a settled chord.
- Added direct pitch-class and bass evidence to neural live decisions, including conservative DSP rescue for complex chord qualities.
- Expanded shared pitch-content refinement through 6/9, 9/11/13 families and common altered dominants while retaining evidence thresholds to avoid inventing extensions from weak harmonics.
- Reworked Classic DSP scoring and vocabulary to cover 6/9, 9/11/13, diminished/half-diminished and common altered dominant families.
- Added a shared neural `ChordStabilizer` so neural transitions suppress stale labels until a new chord is confirmed.
- Added detector confidence and backend identity to `TuningResult.Chord`.
- Added debug `PitchKitChord` traces showing raw model prediction, confidence, alternatives, gesture state, corrections and emitted chord.
- Silence resets chord state after sustained quiet rather than leaking stale recognition state.
- Neural recognizer close/reset paths are synchronized so engine switching cannot close an ONNX session while that recognizer is still executing inference.

### Performance

- Bounded YIN lag analysis to the selected instrument frequency range instead of scanning every lag up to half the buffer.
- Reused YIN and preprocessing scratch buffers.
- Added pooled microphone float buffers and recycling through the DSP pipeline.
- Reduced note-mode capture window to 4096 samples.
- Removed per-template chord `HashSet` allocation by precomputing pitch masks.
- Removed classic pitch-class `MutableList`, `Pair` and sorting churn by using reusable fixed arrays.
- Reused large classic FFT work arrays per DSP thread instead of allocating real/imaginary/magnitude arrays every chord frame.
- Explicit CHORD mode performs one classic FFT path; compatibility AUTO mode may still do extra routing work and is not used by Modern Pitch Tuner.

### Lifecycle

- Microphone collection runs only while the host lifecycle is `RESUMED`.
- `AudioRecord` is stopped and released when collection is cancelled below `RESUMED`.
- Engine stop is restartable; permanent close is reserved for disposal.
- Neural recognizers are tied to the requested engine selection so stale ChordNet/Crema/BTC instances are not reused after an engine change.

### Live harmony backends

- Added pluggable `ChordRecognizer` API.
- Added ChordNet 2E1D ONNX runtime integration with temporal CQT/DSP fusion.
- Added Crema 0.2.0 ONNX runtime integration with HCQT frontend, chord-tag/root/pitch/bass decoding and the same temporal gesture layer.
- Added explicit BTC Live Experimental using the already-pinned BTC checkpoint. It pads unavailable context into the model's native 108-frame bidirectional window and logs the amount of real context used. It remains benchmark-only until phone latency and accuracy justify keeping it.
- Classic DSP remains an independent non-neural fallback/reference path rather than being folded into the neural implementation.

### Offline harmony backends

- Added shared offline-song timeline models and repeated-section grouping.
- `SongChordSegment` now has optional root, bass, pitch-class and alternative-interpretation fields so future learning/instrument-visualization UI can reuse analysis results without re-running audio.
- LV Song currently remains the strongest offline engine in this project's real-song tests. It keeps the five-model large-vocabulary sequence decoder, separates bass from mid/upper harmony evidence, weights persistent pitches over transient peaks, and refines compatible candidates using the full LV dictionary rather than a small hand-written quality list.
- Crema whole-song Viterbi now fuses chord-tag, root and pitch-content heads instead of discarding the root/pitch outputs; segment results retain root, bass and persistent pitch classes.
- BTC whole-song analysis retains the ChordMini recommended overlap/smoothing path and now verifies each stable segment against the same full-song CQT evidence, retaining root/pitch information for downstream use.
- Added Consonance Decomposed as a fourth explicit offline benchmark engine. Its Conformer predicts root, bass and twelve pitch activations separately from a 144-bin linear CQT and preserves arbitrary pitch-set information instead of forcing every frame into a small fixed chord vocabulary.
- Added `HarmonyBenchmarkEvaluator` to compare all offline engines at the same annotated intervals using independent label, root, bass and pitch-set metrics. Multiple accepted spellings are supported for genuinely ambiguous complex harmony.

### Reproducibility

- BTC export metadata records the exact source commit, checkpoint Git blob SHA, generated ONNX SHA-256, and checkpoint normalization mean/std.
- LV Song model/front-end assets and full dictionary are pinned/generated by the existing LV export scripts.
- Consonance tooling pins source commit `d17633aea4e68e616e735d09df97b97ae3428e71`, verifies checkpoint size/blob identity, exports only the inference graph, and generates a dedicated linear-magnitude Android CQT plan.
- Generated neural assets remain uncommitted and are verified at runtime/export time.

### Tests added or extended

- Synthetic Classic tests cover major/minor, half-diminished, diminished seventh and G6/9.
- Live gesture tests cover normal arpeggio formation, simultaneous-chord fast resolution, bounded formation time, and a new pitch after a settled chord.
- ChordNet/Crema diagnostics expose top alternatives for real-device debugging.
- Consonance decoder tests cover G6/9, F#ø7 and F#dim7.
- LV Song full-dictionary refinement tests distinguish G6/9 from a real G7.
- Shared benchmark tests cover label aliases/enharmonics plus root/bass/pitch-set scoring.

### Deliberately not claimed as solved yet

- No neural engine is declared the final live winner until ChordNet, Crema and BTC Experimental are tested on-device with the same simultaneous/fast-arpeggio/medium-arpeggio/slow-arpeggio/reordered-note matrix.
- BTC Live may prove unsuitable because the checkpoint's native bidirectional context is much longer than a low-latency live gesture. It remains explicitly experimental.
- Consonance ONNX export and Android execution still require a local asset-generation/build run; this changelog does not claim those binaries have been successfully generated on the user's machine yet.
- Complex Anomalie/Jacob Collier-style harmony is not claimed to be perfectly solvable from a dense finished mix. The benchmark separates note/root understanding from exact chord-symbol agreement so improvements are measurable rather than anecdotal.
- Crema HCQT generation remains a major live compute cost and still needs phone profiling.
- Model/checkpoint/training-data provenance must be reviewed before a production/commercial distribution decision.

### Maintenance

The `modern-pitch-tuner` branch is the maintained library source. The consuming app pins an exact commit from this branch as a Git submodule rather than editing a second copy.
