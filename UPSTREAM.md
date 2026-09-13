# Fork maintenance

This repository is a maintained fork of `NicosNicolaou16/PitchKit`.

- Upstream baseline: `1.0.1`, commit `26a7909601bb33b08623b32605a6f731a3704cdd`
- Local fork line: `1.1.x`
- Review upstream changes and merge them selectively; do not replace the fork blindly.

## 1.1.0 fork changes

- Explicit NOTE / CHORD / AUTO DSP modes.
- NOTE mode avoids FFT work and uses a 4096-sample frame.
- YIN searches only lags valid for the selected instrument and reuses its work buffer.
- CHORD mode performs one padded FFT per frame.
- Chord templates precompute pitch masks instead of allocating sets during scoring.
- Chroma uses peak filtering and magnitude compression to keep weaker chord tones visible.
- Chord switching uses temporal agreement instead of comparing a new chord against a stale previous score.
- Silence resets chord state so the same chord can be acquired again cleanly.
- Realtime listener stops AudioRecord below RESUMED lifecycle state.
- Changing profile/mode rebuilds the engine correctly.
- Per-frame debug logging is removed.
- Offline `PitchAnalyzer` exposes raw note/chord analysis for deterministic tests and diagnostics.
