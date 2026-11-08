# Changelog

## 0.1.0 (unreleased)

- First release: timed and edge-triggered captures on Logic MSO via `saleae-mso-api`.
- Measurement, min/max-decimated waveform, digital transition and CSV export tools.
- Persistent capture store; simulated backend for development and CI.
- Omitted `sample_rate_hz` now means the highest rate (1.6 GS/s), divided down to stay
  within `SALEAE_MSO_AUTO_MAX_SAMPLES` per channel; the summary reports the rate actually
  used and notes when the device rounded the requested one.
- Frequency, period, duty cycle and edge times are left empty for noise (irregular edges).
- Fixed: every hardware capture failed because `Coupling` and `BandwidthLimit` were read
  from `saleae.mso_api` instead of `saleae.mso_api.capture_config`.
- Unexpected errors now reach the client with their type and message.
