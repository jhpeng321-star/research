# Exposure disclosure (before SUMO)

The first local exposure scanner used a text-suffix allowlist that was too broad.
It opened legacy `driver_params_switching_test/vehicle_params.json` and
`driver_params_switching_val/vehicle_params.json` while looking for six-digit seed
strings. Returned matches were decimal parameter suffixes, not world identities.
No parameter value is used by the audit, selection, controller, or analysis.
This did not open a forbidden-name path, but violated the intended prohibition on
assignment/driver-truth inspection. Original scanner receipt is preserved as
`exposure_local.json`. The corrected scanner prunes parameter/assignment paths
before descent and only opens explicit registration/history filenames. Remote
audits inspect filename metadata only. This disclosure must accompany close-out.
