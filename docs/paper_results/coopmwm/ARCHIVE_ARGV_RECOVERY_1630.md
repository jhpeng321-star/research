# Registered archive partial recovery

V2 stopped after 28 certified batches on `OSError(7, 'Argument list too long')`.
No unverified source is deleted. The prior manifest, owners, failed receipt and
all batch receipts remain immutable.

V3 sends the remote verification program through SSH stdin, avoiding the OS
per-argument limit. Six Linux safety tests pass, including a >300 KB payload and
partial-resume provenance checks. It adopts only manifest-matching sources whose
deletions were recorded in prior verified batches, and rechecks their remote SHA
before proceeding. Remaining source files are rechecked against the original
manifest. Destination proof numbering continues at batch 28.

Manifest: `f5c9fde4c3fddc40b24a746024f46c767f8b4c6022d9a5b7a9a5e0a6fccd1ed0`.
Registration: `edac56580314679a22a6488de2016527efa2d1c6d70771535b9994eacb4ee23a`.
Control: `/home/ubuntu/coopmwm20261103_archive1450_v3`.
No simulation or inference. Dispatch follows this commit; adopt its client/remote
owner receipts rather than launching again.
