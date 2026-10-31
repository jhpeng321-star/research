# Exact EXAM byte preservation

An independent local-versus-Git blob audit caught that the first staging pass
had normalized CSV CRLF before the scoped `-text` attribute took effect.
Local transferred artifacts and their original manifests remained correct.
Forced re-indexing under the scoped byte-preserving attribute restores original
bytes in Git; no label/feature/manifest value is changed or regenerated.
Consumer hashes apply to original bytes, not a newline-normalized representation.
The audit compares every manifest member both locally and in committed Git blobs.
