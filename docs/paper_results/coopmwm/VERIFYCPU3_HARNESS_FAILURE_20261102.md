# CPU3 qualification harness attempt 1

The frozen CPU3 files staged successfully. The remote generated test payload
failed parsing with `IndentationError: unexpected indent` on the pytest invocation,
before imports/tests or any SUMO/inference. This was an operator-harness spacing
fault, repaired locally. The immutable CPU3 implementation is unchanged. A second
test invocation may adopt exactly those files; this is not a physics retry.
