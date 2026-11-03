# Prepared archive overlap — safe decline, no duplicate transfer

The one-use label-runner-copy relay exited before manifest creation, transfer or
source removal. Its dependency guard found the exact source paths in another
session's prepared archive manifest:

- Other root: `/home/ubuntu/coopmwm20261103_archive1450`.
- ManifestSHA`f5c9fde4c3fddc40b24a746024f46c767f8b4c6022d9a5b7a9a5e0a6fccd1ed0`.
- AuthoritySHA`51f152c9185c77605b9d9227dfbae278c453818938b2b4669704c228b5d0aa28`.
- Prepared7,391files /5,048,777,074bytes; four CPU safety tests passed there.
- Prepare owner3873941/start70490505/bootb8bc1fcb completed. At unix1793691348,
  no process with that root in argv was running; metadata says manifest-only,
  moved_bytes0. This session does not start or duplicate that session's transfer.

Our relay: local33076/create1793691235.5774357; remote3898748/start70551749,
now exited. Its claim/dispatch/error are retained and must not be reissued.
The smaller interim-tape archive remains successfully completed and SHA-verified.

Disk free at unix1793691348 was1,823,289,344bytes, below2GiB again as resumed
training outputs consumed the recovered margin. B-u12 completed15:23:03+08;
at15:25:55 B-u13 had14/24 results and R-u12 had12/24, with all original owners
live/noalerts. Keep the guard; coordinate/adopt the existing larger archive plan.
Original bank inputs, canonical branch RESULTs/observables and checkpoints remain.
