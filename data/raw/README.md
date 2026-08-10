# Raw Data

Raw timing profiles collected from the remote GPU server during the profiling phase.

Due to the size of the raw profiling data (~2 GB), this directory is not tracked by git.
Profiling data can be reproduced by running:

```bash
python3 -m src.evaluation.evaluate
```

This regenerates synthetic timing profiles with seed=42 as described in Table II of the paper.
