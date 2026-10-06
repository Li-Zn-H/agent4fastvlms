# Small CPU Example

Run from the repository root with Python 3.12:

```bash
python3.12 -B examples/scoring_demo.py
```

This loads [one synthetic row](samples.jsonl), validates its local image path and
scores two supplied answers using the framework's real scorer. Expected output
is retention `1.0` for "red" and `0.0` for "blue". It does not load a model, use a
GPU, write results or establish benchmark quality. The 2-by-2 red PPM image and
question are synthetic fixtures created for this example, not benchmark samples.

For actual searches, prepare independent datasets/calibration and use the
[setup guide](../docs/getting-started.md). Do not repeat this single example to
fabricate Search128, Dev256 or LMMS evidence.
