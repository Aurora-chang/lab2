# Dataset (local only)

Place the four assignment-provided files here:

- `S4b_train.npz`
- `X11b_train.npz`
- `S4b_test.npz`
- `X11b_test.npz`

Dataset files are excluded from Git. Each file contains `signal` with shape `(540, 750, 2)` and labels 1/2. The loader converts labels to 0/1 and signals to `(N, 1, 2, 750)`. See [reproduction instructions](../../docs/REPRODUCIBILITY.md).
