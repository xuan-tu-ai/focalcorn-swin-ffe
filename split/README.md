# Split index files

The partition used for every result, published so it can be reused exactly rather
than regenerated from a description.

Each file lists one image per line as `<Species> - <Freshness>/<filename>`,
relative to the dataset root. `MANIFEST.json` records the seed, the sizes, and a
SHA-256 prefix for each file.

| File | Contents |
|---|---|
| `development.txt` | the 3,512 development images |
| `test.txt` | the 878 held-out test images |
| `foldN_train.txt`, `foldN_val.txt` | the five species-stratified development folds |
| `session_development.txt`, `session_test.txt` | the capture-session-disjoint partition |

## The two partitions

The **image-level** split (`development.txt` / `test.txt`) is drawn once with
`SPLIT_SEED = 42`, stratified over the 24 species × freshness strata, and is
independent of the training seed. It is the partition comparable with prior work
on this benchmark.

The **session-disjoint** split holds out 24 of the 72 photography sessions, so no
session contributes images to both sides. It is harder, because the benchmark was
built by photographing stored fish repeatedly over six days and images from one
session share background, lighting, and staging. Results on the two partitions
differ substantially; both are reported.

Regenerate either with:

```bash
python src/export_split.py --out split
```
