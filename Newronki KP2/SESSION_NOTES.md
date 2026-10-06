# KP2 – session notes (read this to resume)

Last updated: 2026-10-07, after the first full working session.

## The task

- Course "Нейронні мережі", КП №2 «Згорткові нейронні мережі». Student: **Балабан Ілля Олегович, ІК-61мп**.
- **Variant 2:** dataset **Rock_paper_scissors**; research topic **«Вплив кількості та порядку шарів згортки та пулінгу в базовій CNN»**.
- Task points: (1) build a CNN, train, test, judge under-/overfitting; (2) research for the variant;
  (3) solve the same task with a pretrained network (ImageNet) fine-tuned on our data, mind the input size;
  (4) report from the teacher's template (3–5 pages, Ukrainian).
- The assignment also lists 9 review questions (CNN class, applications, pooling/subsampling, convolution,
  normalisation, how neurons implement conv/pool, how a CNN trains, MLP vs CNN, dropout). They are not part of
  the report template – usually answered at the defence. Not prepared yet.

## Status

- [x] Data downloaded, checked (8 step scripts in `Data/`), split
- [x] Point 1 – base CNN
- [x] Point 2 – research, 7 variants
- [x] Point 3 – MobileNetV2 (frozen head, then fine-tuning)
- [x] Report – every template section filled, pushed to GitHub
- [ ] **Report is 7 pages, template allows 3–5.** User has not picked cuts yet. Candidates: both code blocks
      (~0.6 page), Fig. 4 research val loss (~0.3), Fig. 5 MobileNetV2 curves (~0.3), Fig. 1 base curves (~0.3)
- [ ] Optional: Ukrainian answers to the 9 review questions, with examples from our results
- [ ] Optional: MobileNetV2 + data augmentation run (offered; user chose to report results as they are)
- Leftovers: branch `kp2` on GitHub is identical to `main` and can be deleted;
  `Newronki KP1/r16_mclaren.py` is an empty unrelated file, deliberately not committed.

## Decisions and user preferences

- The user wants to go **step by step** and asks for **simple explanations** ("like to a child"); give the
  Ukrainian term too when it will be needed in the report (згортка, пулінг/підвибірка, карта ознак,
  рецептивне поле, перенавчання …).
- **Random stratified 80/20 split** of the training set (seed 42) – chosen by the user after a split by hand
  model was offered and explained. Consequence: validation accuracy is ≈100 % for every model (val frames are
  near-copies of train frames), so the **test set is the real measure**; the report says so.
- **1 run per variant** (seed 42). Differences of ~1 % are not reliable; the report says so.
- **No global-pooling experiment** in the research (user declined). MobileNetV2's head does use
  GlobalAveragePooling2D – standard for that network.
- Variant names are descriptive snake_case: `one_block`, `base`, `five_blocks`, `pool_first`, `pool_last`,
  `double_conv`, `double_conv_pool`.
- `double_conv` is only **mentioned in the report text**, not in tables/charts (`NOT_CHARTED` in
  `Research/compare_results.py`).
- Report style: Ukrainian, decimal commas, differences written as **"%"** (not «п. п.»), formatting like the
  KP1 report (bold labels, justified text, Courier New code, centred figures, Table Grid tables).
- Charts: matplotlib default style, like KP1.
- Everything runs on **CPU**. TensorFlow on native Windows has no GPU support (the PC's AMD RX 7800 XT is
  unused); a laptop with RTX 3050 over SSH was discussed and dropped.

## Results (test set: 372 images, 124 per class)

| Model | Layers (C = Conv 3×3, P = MaxPool 2×2) | Params | Test acc | Test loss | Correct paper/rock/scissors |
|---|---|---|---|---|---|
| base | C32-P-C64-P-C128-P | 4.83 M | 89.8 % | 0.400 | 116 / 114 / 104 |
| one_block | C32-P | 22.4 M | 75.5 % | 0.971 | |
| five_blocks | C32-P-C64-P-C128-P-C128-P-C128-P | 0.45 M | 90.6 % | 0.544 | 104 / 111 / 122 |
| pool_first | P-C32-P-C64-P-C128 | 3.78 M | 83.1 % | 0.812 | |
| pool_last | C32-C64-C128-P-P-P | 5.40 M | 86.0 % | 0.853 | |
| double_conv | C32-C32-P-C64-C64-P-C128-C128-P | 3.97 M | 81.7 % | 0.799 | 70 / 117 / 117 |
| **double_conv_pool** | C32-C32-P-P-C64-C64-P-P-C128-C128-P-P | **0.30 M** | **91.7 %** | **0.288** | 93 / 124 / 124 |
| MobileNetV2, phase 1 (frozen) | ImageNet body + GAP + Dropout 0.2 + Dense 3 | 3,843 trainable | 84.9 % | 0.307 | 68 / 124 / 124 |
| MobileNetV2, phase 2 (top 30 layers) | same, lr 1e-5 | 1.53 M trainable | 81.2 % | 0.638 | 54 / 124 / 124 |

Main findings used in the report:
- Every model hits ~100 % on validation; only the test set separates them.
- **Where the weights live decides generalisation:** most weights in conv layers + small maps before
  Flatten → better (double_conv_pool, five_blocks); a huge Dense layer after Flatten → memorises the 21
  training hands (one_block, double_conv).
- Pooling must come *after* convolutions (pool_first loses detail despite a larger receptive field).
- Hardest test images: "paper" hands with fingers held together (training paper hands spread them) –
  MobileNetV2 calls them "rock"; fine-tuning increased overfitting.
- Our 0.3 M-parameter CNN from scratch beat the pretrained MobileNetV2.

## Layout

```
Newronki KP2/
  Data/            step1…step8 data-check / split / load scripts (+ dataset, gitignored)
  BaseModel/       base_cnn.py                 – trains the base CNN
  FinalEvaluation/ base_cnn_eval.py, base_cnn_error_examples.py, mobilenet_v2_paper_errors.py
  Research/        architectures.py (VARIANTS), train_variant.py, compare_results.py
  TransferModel/   mobilenet_v2.py             – phase 1 + phase 2, tests after each
  Results/         base_cnn_*, research/, transfer/  (xlsx, png; model files gitignored)
  Report/          build_report.py, KP2_report_template.docx, the report .docx
```

## Resuming on a fresh clone

All results (xlsx/png) and the report are committed; the dataset and trained models are **not** (too big).

1. Environment (Python 3.12 was used):
   `python -m venv venv` → activate → `pip install -r "Newronki KP2/requirements.txt"`.
   On Windows set `PYTHONIOENCODING=utf-8` when running scripts that print Ukrainian/box characters.
2. Rebuilding the report needs **no retraining**: `python "Newronki KP2/Report/build_report.py"`.
   It regenerates the .docx from the template and the PNGs in `Results/`, after saving a `.backup.docx`.
   Close the report in Word first. Manual Word edits are overwritten – put changes into the script.
3. Anything that needs the dataset (re-training, figure scripts that load a model):
   - download `rps.zip` and `rps-test-set.zip` from https://storage.googleapis.com/download.tensorflow.org/data/
     and unzip into `Newronki KP2/Data/` (gives `rps/` and `rps-test-set/`);
   - run `Data/step7_random_split.py` – seed 42 reproduces the exact same train/val split.
4. Re-training (CPU times on a Ryzen 5 5600): `BaseModel/base_cnn.py` ≈ 3 min;
   `Research/train_variant.py all` ≈ 40 min (pool_last alone ≈ 12 min; single variants by name also work);
   `TransferModel/mobilenet_v2.py` ≈ 8 min (+ 9 MB ImageNet weights download).
   Training is deterministic with seed 42 (base was reproduced exactly).
5. Git: close the .docx in Word before switching branches or merging – a Word file lock broke a branch
   switch once.
