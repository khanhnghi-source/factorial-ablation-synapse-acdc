# A controlled factorial ablation of CBAM, BiLSTM and deep supervision

Code, dataset partitions, per-case results and analysis scripts accompanying:

> *Dataset-Dependent Gains at 3.6x the Parameter Cost: A Controlled Factorial
> Ablation of CBAM, BiLSTM, and Deep Supervision in Hybrid CNN-Transformer
> Segmentation of Cardiac MRI and Multi-Organ CT.*

<!-- No status line ("submitted", "under review") belongs here: it goes stale the
     moment the status changes, and the status will change at least once more.
     When the paper is accepted, replace the block above with the full citation
     and its DOI -- once, and correct from then on. -->

## A note on the name `SeqAtt_UNet`

The Python package is called `SeqAtt_UNet` because that was the project's working
name while the 70 runs were trained and scored. The name is embedded in the run
directories and in every line of the evaluation logs under `results/test_log/`,
so renaming the package would break the correspondence between the code released
here and the records the paper's tables are computed from. It is kept for that
reason and no other: the paper does not propose a method under that name, and it
recommends against the configuration the name once referred to.

## What this study found

A 2x2 factorial design over a CBAM attention module and a BiLSTM sequential
sub-block, on a deep-supervision base and against a no-deep-supervision control,
trained for five seeds per cell on two benchmarks -- **70 runs in total**.

- The **sequential block** is the only component with a measurable benefit, and
  only on ACDC (+0.21 pp mean Dice, one-sided Wilcoxon *p* = 0.033). It costs
  269 M parameters -- 3.56x the baseline -- and 7.9x batch-1 latency.
- **CBAM** produces no measurable Dice gain in any of the four factorial cells on
  either dataset.
- **Deep supervision** adds 0.03% parameters and no inference cost, yet produces
  the largest single effect measured: Synapse HD95 falls from 23.81 to 20.50
  voxels (*p* = 0.021, *d* = -0.54).
- No endpoint survives Holm correction across the five pre-specified endpoints.

The controlled design also corrects an attribution that an uncontrolled comparison
invites: deep supervision contributes **0.31** Dice points on Synapse, not the
**3.52** obtained by comparing against a published TransUNet value. 91% of that
gap reflects pipeline differences rather than the component.

**The configuration this study set out to evaluate is not the one it recommends.**
Within this design space the most favourable accuracy-cost operating point is the
105.31 M-parameter deep-supervision baseline itself.

## Repository layout

```
SeqAtt_UNet/          model, training and evaluation code
  networks/           CBAM backbone, ViT configs, the Phase-3 model, Skip-CBAM
  datasets/           Synapse and ACDC loaders (loaders only -- no image data)
  lists/              the exact dataset partitions used, as text
  aggregate_results.py        parses test logs into the results workbook
  statistical_tests*.py       Wilcoxon endpoints, Holm correction, bootstrap CIs
  compute_flops.py            parameters, GFLOPs, throughput, peak VRAM
  verify_run_inventory.py     checks that every configuration has five seeds
  verify_alpha_fingerprint.py checks each log against its checkpoint (see below)
  audit_log_staleness.ps1     checks for test logs older than their checkpoint
results/
  KetQua_v2.xlsx      per-run, per-case and per-class results for all 70 runs
  test_log/           the raw evaluation logs those results are parsed from
```

The logs under `results/test_log/` are byte-for-byte as the evaluation runs wrote
them. They were produced by authors whose working language is Vietnamese, and a
handful of status lines are in that language -- for example
`Mac dinh ACDC: (1.52, 1.52)`, recording the default in-plane spacing applied
when HD95 is also expressed in millimetres (Section IV-D of the paper). We have
not translated or otherwise tidied these files. They are the primary record the
tables are computed from, and an unedited log is worth more to a reader checking
our numbers than a cleaned one. Every value the paper reports is parsed from
them by `aggregate_results.py`, whose output is `KetQua_v2.xlsx`.

## Reproducing the tables

```bash
python SeqAtt_UNet/aggregate_results.py --test_log_dir results/test_log \
                                        --output results/KetQua_v2.xlsx
python SeqAtt_UNet/statistical_tests_primary.py
python SeqAtt_UNet/compute_flops.py --num_classes 9 --n_warmup 100 --n_iters 500
```

Throughput depends on the runtime, not only the hardware. The figures in Table V
were measured on an NVIDIA GTX 1080 Ti under PyTorch 2.6.0 with CUDA 12.4.
Repeated sessions varied by roughly +-2%, so differences below that margin in the
throughput column should not be interpreted. Parameters, GFLOPs and peak memory
were identical across four sessions and two PyTorch versions.

## Two integrity checks, and why they are here

While building the factorial controls we found that one stored checkpoint had been
overwritten by a later training run without its evaluation being re-run, so a
reported score belonged to a model that no longer existed. File modification times
could not settle this reliably, because the results were held on a cloud-synced
drive that rewrites timestamps.

`verify_alpha_fingerprint.py` compares the CBAM gate values recorded in each test
log against those in the checkpoint on disk. A mismatch proves the checkpoint
changed after the log was written, and no file-system metadata can forge it. It
covers the 40 runs that use CBAM; `audit_log_staleness.ps1` gives weaker but
universal coverage by comparing modification times.

Both are released so that the audit reported in the paper can be repeated rather
than taken on trust.

## Data

The imaging data is **not** redistributed here. See `DATA.md` for how to obtain
Synapse and ACDC, and for what this repository does contain: the exact partition
lists and the per-case metrics derived from our runs.

## Trained checkpoints

Checkpoints for all 70 runs are archived at
[doi:10.5281/zenodo.22833561](https://doi.org/10.5281/zenodo.22833561), under the
same Apache-2.0 licence. Fourteen tar archives, one per dataset and
configuration, each holding that cell's five seeds, plus `MANIFEST_SHA256.txt`.

Each archive holds the `best_model.pth` of its runs -- the file `test_phase3.py`
loads. The intermediate `epoch_99` and `epoch_149` snapshots are not released,
because no reported result is computed from them. The manifest gives a SHA-256
digest and byte size for every file, so a downloader can verify integrity without
trusting us.

The DOI above is the Zenodo *concept* DOI: it always resolves to the latest
version of that record.

## Licence

Apache License 2.0 (`LICENSE`), with attribution in `NOTICE`.

This repository is a derivative work of
[TransUNet](https://github.com/Beckschen/TransUNet) (Chen et al., 2021), which
is Apache-2.0. Sixteen files contain code from that project and each carries a
notice at the top stating that it was modified, as Section 4(b) requires. The
remaining files were written for this study and are offered under the same
licence so that the repository has one set of terms throughout.

Pretrained weights and imaging data are not covered by this licence and are not
redistributed here; see `THIRD_PARTY.md` and `DATA.md`.

## Citation

```bibtex
[BIBTEX ENTRY -- fill in after acceptance]
```