# Independent M2 training

Run from this repository in a dedicated Python 3.11+ environment:

```sh
python -m pip install -e ".[test]"
python -m m2_training --help
python -m m2_training train --checkpoint runs/selective.pt --experiment selective_state --steps 100 --save-every 10 --device cuda
python -m m2_training resume --checkpoint runs/selective.pt --steps 200 --device cuda
python -m m2_training evaluate --checkpoint runs/selective.pt --device cuda
```

Use `--device cpu` without a CUDA GPU. `--steps` is the total target, not
additional steps. Training refuses to overwrite an existing checkpoint.
Resume/evaluate reconstruct the original recipe; task, seed and experiment
cannot be overridden. Writes use a unique temporary file and atomic replacement.
An interrupted process loses at most the work since the last saved interval.

The checkpoint stores model parameters/buffers, AdamW state, completed step,
recipe, training-data RNG, torch CPU RNG and CUDA RNG when applicable.
It is loaded using `weights_only=True`. Resume requires the same device type.
Bit-exact reproducibility is tested on CPU; GPU portability across hardware,
PyTorch versions or nondeterministic kernels is not guaranteed.

The default is a small FP32 synthetic-reasoning trainer (104-wide, two layers,
batch 8, constant learning rate, answer-only supervision). Tasks are
`pointer_chase`, `mod_chain` and `text`. Text training uses all next-token
targets, an independently held-out corpus and validation PPL. Synthetic
evaluation reports accuracy. `--grad-accum` controls microbatches per optimizer
step; checkpoints are written only at completed optimizer steps.

## Real text and scale configuration

Prepare separate UTF-8 files, then train:

```sh
python -m m2_training.prepare --train train.txt --validation validation.txt --output data/corpus
python -m m2_training train --task text --corpus data/corpus/manifest.json --checkpoint runs/text.pt --sequence-length 128 --batch 2 --grad-accum 4 --steps 100 --device cuda
python -m m2_training resume --checkpoint runs/text.pt --steps 200 --device cuda
python -m m2_training evaluate --checkpoint runs/text.pt --device cuda
python -m m2_training.inspect --size 2b --vocab-size 32768
```

Preparation uses UTF-8 **byte tokens** (vocabulary 256), a dependency-free
pipeline check, not a recommended production tokenizer. A custom pretokenized
corpus uses the same manifest schema: `format_version=1`, a nonempty `tokenizer`
identity, `vocab_size`, and `train`/`validation` objects containing a relative
`.npy` `file` path and SHA-256 `sha256`. Arrays must be one-dimensional integer
token IDs in vocabulary range. Record the tokenizer revision in its identity.
Corpus checksums and manifest identity are verified on resume. Keep corpus
paths unchanged when resuming; corpus relocation is not implemented.

`--size 2b` configures width 2080, 34 layers, 16 query heads and 4 KV heads.
At vocabulary 32768 its baseline has **2,064,984,584 parameters**, counted on
the meta device without allocating weights. Parameter count changes with
vocabulary and experiment. FP32 Adam training-state lower bound is
33,039,753,344 bytes, **excluding activations, temporary allocations and
checkpoint loading peaks**. This preset has not been trained at full scale.
Use a suitably provisioned machine; the local 8GB GPU is only for probe runs.

This is a single-device FP32 training path. Scheduler, AMP, activation
checkpointing, distributed training and production instruction tuning remain
outside its implemented scope. Do not describe the preset as trained 2B weights.

## Module wiring

| Module | Status in this trainer |
|---|---|
| baseline | Complete train/save/resume/evaluate path |
| selective_state | Complete path; input-dependent liquid transition |
| latent_core / latent_stack | Complete path; fixed four iterations |
| workspace | Complete path; GWT parent enabled, four workspace iterations |
| top_down | Not exposed: model accepts the signal, trainer has no goal source |
| world_model / predictive_coding / Hamiltonian head | Imported implementations and some direct tests; task-specific objectives and resumable auxiliary state not audited here |
| fast-weight core / Hebbian / rhythm / competitive workspace / coherence | Imported; not exposed until checkpoint-state and task wiring are audited |
| astrocyte / neuromodulation / sleep | Research components; no closed training loop in this entry point |

CLI rejects unsupported experiment names. Availability of a Python profile
does not establish that its mechanism is useful or fully integrated.

## Source boundary

`mt_lnn/research/` now owns ten physically extracted research implementations:
world model, Hamiltonian head, rhythm, predictive coding, Hebbian plasticity,
astrocyte, neuromodulation, sleep consolidation, coherence and GWT workspace.
Their old `mt_lnn.<module>` paths resolve to the same module objects for
compatibility. Main-model integration and core/stack loop assembly remain in
`mt_lnn/model.py`; not every experimental code path has been extracted.
`m2_training/` owns the M2 recipe, runner and CLI;
`benchmarks/reasoning_tasks.py` supplies synthetic data. Packaging includes
these modules, so installed usage does not reach into an M1 checkout.
No corresponding source removal was performed in the separate M1 repository.

Validation: five configurations pass exact CPU continuous-vs-resumed parameter
comparisons. CUDA selective-state training, cross-process resume and evaluation
were manually exercised. Short-run accuracies are smoke evidence only.
