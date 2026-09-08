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

This entry point is a small FP32 synthetic-reasoning trainer, not a 2B language
pretraining pipeline. It fixes a 104-wide, two-layer model, batch 8, constant
learning rate, and answer-only supervision. Tasks are `pointer_chase` and
`mod_chain`; evaluation regenerates fixed held-out synthetic batches.
It does not implement dataset-file training, a scheduler, AMP, distributed
training or production-model instruction tuning.

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

`mt_lnn/` remains the imported compatibility runtime, including shared core
and embedded experiments. `m2_training/` owns the new M2 recipe, runner and CLI;
`benchmarks/reasoning_tasks.py` supplies synthetic data. Packaging includes
these modules, so installed usage does not reach into an M1 checkout.
Physical extraction of embedded experimental layers remains a separate task.

Validation: five configurations pass exact CPU continuous-vs-resumed parameter
comparisons. CUDA selective-state training, cross-process resume and evaluation
were manually exercised. Short-run accuracies are smoke evidence only.
