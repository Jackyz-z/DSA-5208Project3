# Parallel Decision Tree Training with MPI + OpenMP + CUDA

This project trains a CART classification tree (Gini impurity, histogram-based
split finding) in C++, with a serial reference implementation and a parallel
implementation that combines:

* **MPI** across ranks, using two different partitioning strategies:
  * **data-parallel** (rows split across ranks), with two collective patterns:
    `Allreduce` or `Reduce_scatter` + `Allgather`;
  * **feature-parallel** (features split across ranks), with an
    `Allgather` of candidate splits and a bitwise-OR `Allreduce` of the row
    partition bitmap;
* **OpenMP** within a rank, with two work-sharing strategies for histogram
  construction (`feature`: threads own features; `sample`: threads own rows and
  private histograms);
* **CUDA** within a rank (optional), building histograms on the GPU with a
  shared-memory privatised kernel or a global-atomic kernel.

All parallel configurations produce trees that are **byte-identical** to the
serial reference. Histogram counts are integers, so the reductions are exact
regardless of their order, and ties between splits are broken by a fixed
total order.

## Algorithm in brief

1. **Binning.** Each feature is discretised into at most `B` (≤ 256) quantile
   bins, estimated from a sample of rows (every `stride`-th global row, so the
   sample does not depend on the number of ranks). A row is stored as one
   `uint8` bin index per feature.
2. **Level-wise growth.** All nodes of the current depth (the *frontier*) are
   processed together. For each frontier node we build a histogram
   `hist[feature][bin][class]` of the rows it contains.
3. **Split search.** For every node and feature, scanning the histogram's
   prefix sums gives the Gini decrease of each threshold. The best split
   overall is chosen with the order *(larger gain, smaller feature, smaller bin)*.
4. **Histogram subtraction.** For a sibling pair, only the smaller child's
   histogram is built; the larger one is `parent − smaller`.
5. **Routing.** Rows move to the child node they belong to; rows in leaves are
   deactivated.

The histogram of a level holds `nodes × features × bins × classes` counters,
so its size grows with depth, while the work of building it is proportional to
the number of rows. This trade-off between computation and communication is what the
different parallel designs explore.

## Repository layout

```
include/dt/          headers
  binning.hpp        quantile bin edges, value -> bin mapping
  dataset.hpp        DTREEBIN reader (column-major binary format)
  split.hpp          Gini split search, tie-breaking order, stopping rules
  tree.hpp           tree container, prediction, canonical text output
  hist_backend.hpp   interface for histogram builders
  hist_cpu.hpp       OpenMP backend (feature / sample strategies)
  hist_cuda.hpp      CUDA backend
  trainer.hpp        MPI trainers (data-parallel, feature-parallel)
  stats.hpp, report.hpp   timing statistics and CSV output
src/
  main_serial.cpp    dt_serial: independent serial reference (recursive, no subtraction)
  main_mpi.cpp       dt_mpi: parallel driver
  trainer.cpp        level-wise MPI training loops
  hist_cpu.cpp       OpenMP histogram construction and routing
  hist_cuda.cu       CUDA kernels
  binning.cpp, dataset.cpp, split.cpp, tree.cpp
tests/test_core.cpp  unit tests (hand-computed Gini example, binning rule, tie-breaking, ...)
scripts/
  prepare_data.py    CSV (e.g. HIGGS) -> DTREEBIN train/test files
  gen_synthetic.py   synthetic datasets for weak scaling and quick tests
  validate.sh        compares every parallel configuration against dt_serial
  sklearn_baseline.py   scikit-learn DecisionTreeClassifier on the same data
  feature_importance.py Gini importance from a dumped tree
  plot_results.py    scaling, efficiency and time-breakdown figures
pbs/                 PBS job scripts for Vanda (validation, strong/weak scaling, hybrid, GPU, depth)
CMakeLists.txt, Makefile
```

## Building

Requirements: a C++17 compiler with OpenMP, an MPI library (tested with Open MPI),
CMake ≥ 3.18 (or GNU make), optionally the CUDA toolkit. Python 3 with
`numpy`, `pandas`, `matplotlib` and `scikit-learn` is needed for the scripts.

On Vanda (adjust module names to the output of `module avail`):

```bash
module load gcc openmpi cmake          # add: cuda   for the GPU backend
cmake -S . -B build                    # CPU only
cmake -S . -B build -DDT_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80   # with CUDA (80 = A100, 90 = H100)
cmake --build build -j
./build/dt_tests
```

Without CMake: `make` (CPU) or `make CUDA=1 CUDA_ARCH=sm_80`; binaries go to `build/`.

## Preparing data

**HIGGS** (UCI Machine Learning Repository, 11M rows, 28 features, binary
labels; by convention the last 500,000 rows are the test set):

```bash
mkdir -p data
wget -P data https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
python3 scripts/prepare_data.py data/HIGGS.csv.gz \
    --train data/higgs_train.bin --test data/higgs_test.bin --test-rows 500000
```

**Synthetic data** (any size; used for weak scaling):

```bash
python3 scripts/gen_synthetic.py data/synth_4M.bin --rows 4000000 --features 32
```

The binary format is documented in `include/dt/dataset.hpp` and `scripts/dtbin.py`.

## Running

```bash
# serial reference
./build/dt_serial --train data/higgs_train.bin --test data/higgs_test.bin --max-depth 10

# data-parallel, 16 ranks x 4 threads
OMP_NUM_THREADS=4 mpirun -np 16 --map-by ppr:16:node:PE=4 --bind-to core \
    ./build/dt_mpi --train data/higgs_train.bin --test data/higgs_test.bin --max-depth 10

# data-parallel with reduce-scatter, feature-parallel, GPU
mpirun -np 32 ./build/dt_mpi --train ... --comm reducescatter
mpirun -np 14 ./build/dt_mpi --train ... --mode feature
mpirun -np 4  ./build/dt_mpi --train ... --backend cuda       # one rank per GPU
```

| option | meaning | default |
|---|---|---|
| `--train FILE` / `--test FILE` | DTREEBIN datasets | – |
| `--mode data\|feature` | partition rows or features across ranks | `data` |
| `--comm allreduce\|reducescatter` | histogram collective (data mode) | `allreduce` |
| `--backend cpu\|cuda` | histogram builder (cuda: data mode) | `cpu` |
| `--omp feature\|sample` | OpenMP strategy for CPU histograms | `feature` |
| `--max-depth D` | maximum depth (root = 0) | 10 |
| `--min-samples-leaf N`, `--min-samples-split N` | stopping rules | 1, 2 |
| `--bins B` | maximum bins per feature (2–256) | 256 |
| `--sample-rows N` | rows used to estimate bin edges | 200000 |
| `--no-subtract` | disable histogram subtraction | off |
| `--dump-tree FILE` | write the canonical tree | – |
| `--csv FILE` | append a summary row (times are max over ranks) | – |
| `--level-csv FILE` | append per-level statistics of rank 0 | – |
| `--tag NAME` | label stored in the CSV files | `run` |

`dt_serial` accepts the training, depth, leaf, bin, dump, CSV and tag options.

Each run prints the tree size, a time breakdown (read, bin, train = hist +
comm + split + route, eval), the bytes communicated per rank, and the
train/test accuracy.

## Validation

```bash
./build/dt_tests                                    # unit tests
scripts/validate.sh build data/higgs_train_1M.bin data/higgs_test.bin 8          # CPU
scripts/validate.sh build data/higgs_train_1M.bin data/higgs_test.bin 8 --cuda   # + GPU
python3 scripts/sklearn_baseline.py data/higgs_train.bin data/higgs_test.bin --max-depth 10
```

`validate.sh` runs the serial reference, then each combination of mode,
collective, OpenMP strategy, rank count (1, 2, 3, N), thread count and
subtraction setting, and compares the dumped trees byte by byte. Odd rank
counts check uneven partitions, and feature mode with more ranks than features
checks ranks that own no features. The scikit-learn baseline gives an independent
accuracy reference; it uses exact thresholds, so its accuracy is close to, but
not identical to, ours.

## Experiments on Vanda

Submit from the repository root after building and preparing data:

```bash
qsub pbs/validate.pbs     # correctness (validation.txt)
qsub pbs/strong_cpu.pbs   # strong scaling, pure MPI, all MPI variants
qsub pbs/hybrid.pbs       # OpenMP scaling and MPI x OpenMP splits on one node
qsub pbs/weak.pbs         # weak scaling with synthetic data
qsub pbs/gpu.pbs          # CUDA backend, 1..N GPUs
qsub pbs/depth.pbs        # effect of tree depth and bin count
```

Edit the `#PBS` resource lines (queue, project, cores per node, GPUs) and
`pbs/env.sh` (modules, paths) to match your allocation. All jobs append to
`results/results.csv` and `results/levels.csv`.

Figures:

```bash
python3 scripts/plot_results.py results/results.csv --tag strong_cpu --out figures/strong \
    --levels results/levels.csv
python3 scripts/plot_results.py results/results.csv --tag omp_threads --x threads --out figures/omp
python3 scripts/plot_results.py results/results.csv --tag hybrid --x ranks --out figures/hybrid
python3 scripts/plot_results.py results/results.csv --tag weak --weak --out figures/weak
python3 scripts/plot_results.py results/results.csv --tag gpu --x ranks --out figures/gpu
python3 scripts/feature_importance.py results/higgs_tree_serial.txt --names higgs
```

## Output formats

* **Tree dump** (`--dump-tree`): one line per node in breadth-first order,
  e.g. `node 3 depth 2 n 1234 counts 800 434 split feature 25 bin 117 threshold 0.8731 left 7 right 8`.
* **Results CSV**: configuration, ranks, threads, phase times (max over ranks),
  MB sent per rank, tree size, and accuracies.
* **Level CSV**: per tree level, frontier size, number of built histograms,
  phase times and bytes sent by rank 0.

## Limitations

* Classification only (up to 16 classes); missing values are not supported.
* At most 2^32 rows per rank, and 256 bins per feature.
* Feature-parallel mode cannot use more useful ranks than there are features.
* The CUDA backend runs in data-parallel mode only, and the histograms are
  staged through host memory before the MPI reduction.
