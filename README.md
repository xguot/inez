# Inezita

Stealing the quiet hours of my twenties.

Inezita (`inezita`) is the current ~29.2M model. The future larger model will be
called Inze (`inze`).

The 0.1.3 development branch adds full-corpus replay, exact exported-weight
validation and paired promotion gates. See the [0.1.3 training and evaluation
guide](docs/inezita-0.1.3.md). The 2200 reference is a development target; measured
strength remains documented in the [frozen 0.1.2 baseline](docs/evaluation/inezita-0.1.2/README.md).

## Quickstart

Build and test the C++20 rules and PUCT engine:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Play against the trained neural engine (`build/inezita`) in a UCI client, or evaluate it directly:

```sh
./build/inezita
.venv-model/bin/python tools/evaluate_neural.py --output results/diagnostics.json
```

See [`docs/`](docs/) for rules, search, training, and neural UCI documentation.
The [benchmark protocol](docs/benchmarking.md) defines paired engine matches,
game counts, model-size comparisons, and Slurm execution. On Rivanna, start with
`sbatch hpc/benchmark_inezita.slurm` for the 20-game pilot.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
