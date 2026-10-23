# Inezita

Stealing the quiet hours of my twenties.

Inezita (`inezita`) is the current ~29.2M model. The future larger model will be
called Inze (`inze`).

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

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
