# Inez

Stealing the quiet hours of my twenties.

## Development

See [TODO](TODO) for the development roadmap and next implementation milestone.

The first implementation is a C++20 standard-chess rules library with legal
move generation, FEN import/export, reversible moves, and draw-history handling.
See [the rules guide](docs/rules.md) for API semantics and known limitations.

### Build and test

Requires CMake 3.20 or newer and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For the independent python-chess oracle and debug sanitizers:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DINEZ_SANITIZERS=ON -DINEZ_DIFFERENTIAL_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
```

Python is only needed for the differential tests. The engine has no runtime
dependencies beyond the C++ standard library.

### Inspect a position

Run `build/inez-rules` and enter one command per line:

```text
start
push e2e4
state
pop
perft 4
quit
```

The tool returns JSON snapshots with legal moves, FEN, and draw information.
Use `fen <six-field FEN>` to inspect another position.

## License

Inez is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for details.
