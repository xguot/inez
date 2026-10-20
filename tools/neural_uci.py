#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Launch the trained Python neural player as a UCI engine."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))

from inez.uci import main


if __name__ == '__main__':
    raise SystemExit(main())
