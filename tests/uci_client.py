# SPDX-License-Identifier: GPL-3.0-only
"""Raw UCI test client for assertions about protocol ordering and cancellation."""

from collections import deque
import queue
import subprocess
import threading
import time


class UciClient:
    def __init__(self, command, options=None, *, env=None):
        self.options = options or {}
        self.process = subprocess.Popen(command, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lines = queue.Queue()
        self.diagnostics = deque(maxlen=100)
        self.readers = [threading.Thread(target=self._stdout, daemon=True),
                        threading.Thread(target=self._stderr, daemon=True)]
        for reader in self.readers:
            reader.start()

    def _stdout(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _stderr(self):
        for line in self.process.stderr:
            self.diagnostics.append(line.strip())

    def send(self, command):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def read_until(self, prefix, timeout):
        deadline, prior = time.monotonic() + timeout, []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, prior, 'timeout'
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                return None, prior, 'timeout'
            if line is None:
                return None, prior, 'crash'
            if line.startswith(prefix):
                return line, prior, 'ok'
            prior.append(line)

    def initialize(self, timeout=5):
        self.send('uci')
        self._expect('uciok', timeout)
        for name, value in self.options.items():
            self.send(f'setoption name {name} value {value}')
        self.send('isready')
        self._expect('readyok', timeout)

    def _expect(self, prefix, timeout):
        _, _, status = self.read_until(prefix, timeout)
        if status != 'ok':
            raise RuntimeError(f'UCI {prefix}: {status}; ' + '\n'.join(self.diagnostics))

    def close(self):
        if self.process.poll() is None:
            try:
                self.send('quit')
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        for reader in self.readers:
            reader.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
