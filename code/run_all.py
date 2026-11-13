"""Convenience launcher. Run from the package root."""
import subprocess, sys
commands = [
    [sys.executable, 'code/run_cicids2017_analysis.py'],
    [sys.executable, 'code/run_unsw_nb15_validation.py'],
]
for cmd in commands:
    print('Running:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
