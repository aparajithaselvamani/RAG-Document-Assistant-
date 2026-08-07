from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    print("Creating virtual environment...")
    run([sys.executable, "-m", "venv", ".venv"])
    python_executable = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_executable.exists():
        python_executable = ROOT / ".venv" / "bin" / "python"
    print("Installing dependencies...")
    run([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_executable), "-m", "pip", "install", "-r", "requirements.txt", "pytest"])
    print("Building vector database...")
    run([str(python_executable), "ingest.py"])
    print("Setup complete. Run '.venv\\Scripts\\activate' and then 'python app.py'.")
