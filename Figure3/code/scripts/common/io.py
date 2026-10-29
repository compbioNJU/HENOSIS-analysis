"""Read-only input configuration and strict figure-data CSV output."""

from pathlib import Path
import os
from typing import Sequence

import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    root = Path(os.environ.get('ANALYSIS_DATA_DIR', path.resolve().parents[4] / 'analysis_data')).resolve()
    return {key: root / value for key, value in config.items()}


def check_inputs(config: dict) -> None:
    missing = [f'{key}: {path}' for key, path in config.items() if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing configured inputs:\n' + '\n'.join(missing))


def write_plot_csv(path: Path, frame: pd.DataFrame,
                   required_columns: Sequence[str]) -> None:
    missing = set(required_columns) - set(frame)
    if missing:
        raise ValueError(f'Plot data missing columns: {sorted(missing)}')
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def results_root(code_root: Path) -> Path:
    """Use the shared output base when set; preserve direct-script defaults."""
    override = os.environ.get('ANALYSIS_RESULTS_DIR')
    return Path(override).expanduser().resolve() / 'Figure3' if override else code_root / 'results'
