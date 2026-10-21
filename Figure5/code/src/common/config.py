"""Configuration loading. One YAML file, one place where a path is decided.

Usage
-----
    from common.config import load_config
    cfg = load_config()                  # config/config.yaml
    cfg = load_config("config/other.yaml")

Every path in the returned object is absolute. Paths under `inputs:` are
resolved against `project_root`; a path that is already absolute is returned
unchanged, so a site-specific override can be dropped in without editing the
shipped config.

`project_root` itself defaults to the directory that contains both this
repository and the `data/` tree, i.e. the parent of `src/../..`. Override it
with the `SCATAC_PROJECT_ROOT` environment variable.

This module deliberately performs no I/O other than reading the YAML: it does
not create directories and does not check that inputs exist. `require()` is
where an existence check belongs, so that a script fails with the name of the
missing input rather than a bare `FileNotFoundError`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# config/ lives directly under the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

ENV_PROJECT_ROOT = "SCATAC_PROJECT_ROOT"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing a key the caller needs."""


class Config:
    """A thin, explicit view over the config YAML.

    Attribute-style access is intentionally not provided: `cfg["inputs"]["h5ad"]`
    is greppable, and a typo raises KeyError naming the key rather than silently
    returning None.
    """

    def __init__(self, raw: dict[str, Any], project_root: Path, source: Path) -> None:
        self._raw = raw
        self.project_root = project_root
        self.source = source

    # -- raw access ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            return self._raw[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ConfigError(f"{self.source}: missing top-level key {key!r}") from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    @property
    def analysis(self) -> dict[str, Any]:
        return self["analysis"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self["runtime"]

    def param(self, name: str) -> Any:
        """A value from the `analysis:` block, with a clear error if absent."""
        try:
            return self["analysis"][name]
        except KeyError as exc:
            raise ConfigError(
                f"{self.source}: analysis.{name} is not set. Every number a "
                f"figure depends on belongs in the config; add it there rather "
                f"than defaulting it in code."
            ) from exc

    # -- paths --------------------------------------------------------------
    def input(self, name: str) -> Path:
        """Resolve `inputs.<name>` against the DATA root (existence not checked)."""
        try:
            value = self["inputs"][name]
        except KeyError as exc:
            raise ConfigError(f"{self.source}: inputs.{name} is not set") from exc
        return self._resolve(value, base=self.project_root)

    def output_root(self) -> Path:
        """Where results go: inside THIS repository, never beside it.

        Inputs live in a data tree that may sit anywhere; outputs belong to the
        repository so that cloning it and running it produces the results in the
        clone. Resolving outputs against `project_root` instead would scatter
        them into the data tree, where a later run of a different analysis would
        find and overwrite them.
        """
        return self._resolve(self["outputs"]["results_root"], base=REPO_ROOT)

    def logs_root(self) -> Path:
        return self._resolve(self["outputs"]["logs_root"], base=REPO_ROOT)

    def result_dir(self, analysis: str) -> Path:
        """`results/<analysis>/`; created on demand by `ensure_dir`."""
        return self.output_root() / analysis

    @staticmethod
    def _resolve(value: Any, base: Path) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else (base / path)

    # -- checked paths ------------------------------------------------------
    @staticmethod
    def require(path: Path, what: str) -> Path:
        """Fail loudly, naming the missing input, instead of later and vaguely."""
        if not path.exists():
            raise ConfigError(
                f"missing {what}: {path}\n"
                f"Check the matching entry in config/config.yaml, or set "
                f"{ENV_PROJECT_ROOT} if the data tree lives somewhere else."
            )
        return path

    def require_input(self, name: str) -> Path:
        return self.require(self.input(name), f"input '{name}'")


def _default_project_root() -> Path:
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    # REPO_ROOT/../ = the scATAC_science_result/ directory holding data/.
    return (REPO_ROOT.parents[2] / "analysis_data" / "Figure5").resolve()


def load_config(path: str | Path | None = None) -> Config:
    source = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    if not source.exists():
        raise ConfigError(f"config file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")

    project_root = raw.get("project_root")
    if project_root:
        project_root = Path(str(project_root)).expanduser().resolve()
    else:
        project_root = _default_project_root()

    if not project_root.is_dir():
        raise ConfigError(
            f"project_root is not a directory: {project_root}\n"
            f"Set it explicitly in {source} or via {ENV_PROJECT_ROOT}."
        )
    return Config(raw, project_root, source)


def ensure_dir(path: Path) -> Path:
    """mkdir -p, returning the path so it can be used inline."""
    path.mkdir(parents=True, exist_ok=True)
    return path


