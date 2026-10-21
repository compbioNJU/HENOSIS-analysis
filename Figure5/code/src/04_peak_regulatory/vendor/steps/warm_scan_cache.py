"""Pre-compute the reference sequences and FIMO motif scan for every species.

This is pure cache warming: it computes exactly what `s01` would compute, so
that the later steps start from a warm cache. It changes no result.

    python script/warm_scan_cache.py --workers 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def one_species(config: dict, species: str, workers: int, force: bool) -> str:
    genomes = C.species_table(config)
    genome = genomes[species]

    manifest, rankings = C.load_rankings(config)
    manifest = manifest[~manifest.is_empty]
    peaks = C.model_species_peaks(manifest, rankings)[species]
    C.log(species, "universe", len(peaks), "peaks")

    sequences = C.read_sequences(config, genome, peaks, use_cache=not force)
    n_valid = int(sequences.valid.sum())
    C.log(species, "sequences", n_valid, "/", len(sequences), "valid")
    if n_valid / max(len(sequences), 1) < 0.99:
        C.log(species, "WARNING low sequence recovery")

    hits = C.scan_motifs(config, genome, sequences, workers=workers, force=force)
    C.log(species, "motifs with at least one hit:", len(hits))
    return species


def main() -> None:
    parser = argparse.ArgumentParser()
    C.add_common_args(parser)
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    config = C.load_config(args.config)
    species_list = args.species or C.SPECIES_ORDER
    for species in species_list:
        with C.StepTimer(f"warm {species}"):
            one_species(config, species, args.workers, args.force)


if __name__ == "__main__":
    main()
