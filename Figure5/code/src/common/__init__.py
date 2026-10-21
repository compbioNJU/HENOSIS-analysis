"""Shared vocabulary, I/O, estimators and plotting for this repository.

HOW A STEP SCRIPT FINDS THIS PACKAGE
------------------------------------
Step scripts live in `src/<NN>_<name>/` and are run directly
(`python src/03_developmental/run_developmental_direction.py`). Python puts the
*script's own* directory on `sys.path`, not `src/`, so `from common import ...`
would fail. Each step therefore starts with two lines:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from common import io, labels, plotting, stats

They are repeated in each script rather than hidden behind a package install,
because those two lines are then visible to anyone reading or copying one script
out of the repository -- which is the case that matters when the point of the
repository is that a reader can check the code.

MODULES
-------
    config.py     config/config.yaml, resolved against the data root
    labels.py     species, stages, ranks, cell types and colours -- defined once
    io.py         reading the atlas, auditing the input, writing results
    stats.py      sample centroids, directions, permutations, DTW, LOSO
    alignment.py  the cross-species stage correspondence built on top of stats
    plotting.py   figure style, flexible raster/vector figures, PDF text checks
"""
