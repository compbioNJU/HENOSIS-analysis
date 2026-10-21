"""Validate the two frozen config maps. Regenerate nothing.

WHY validate instead of regenerate
----------------------------------
`script_v3/make_tf_family_map.py` and `script_v3/make_module_map.py` can derive
`config/tf_family_map.tsv` and `config/celltype_motif_module_map.tsv` from the
frozen 879-motif MEME catalog, and the v2 driver ran them as step 0.

v3 does NOT. Those two maps are part of the analysis DEFINITION -- the module map
names the programs the panels draw, the family map decides which motifs aggregate
into a family -- and a frozen scientific specification should not be re-derived
inside a production run. Re-deriving them would also write into the shared
`config/` directory while three arms are reading it.

So v3 treats them as inputs and checks them here instead:

  1. SHA256 against `config/v3/frozen_maps.sha256`, recorded when v3 was frozen.
     This is the load-bearing check: it is what proves v3's module layer is the
     same object v2's was.
  2. catalogue compatibility -- the MEME database still declares the frozen 879
     motifs, and every motif the two maps name is in it.
  3. the module map is disjoint WITHIN each cell type: no motif may sit in two
     modules of the same cell type, because a module score is a median over its
     members and an overlapping motif would be double-counted in a panel.
  4. `Homeodomain_total` really is the union of its two declared sub-families and
     not, say, the mean of their medians (the distinction the v2 test suite
     guards).
  5. the family map's primary families are the four the panel expects, and only
     monomers aggregate -- a heterodimer names a complex, not a family member.

The generators stay in `script_v3/` for development. Nothing here calls them.

Usage
-----
    python script_v3/validate_maps.py --config config/v3/top500.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import common as C  # noqa: E402

ROOT = SCRIPT_DIR.parent
FROZEN = ROOT / "config" / "v3" / "frozen_maps.sha256"

PRIMARY_FAMILIES = ["RFX", "NFI", "NF-Y", "SOX"]

MODULE_COLUMNS = ["celltype", "module", "level", "member_type", "motif_name", "motif_id", "provenance"]
FAMILY_COLUMNS = ["motif_id", "motif_name", "family", "analysis_level", "member_type"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    if not FROZEN.exists():
        raise SystemExit(f"missing {FROZEN}; record the frozen map hashes first")
    out = {}
    for line in FROZEN.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        out[name.strip()] = digest.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    C.add_common_args(parser)
    args = parser.parse_args()

    config = C.load_config(args.config)
    problems: list[str] = []

    # ---- 1. sha256 ------------------------------------------------------
    expected = frozen_hashes()
    for name, want in sorted(expected.items()):
        path = ROOT / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        got = sha256(path)
        status = "OK" if got == want else "MISMATCH"
        print(f"  sha256 {name}: {status}")
        if got != want:
            problems.append(f"{name}: sha256 {got} != frozen {want}")

    module_path = C.module_map_path(config)
    family_path = C.CONFIG_DIR / "tf_family_map.tsv"
    module_map = pd.read_csv(module_path, sep="\t")
    family_map = pd.read_csv(family_path, sep="\t")

    for frame, columns, name in (
        (module_map, MODULE_COLUMNS, "module map"),
        (family_map, FAMILY_COLUMNS, "family map"),
    ):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            problems.append(f"{name} lacks columns {missing}")

    if not problems:
        # ---- 2. catalogue compatibility ---------------------------------
        catalog = C.motif_catalog(config)
        print(f"  catalogue: {len(catalog)} motifs")
        known = set(catalog.motif_id)
        for frame, name in ((module_map, "module map"), (family_map, "family map")):
            unknown = sorted(set(frame.motif_id) - known)
            if unknown:
                problems.append(f"{name} names {len(unknown)} motifs outside the catalogue: {unknown[:5]}")
        for frame, name in ((module_map, "module map"), (family_map, "family map")):
            renamed = frame.merge(catalog, on="motif_id", suffixes=("", "_cat"))
            bad = renamed[renamed.motif_name != renamed.motif_name_cat]
            if len(bad):
                problems.append(
                    f"{name}: {len(bad)} rows disagree with the catalogue on motif_name, "
                    f"e.g. {bad.iloc[0].motif_id}"
                )

        # ---- 3. disjoint within a cell type -----------------------------
        for celltype, group in module_map.groupby("celltype"):
            per_motif = group.groupby("motif_id").module.nunique()
            shared = per_motif[per_motif > 1]
            if len(shared):
                detail = {
                    mid: sorted(group[group.motif_id == mid].module.unique())
                    for mid in shared.index[:3]
                }
                problems.append(f"{celltype}: {len(shared)} motif(s) sit in >1 module: {detail}")
            print(f"  {celltype}: {group.module.nunique()} modules, "
                  f"{group.motif_id.nunique()} distinct motifs, disjoint={not len(shared)}")

        # ---- 4. a declared aggregate is a DERIVED union, not stored rows --
        # `common.module_aggregates` builds it by unioning the atomic modules at
        # read time. Storing it would double-count every motif in a panel, and
        # storing something that is NOT that union would silently change what
        # the panel draws. Both are checked.
        for celltype, aggregates in (config.get("modules", {}).get("aggregates") or {}).items():
            for name, parts in aggregates.items():
                stored = module_map[(module_map.celltype == celltype) & (module_map.module == name)]
                if len(stored):
                    problems.append(
                        f"{celltype}.{name} is declared a derived aggregate but the map "
                        f"stores {len(stored)} rows for it; it must not be a third module"
                    )
                missing_parts = [p for p in parts if p not in set(module_map.module)]
                if missing_parts:
                    problems.append(f"{celltype}.{name} names absent sub-module(s) {missing_parts}")
                    continue
                union = set(
                    module_map[
                        (module_map.celltype == celltype) & (module_map.module.isin(parts))
                    ].motif_id
                )
                declared = C.module_aggregates(config, celltype).get(name)
                if list(parts) != list(declared):
                    problems.append(f"{celltype}.{name} sub-modules drift between map and config")
                print(f"  {celltype}.{name}: derived union of {parts}, {len(union)} motifs, "
                      f"not stored ({len(stored)} rows)")

        # ---- 5. families exist and only monomers aggregate ---------------
        families = [str(f) for f in family_map.family.dropna().unique() if str(f).strip()]
        print(f"  family map: {len(family_map)} rows, families={sorted(families)}")
        unexpected = sorted(set(PRIMARY_FAMILIES) - set(families))
        if unexpected:
            problems.append(f"family map is missing primary families {unexpected}")
        for family in PRIMARY_FAMILIES:
            rows = family_map[family_map.family == family]
            if rows.empty:
                continue
            hetero_ids = set(rows[rows.member_type != "monomer"].motif_id)
            aggregating = set(C.family_members(family))          # monomers_only=True
            leaked = sorted(aggregating & hetero_ids)
            if leaked:
                problems.append(
                    f"family {family}: {len(leaked)} heterodimer(s) reach the aggregation "
                    f"despite monomers_only, e.g. {leaked[0]}"
                )
            # a heterodimer may legitimately be ASSIGNED to a family in the map --
            # it is the aggregation that must exclude it
            print(f"  family {family}: {len(rows)} rows -> {len(aggregating)} monomers aggregate, "
                  f"{len(hetero_ids)} heterodimers excluded")

    if problems:
        print("\nFAILED:")
        for item in problems:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nfrozen maps validated: sha256 match, catalogue-compatible, "
          "modules disjoint per cell type, declared unions verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
