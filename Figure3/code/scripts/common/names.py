"""One canonical mapping between raw heart species IDs and figure labels."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SpeciesNames:
    table: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path) -> 'SpeciesNames':
        table = pd.read_csv(path).sort_values('display_order')
        required = {'species_id', 'full_name', 'short_name', 'display_order'}
        if not required.issubset(table) or table.species_id.duplicated().any():
            raise ValueError(f'Invalid species name mapping: {path}')
        return cls(table.set_index('species_id'))

    @property
    def order(self) -> list[str]:
        return self.table.index.tolist()

    def display(self, raw_id: str, compact: bool = False) -> str:
        column = 'short_name' if compact else 'full_name'
        return str(self.table.loc[raw_id, column])
