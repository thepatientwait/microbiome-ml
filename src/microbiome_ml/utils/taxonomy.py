from enum import IntEnum
from typing import Optional

class TaxonomicRanks(IntEnum):
    """Enumeration of supported taxonomic levels."""
    DOMAIN = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6

    @property
    def name(self) -> str:
        return super().name.lower()
    
    @property
    def prefix(self) -> str:
        return f"{self.name[0]}__"
    
    @property
    def child(self) -> Optional["TaxonomicRanks"]:
        """Get the child (more specific) taxonomic rank."""
        try:
            return TaxonomicRanks(self.value + 1)
        except ValueError:
            return None  # Already at lowest rank
    
    @property
    def parent(self) -> Optional["TaxonomicRanks"]:
        """Get the parent (broader) taxonomic rank."""
        try:
            return TaxonomicRanks(self.value - 1)
        except ValueError:
            return None  # Already at highest rank

    @classmethod
    def from_name(cls, rank: str) -> "TaxonomicRanks":
        """Get enum member from rank name."""
        rank = rank.upper()
        try:
            return cls[rank]
        except KeyError:
            raise ValueError(f"Invalid taxonomic rank: {rank}")

    @classmethod
    def from_prefix(cls, prefix: str) -> "TaxonomicRanks":
        """Get enum member from taxonomic prefix."""
        if not hasattr(cls, '_prefix_to_rank'):
            cls._prefix_to_rank = {rank.prefix: rank for rank in cls}
        
        if prefix not in cls._prefix_to_rank:
            raise ValueError(f"Invalid taxonomic prefix: {prefix}")
        
        return cls._prefix_to_rank[prefix]

    @classmethod
    def iter_from_domain(cls):
        """Yield ranks from DOMAIN (broadest) to SPECIES (most specific)."""
        rank = cls.DOMAIN
        while rank is not None:
            yield rank
            rank = rank.child

    @classmethod
    def iter_from_species(cls):
        """Yield ranks from SPECIES (most specific) to DOMAIN (broadest)."""
        rank = cls.SPECIES
        while rank is not None:
            yield rank
            rank = rank.parent

    def iter_up(self):
        """Yield ranks from the current rank up to DOMAIN (inclusive)."""
        rank = self
        while rank is not None:
            yield rank
            rank = rank.parent

    def iter_down(self):
        """Yield ranks from the current rank down to SPECIES (inclusive)."""
        rank = self
        while rank is not None:
            yield rank
            rank = rank.child

