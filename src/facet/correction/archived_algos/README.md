# Archived correction algorithms

The Python files in this directory are unchanged snapshots of the former
hard-coded AAS, FARM, and structural/motion averaging-matrix implementations.
They are retained for historical comparison and are intentionally not imported
by FACETpy.

Active correction modes use `facet.correction.flex.Flex` with the named
recipes in `facet.correction.presets`. Backwards-compatible public processor
names are thin Flex adapters in `facet.correction.legacy_adapters`.

The archived files are source records, not an importable package; their
relative imports reflect their original location.
