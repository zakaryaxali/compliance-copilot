# Source standards

Source PDFs are **not committed** — they have redistribution terms (PCI SSC, FATF, AICPA each publish their standards under their own conditions). Download them locally and run the indexer.

## PCI-DSS 4.0.1

Download the canonical PDF from the PCI SSC document library:

- https://www.pcisecuritystandards.org/document_library/ (click-through agreement)

The file is named `PCI-DSS-v4_0_1.pdf`, ~4.3 MB, 397 pages, published June 2024.

Build the index:

```bash
python src/build_source_index.py /path/to/PCI-DSS-v4_0_1.pdf \
    --out data/sources/pci_dss_4_0_1.jsonl
```

Expected output: 307 requirements across the 12 principal requirements (Reqs 1–12). Spot-checked critical IDs: `3.3`, `3.4`, `3.4.1`, `3.5`, `3.5.1`, `3.5.1.1`, `8.4`, `10.4.2.1`, `12.10.1` all present.

### Known parser limitations

- One sub-requirement title (`10.4.2.1`) is truncated mid-sentence due to a PDF column-wrap edge case. The ID itself is captured correctly, so the Tier 1 citation-exists check is unaffected.
- Appendix A requirements (`A1.x`, `A2.x`, `A3.x`) are not yet captured — the parser only scans the body (pages 40–333). Add them if/when the dataset needs to cover service-provider-specific or DESV requirements.

## FATF 40 Recommendations (Nov 2023)

Not yet supported. The indexer pattern (`{requirement_id, title, source}`) generalizes — extend `src/build_source_index.py` with a separate parser when needed.

## SOC 2 Trust Services Criteria

Not yet supported. AICPA publishes the TSC document with stricter redistribution terms than PCI/FATF; sourcing strategy needs to be decided before adding a parser.
