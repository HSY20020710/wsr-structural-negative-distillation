# Data Provenance

## Private Evaluation Data

The private evaluation set was created through human annotation and human
review. It is used as the unified benchmark for model evaluation and ablation
comparisons.

The original ship welding quality reports and their annotations contain
confidential production information and are not distributed in this
repository. The files under `examples/` are anonymized synthetic records that
demonstrate the public input and output formats; they are not part of the
private evaluation set.

## Frozen Teacher Predictions

Teacher comparisons use frozen prediction caches so that every method is
evaluated on identical teacher outputs. These private caches are not included
in the repository. The commands used to generate and evaluate them are
documented in `docs/reproduction.md`.

## WSR Gate Components

The complete WSR validation pipeline includes:

1. the Wuli-Shili-Renli ontology;
2. entity refinement and consistency checks;
3. relation endpoint remapping;
4. ontology-guided tri-state gating.
