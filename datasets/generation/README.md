# Generation (planned)

Pipeline to *generate* failures (for example via mutation / bug injection) and
capture their logs, following the validity rule in `datasets/README.md`: the
target test must **pass before** and **fail after** the change, and the failure
must be attributable to the change (not to environment/network/install errors).

> Status: **planned.** No generation code exists yet. Large-scale generation must
> never run in CI.
