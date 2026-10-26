# Sources (notes)

Notes on candidate data sources and, critically, their **license and
redistribution terms**. Record for each source whether its data (or logs/code
derived from it) may be redistributed, and under what conditions.

Candidate sources discussed in the project issue (verify terms before use):

- **SWE-smith** - generates synthetic software-fix tasks from real repositories
  ([GitHub](https://github.com/SWE-bench/SWE-smith),
  [dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith)). Not every
  instance is directly usable for fault localization; check for failing tests,
  obtainable stack traces, and changed-code locations.
- **SWE-bench** - real issues, fix patches, and repository history
  ([GitHub](https://github.com/SWE-bench/SWE-bench)). A fix patch is **not** a
  complete root-cause label, and many cases lack real error logs.
- Open-source bug-fix histories, test-failure/fix commit pairs, mutation-testing
  data, and license-cleared public bug datasets.

> Status: **notes only.** No third-party data is stored here. Prefer publishing
> collection/transformation *code* over redistributing data whose terms are
> unclear.
