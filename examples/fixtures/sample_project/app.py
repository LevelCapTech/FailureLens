"""Entry point that triggers the latent bug in the sample project.

Run it to reproduce the failure captured in ``failure.log``:

    cd examples/fixtures/sample_project && python app.py
"""

from calculator.core import average


def report_scores(scores: list[float]) -> None:
    print(f"Average score: {average(scores)}")


def main() -> None:
    # An empty result set slips through and triggers the failure.
    report_scores([])


if __name__ == "__main__":
    main()
