from __future__ import annotations

from pathlib import Path

from .run_smoke_batch1 import run_batch1


def main() -> None:
    run_batch1(
        carrier="DL",
        dates=[
            "2025-01-20",
            "2025-01-30",
            "2025-01-11",
            "2025-01-06",
            "2025-01-07",
        ],
        scenarios=24,
        time_limit=15.0,
        scenario_mode="stress",
        include_certificate=True,
        output_dir=Path("results/trc_smoke/batch2_stress"),
    )


if __name__ == "__main__":
    main()
