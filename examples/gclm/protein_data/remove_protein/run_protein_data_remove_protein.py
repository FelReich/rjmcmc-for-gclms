"""Run the existing simple_li configuration on the Raf-removed protein data.

Five independent chains are written to this directory's runs folder.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.protein_data.run_protein_data import (  # noqa: E402
    load_data, run_chain, verify_completed,
)
from examples.gclm.protein_data.settings import get_simulation_case  # noqa: E402


DATA_PATH = Path(__file__).resolve().with_name("preprocessed_remove_protein.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip verified completed chains and restart interrupted chains",
    )
    args = parser.parse_args()

    case = replace(
        get_simulation_case("simple_li"),
        id="simple_li_remove_raf",
        data_path=DATA_PATH,
    )
    names, observations = load_data(case.data_path)
    if len(names) != 10 or "Raf" in names:
        raise ValueError("Expected the ten standardized protein columns without Raf")

    folder = Path(__file__).resolve().parent / "runs"
    started = perf_counter()
    for chain_number, seed in enumerate(case.chain_seeds, start=1):
        progress = f"[{chain_number}/{len(case.chain_seeds)}]"
        destination = folder / f"chain_{chain_number:02d}.csv.gz"
        partial = destination.with_name(destination.name + ".partial")
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"{destination} already exists; use --resume")
            verify_completed(destination, case, chain_number, seed,
                             len(observations), len(names))
            print(f"{progress} Skipped completed {destination}", flush=True)
            continue
        if partial.exists() and not args.resume:
            raise FileExistsError(f"{partial} exists; use --resume to restart")

        folder.mkdir(parents=True, exist_ok=True)
        print(f"{progress} Running {case.id} chain {chain_number} (seed {seed})",
              flush=True)
        chain_started = perf_counter()
        run_chain(case, chain_number, seed, observations, destination)
        print(f"{progress} Saved {destination} "
              f"in {perf_counter() - chain_started:.1f}s", flush=True)
    print(f"Done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
