"""Analyze saved G2/n=800/lambda_P=5 general-graph RJMCMC chains.

Run from the Code repository root:
    python3 plots/gclm/simulation_study/base_general_reciprocal_pair/analyze_reciprocal_pairs.py

Only the five retained CSVs are read. No sampler is run. Arrays and graph
columns in the source CSV use zero-based vertices; figures and report use
one-based vertex labels. The stored log-likelihood is a total log-likelihood
up to an additive constant, not a per-observation log-likelihood.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import arviz as az
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case


CASE = get_simulation_case("G2_n800_lambda5_general_base")
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs" / CASE.id
P = CASE.generating_model.p
EDGES = tuple((i, j) for i in range(P) for j in range(P) if i != j)
EDGE_INDEX = {edge: index for index, edge in enumerate(EDGES)}
PAIRS = tuple((i, j) for i in range(P) for j in range(i + 1, P))
EDGE_COLUMNS = [f"edge_{i}_{j}" for i, j in EDGES]
M_COLUMNS = [f"M_{i}_{j}" for i in range(P) for j in range(P)]
SIGMA_COLUMNS = [f"Sigma_{i}_{j}" for i in range(P) for j in range(P)]
READ_COLUMNS = [
    "iteration", "k", "graph_bitmask", "graph_edges_zero_based", "eta",
    "log_likelihood", "move_type", "accepted", "graph_changed",
    *EDGE_COLUMNS, *M_COLUMNS, *SIGMA_COLUMNS,
]
MOVES = ("mean", "contrast", "margin", "coefficient", "reversal", "birth", "death")
MOVE_CODE = {name: i for i, name in enumerate(MOVES)}
FIRST_CHAIN_INTERVALS = (
    (0, 60_000), (60_000, 130_000), (130_000, 240_000),
    (240_000, 270_000), (270_000, 330_000), (330_000, 500_000),
)
CHUNK_SIZE = 50_000


@dataclass
class Chain:
    number: int
    seed: int
    mask: np.ndarray
    k: np.ndarray
    c: np.ndarray
    eta: np.ndarray
    ll: np.ndarray
    move: np.ndarray
    accepted: np.ndarray


class MatrixSums:
    """Stream matrix means without retaining 2.5 million fitted matrices."""

    def __init__(self):
        self.n = 0
        self.eta = 0.0
        self.ll = 0.0
        self.m = np.zeros((P, P), dtype=float)
        self.sigma = np.zeros((P, P), dtype=float)

    def add(self, selected, eta, ll, m, sigma):
        if not np.any(selected):
            return
        self.n += int(np.sum(selected))
        self.eta += float(np.sum(eta[selected]))
        self.ll += float(np.sum(ll[selected]))
        self.m += np.sum(m[selected], axis=0).reshape(P, P)
        self.sigma += np.sum(sigma[selected], axis=0).reshape(P, P)

    def subtract(self, other):
        out = MatrixSums()
        out.n, out.eta, out.ll = self.n - other.n, self.eta - other.eta, self.ll - other.ll
        out.m, out.sigma = self.m - other.m, self.sigma - other.sigma
        return out


def pair_flags(mask):
    """Rows correspond to draws; columns to unordered vertex pairs."""
    return np.column_stack([
        ((mask >> EDGE_INDEX[(i, j)]) & 1).astype(bool)
        & ((mask >> EDGE_INDEX[(j, i)]) & 1).astype(bool)
        for i, j in PAIRS
    ])


def pair_label(pair):
    return f"{pair[0] + 1}↔{pair[1] + 1}"


def graph_edges(mask):
    return [f"{i + 1}→{j + 1}" for index, (i, j) in enumerate(EDGES)
            if int(mask) & (1 << index)]


def write_csv(name, rows, columns=None):
    rows = list(rows)
    if columns is None:
        if not rows:
            raise ValueError(f"No rows for {name}")
        columns = list(rows[0])
    with (HERE / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def describe(values, prefix=""):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {f"{prefix}mean": np.nan, f"{prefix}sd": np.nan,
                f"{prefix}q05": np.nan, f"{prefix}median": np.nan,
                f"{prefix}q95": np.nan}
    q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {f"{prefix}mean": float(values.mean()),
            f"{prefix}sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            f"{prefix}q05": float(q05), f"{prefix}median": float(median),
            f"{prefix}q95": float(q95)}


def read_chain(number, seed, sums):
    path = RUNS / f"chain_{number:02d}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        first = next(reader)
    expected = (CASE.id, "G2", "800", "general", "5", "0", "0",
                str(number), str(seed), str(CASE.generating_model.dataset_seed))
    actual = tuple(first[name] for name in (
        "configuration_id", "generating_model", "n", "inference_graph_space",
        "lambda_p", "adaptive_margin", "cycle_moves", "chain_number",
        "chain_seed", "dataset_seed",
    ))
    if actual != expected:
        raise ValueError(f"Metadata mismatch in {path}: {actual}")

    parts = {name: [] for name in ("mask", "k", "c", "eta", "ll", "move", "accepted")}
    offset = 0
    previous_mask = None
    previous_k = None
    for chunk in pd.read_csv(path, usecols=READ_COLUMNS, chunksize=CHUNK_SIZE):
        n = len(chunk)
        iterations = chunk["iteration"].to_numpy(dtype=np.int64)
        expected_iterations = CASE.burn_in + 1 + np.arange(offset, offset + n)
        if not np.array_equal(iterations, expected_iterations):
            raise ValueError(f"Nonconsecutive retained iterations in {path}")
        mask = chunk["graph_bitmask"].to_numpy(dtype=np.uint32)
        indicators = chunk[EDGE_COLUMNS].to_numpy(dtype=np.uint8)
        if not np.isin(indicators, [0, 1]).all():
            raise ValueError(f"Nonbinary edge indicator in {path}")
        encoded = indicators.astype(np.uint32) @ (1 << np.arange(len(EDGES), dtype=np.uint32))
        k = chunk["k"].to_numpy(dtype=np.uint8)
        if not np.array_equal(encoded, mask) or not np.array_equal(indicators.sum(axis=1), k):
            raise ValueError(f"Graph indicators, bitmask, or k disagree in {path}")
        # Check the explicit graph representation on a deterministic sample.
        for index in range(0, n, 997):
            recorded = tuple(tuple(edge) for edge in json.loads(chunk["graph_edges_zero_based"].iat[index]))
            decoded = tuple(edge for bit, edge in enumerate(EDGES) if mask[index] & (1 << bit))
            if recorded != decoded:
                raise ValueError(f"Explicit edges disagree with bitmask in {path}")

        flags = pair_flags(mask)
        c = flags.sum(axis=1).astype(np.uint8)
        if np.any(2 * c > k):
            raise ValueError(f"Invalid reciprocal-pair count in {path}")
        eta = chunk["eta"].to_numpy(dtype=float)
        ll = chunk["log_likelihood"].to_numpy(dtype=float)
        if not np.isfinite(eta).all() or np.any(eta <= 0) or not np.isfinite(ll).all():
            raise ValueError(f"Non-finite eta or log-likelihood in {path}")
        move = chunk["move_type"].map(MOVE_CODE)
        if move.isna().any():
            raise ValueError(f"Unknown move in {path}")
        accepted = chunk["accepted"].to_numpy(dtype=np.uint8)
        changed = chunk["graph_changed"].to_numpy(dtype=np.uint8)
        if not np.isin(accepted, [0, 1]).all() or not np.isin(changed, [0, 1]).all():
            raise ValueError(f"Nonbinary acceptance/graph-change flag in {path}")
        if np.any(changed > accepted):
            raise ValueError(f"Graph change without acceptance in {path}")
        prior_mask = np.r_[previous_mask, mask[:-1]] if previous_mask is not None else mask[:-1]
        actual_changed = mask != prior_mask if previous_mask is not None else mask[1:] != mask[:-1]
        stored_changed = changed if previous_mask is not None else changed[1:]
        if not np.array_equal(actual_changed, stored_changed):
            raise ValueError(f"Stored graph-change flag disagrees with trajectory in {path}")
        before_k = np.r_[previous_k, k[:-1]] if previous_k is not None else k[:-1]
        after_k = k if previous_k is not None else k[1:]
        checked_move = move.to_numpy(dtype=np.uint8) if previous_k is not None else move.to_numpy(dtype=np.uint8)[1:]
        checked_change = changed if previous_k is not None else changed[1:]
        delta = after_k.astype(np.int16) - before_k.astype(np.int16)
        valid_change = (
            ((checked_move == MOVE_CODE["birth"]) & (delta == 1))
            | ((checked_move == MOVE_CODE["death"]) & (delta == -1))
            | ((checked_move == MOVE_CODE["reversal"]) & (delta == 0))
        )
        if np.any(checked_change.astype(bool) & ~valid_change):
            raise ValueError(f"Graph-size change is inconsistent with move type in {path}")
        previous_mask = mask[-1]
        previous_k = k[-1]

        m = chunk[M_COLUMNS].to_numpy(dtype=float)
        sigma = chunk[SIGMA_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(m).all() or not np.isfinite(sigma).all():
            raise ValueError(f"Non-finite fitted matrix in {path}")
        sums["all"].add(np.ones(n, dtype=bool), eta, ll, m, sigma)
        sums["c0"].add(c == 0, eta, ll, m, sigma)
        sums["cpos"].add(c > 0, eta, ll, m, sigma)
        for pair_index, pair in enumerate(PAIRS):
            sums[pair].add(flags[:, pair_index], eta, ll, m, sigma)

        for key, values in (("mask", mask), ("k", k), ("c", c),
                            ("eta", eta), ("ll", ll),
                            ("move", move.to_numpy(dtype=np.uint8)),
                            ("accepted", accepted.astype(bool))):
            parts[key].append(values.copy())
        offset += n
    if offset != CASE.iterations - CASE.burn_in:
        raise ValueError(f"Expected 500,000 retained draws in {path}; found {offset}")
    chain = Chain(number, seed, *(np.concatenate(parts[name]) for name in parts))
    print(f"Loaded and validated chain {number}/5", flush=True)
    return chain


def run_bounds(flags):
    """Return half-open start/end indices of True runs."""
    changes = np.diff(np.r_[False, flags, False].astype(np.int8))
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def elevated_regimes(chain, width=5_000, gap=1_000, min_length=5_000):
    """Find sustained high-k stretches using an explicit exploratory rule."""
    k = chain.k.astype(float)
    cumsum = np.r_[0.0, np.cumsum(k)]
    left = np.maximum(np.arange(len(k)) - width // 2, 0)
    right = np.minimum(np.arange(len(k)) + (width + 1) // 2, len(k))
    smooth = (cumsum[right] - cumsum[left]) / (right - left)
    threshold = float(np.median(k) + 0.8)
    starts, ends = run_bounds(smooth >= threshold)
    merged = []
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return threshold, [(a, b) for a, b in merged if b - a >= min_length]


def save_core_statistics(chains):
    distribution, pair_rows, joint, move_rows = [], [], [], []
    labels = [(f"chain_{chain.number:02d}", [chain]) for chain in chains]
    labels.append(("pooled", chains))
    for label, group in labels:
        k = np.concatenate([ch.k for ch in group])
        c = np.concatenate([ch.c for ch in group])
        mask = np.concatenate([ch.mask for ch in group])
        n = len(k)
        for value in range(int(c.max()) + 1):
            count = int(np.sum(c == value))
            distribution.append(dict(chain=label, c=value, draws=count, probability=count / n,
                                     mean_c=float(c.mean()), p_any_pair=float(np.mean(c > 0))))
        for i, j in PAIRS:
            first = ((mask >> EDGE_INDEX[(i, j)]) & 1).astype(bool)
            second = ((mask >> EDGE_INDEX[(j, i)]) & 1).astype(bool)
            pair_rows.append(dict(chain=label, vertex_i=i + 1, vertex_j=j + 1,
                                  p_i_to_j=float(first.mean()), p_j_to_i=float(second.mean()),
                                  p_both=float(np.mean(first & second))))
        counts = Counter(zip(k.tolist(), c.tolist()))
        count_k = Counter(k.tolist())
        for (kval, cval), count in sorted(counts.items()):
            joint.append(dict(chain=label, k=int(kval), c=int(cval), draws=count,
                              p_joint=count / n, p_c_given_k=count / count_k[kval]))
        moves = np.concatenate([ch.move for ch in group])
        accepted = np.concatenate([ch.accepted for ch in group])
        for name, code in MOVE_CODE.items():
            selected = moves == code
            proposed = int(selected.sum())
            accepted_count = int(np.sum(accepted & selected))
            move_rows.append(dict(chain=label, move=name, proposed=proposed,
                                  accepted=accepted_count,
                                  acceptance_rate=accepted_count / proposed if proposed else np.nan))
    write_csv("reciprocal_count_distribution.csv", distribution)
    write_csv("edge_and_pair_inclusion.csv", pair_rows)
    write_csv("k_c_joint.csv", joint)
    write_csv("move_acceptance.csv", move_rows)

    c_values = np.stack([ch.c for ch in chains])
    rows = []
    for chain in chains:
        rows.append(dict(chain=f"chain_{chain.number:02d}", draws=len(chain.c),
                         mean_c=float(chain.c.mean()), p_any_pair=float(np.mean(chain.c > 0)),
                         ess_bulk_c=float(az.ess(chain.c[None, :], method="bulk")),
                         rhat_rank_c="", corr_k_c=float(np.corrcoef(chain.k, chain.c)[0, 1])))
    rows.append(dict(chain="pooled", draws=c_values.size, mean_c=float(c_values.mean()),
                     p_any_pair=float(np.mean(c_values > 0)),
                     ess_bulk_c=float(az.ess(c_values, method="bulk")),
                     rhat_rank_c=float(az.rhat(c_values, method="rank")),
                     corr_k_c=float(np.corrcoef(
                         np.concatenate([ch.k for ch in chains]), c_values.ravel()
                     )[0, 1])))
    write_csv("reciprocal_diagnostics.csv", rows)
    return rows, pair_rows, move_rows


def save_conditional_statistics(chains):
    rows_c, rows_k = [], []
    for label, group in [(f"chain_{ch.number:02d}", [ch]) for ch in chains] + [("pooled", chains)]:
        c = np.concatenate([ch.c for ch in group])
        k = np.concatenate([ch.k for ch in group])
        eta = np.concatenate([ch.eta for ch in group])
        ll = np.concatenate([ch.ll for ch in group])
        for value in sorted(np.unique(c)):
            selected = c == value
            rows_c.append(dict(chain=label, c=int(value), draws=int(selected.sum()),
                               mean_k=float(k[selected].mean()),
                               **describe(eta[selected], "eta_"),
                               **describe(ll[selected], "log_likelihood_")))
        for value in sorted(np.unique(k)):
            selected = k == value
            rows_k.append(dict(chain=label, k=int(value), draws=int(selected.sum()),
                               mean_c=float(c[selected].mean()),
                               **describe(eta[selected], "eta_"),
                               **describe(ll[selected], "log_likelihood_")))
    write_csv("continuous_by_reciprocal_count.csv", rows_c)
    write_csv("continuous_by_graph_size.csv", rows_k)
    return rows_c


def save_regimes(chains):
    intervals = [("chain_01", "exploratory", a, b) for a, b in FIRST_CHAIN_INTERVALS]
    automatic = []
    for chain in chains:
        threshold, spans = elevated_regimes(chain)
        for start, end in spans:
            automatic.append(dict(chain=f"chain_{chain.number:02d}",
                                  start=start, end=end, length=end - start,
                                  k_rolling_threshold=threshold,
                                  mean_k=float(chain.k[start:end].mean()),
                                  mean_c=float(chain.c[start:end].mean()),
                                  mean_eta=float(chain.eta[start:end].mean()),
                                  mean_log_likelihood=float(chain.ll[start:end].mean())))
            intervals.append((f"chain_{chain.number:02d}", "automatic_high_k", start, end))
    write_csv("automatic_high_k_regimes.csv", automatic, columns=(
        "chain", "start", "end", "length", "k_rolling_threshold", "mean_k",
        "mean_c", "mean_eta", "mean_log_likelihood"))

    by_number = {f"chain_{ch.number:02d}": ch for ch in chains}
    interval_rows, distribution, graphs, pairs = [], [], [], []
    for chain_name, kind, start, end in intervals:
        chain = by_number[chain_name]
        section = slice(start, end)
        k, c, mask = chain.k[section], chain.c[section], chain.mask[section]
        interval_rows.append(dict(chain=chain_name, interval_type=kind, start=start,
                                  end=end, draws=end - start, mean_k=float(k.mean()),
                                  mean_c=float(c.mean()), p_any_pair=float(np.mean(c > 0)),
                                  **describe(chain.eta[section], "eta_"),
                                  **describe(chain.ll[section], "log_likelihood_")))
        for name, values in (("k", k), ("c", c)):
            unique, counts = np.unique(values, return_counts=True)
            for value, count in zip(unique, counts):
                distribution.append(dict(chain=chain_name, interval_type=kind,
                                         start=start, end=end, quantity=name,
                                         value=int(value), draws=int(count),
                                         probability=float(count / len(values))))
        unique, counts = np.unique(mask, return_counts=True)
        order = np.argsort(counts)[::-1][:10]
        for rank, index in enumerate(order, start=1):
            graph_mask = int(unique[index])
            graphs.append(dict(chain=chain_name, interval_type=kind, start=start,
                               end=end, rank=rank, graph_bitmask=graph_mask,
                               k=int(graph_mask).bit_count(), draws=int(counts[index]),
                               probability=float(counts[index] / len(mask)),
                               edges_one_based=json.dumps(graph_edges(graph_mask), ensure_ascii=False)))
        flags = pair_flags(mask)
        for pair_index, pair in enumerate(PAIRS):
            count = int(flags[:, pair_index].sum())
            pairs.append(dict(chain=chain_name, interval_type=kind, start=start,
                              end=end, pair=pair_label(pair), draws=count,
                              probability=count / len(mask)))
    write_csv("regime_summaries.csv", interval_rows)
    write_csv("regime_distributions.csv", distribution)
    write_csv("regime_top_graphs.csv", graphs)
    write_csv("regime_pairs.csv", pairs)
    return interval_rows, automatic


def save_transitions(chains, automatic):
    rows = []
    boundary_rows = []
    boundaries = {ch.number: [] for ch in chains}
    boundaries[1] += [270_000, 330_000]
    for regime in automatic:
        number = int(regime["chain"].split("_")[1])
        boundaries[number] += [regime["start"], regime["end"]]
    for chain in chains:
        flags = pair_flags(chain.mask)
        for index, pair in enumerate(PAIRS):
            present = flags[:, index]
            starts, ends = run_bounds(present)
            lengths = ends - starts
            arrivals = starts[starts > 0]
            departures = ends[ends < len(present)]
            if np.any(chain.move[arrivals] != MOVE_CODE["birth"]):
                raise ValueError("A reciprocal pair was completed without a birth move")
            if np.any(chain.move[departures] != MOVE_CODE["death"]):
                raise ValueError("A reciprocal pair was broken without a death move")
            if np.any(~chain.accepted[arrivals]) or np.any(~chain.accepted[departures]):
                raise ValueError("A reciprocal-pair transition was not accepted")
            rows.append(dict(chain=f"chain_{chain.number:02d}", pair=pair_label(pair),
                             present_draws=int(present.sum()), p_present=float(present.mean()),
                             completed=int(len(arrivals)), broken=int(len(departures)),
                             residence_episodes=int(len(lengths)),
                             residence_mean=float(lengths.mean()) if len(lengths) else np.nan,
                             residence_median=float(np.median(lengths)) if len(lengths) else np.nan,
                             residence_q95=float(np.quantile(lengths, .95)) if len(lengths) else np.nan,
                             residence_max=int(lengths.max()) if len(lengths) else 0,
                             longest_start=int(starts[np.argmax(lengths)]) if len(lengths) else "",
                             longest_end=int(ends[np.argmax(lengths)]) if len(lengths) else ""))
            for direction, indices in (("completed", arrivals), ("broken", departures)):
                for time in indices:
                    nearest = min((abs(int(time) - b), b) for b in boundaries[chain.number]) if boundaries[chain.number] else None
                    if nearest is not None and nearest[0] <= 2_000:
                        boundary_rows.append(dict(chain=f"chain_{chain.number:02d}",
                                                  boundary=nearest[1], event_index=int(time),
                                                  distance_to_boundary=int(time - nearest[1]),
                                                  pair=pair_label(pair), direction=direction,
                                                  k_before=int(chain.k[time - 1]),
                                                  k_after=int(chain.k[time]),
                                                  eta_before=float(chain.eta[time - 1]),
                                                  eta_after=float(chain.eta[time]),
                                                  log_likelihood_before=float(chain.ll[time - 1]),
                                                  log_likelihood_after=float(chain.ll[time])))
    write_csv("pair_transition_residence.csv", rows)
    write_csv("transitions_near_regime_boundaries.csv", boundary_rows, columns=(
        "chain", "boundary", "event_index", "distance_to_boundary", "pair",
        "direction", "k_before", "k_after", "eta_before", "eta_after",
        "log_likelihood_before", "log_likelihood_after"))
    return rows


def save_matrix_comparisons(sums, pair_rows):
    pooled_pairs = sorted((row for row in pair_rows if row["chain"] == "pooled"),
                          key=lambda row: row["p_both"], reverse=True)
    top_pair = (pooled_pairs[0]["vertex_i"] - 1, pooled_pairs[0]["vertex_j"] - 1)
    groups = {"c=0": sums["c0"], "c>=1": sums["cpos"],
              f"{pair_label(top_pair)} present": sums[top_pair],
              f"{pair_label(top_pair)} absent": sums["all"].subtract(sums[top_pair])}
    matrix_rows, comparison_rows = [], []
    for label, group in groups.items():
        for i in range(P):
            for j in range(P):
                matrix_rows.append(dict(group=label, draws=group.n,
                                        row=i + 1, column=j + 1,
                                        mean_M=float(group.m[i, j] / group.n),
                                        mean_Sigma=float(group.sigma[i, j] / group.n)))
    for left, right in (("c=0", "c>=1"),
                        (f"{pair_label(top_pair)} present", f"{pair_label(top_pair)} absent")):
        a, b = groups[left], groups[right]
        comparison_rows.append(dict(group_a=left, group_b=right,
                                    draws_a=a.n, draws_b=b.n,
                                    mean_eta_a=a.eta / a.n, mean_eta_b=b.eta / b.n,
                                    mean_log_likelihood_a=a.ll / a.n,
                                    mean_log_likelihood_b=b.ll / b.n,
                                    frobenius_mean_M=float(np.linalg.norm(a.m / a.n - b.m / b.n, "fro")),
                                    frobenius_mean_Sigma=float(np.linalg.norm(a.sigma / a.n - b.sigma / b.n, "fro"))))
    write_csv("conditional_mean_matrices.csv", matrix_rows)
    write_csv("matrix_group_comparisons.csv", comparison_rows)
    return comparison_rows


def save_eta_strata(chains):
    eta = np.concatenate([ch.eta for ch in chains])
    low_cut, high_cut = np.quantile(eta, [.25, .75])
    summary, graphs = [], []
    for label, select_fn in (("low_eta_quartile", lambda x: x <= low_cut),
                             ("high_eta_quartile", lambda x: x >= high_cut)):
        for chain in chains:
            selected = select_fn(chain.eta)
            n = int(selected.sum())
            summary.append(dict(stratum=label, chain=f"chain_{chain.number:02d}",
                                draws=n, mean_k=float(chain.k[selected].mean()),
                                mean_c=float(chain.c[selected].mean()),
                                mean_eta=float(chain.eta[selected].mean()),
                                mean_log_likelihood=float(chain.ll[selected].mean())))
            unique, counts = np.unique(chain.mask[selected], return_counts=True)
            for rank, index in enumerate(np.argsort(counts)[::-1][:5], start=1):
                graph_mask = int(unique[index])
                graphs.append(dict(stratum=label, chain=f"chain_{chain.number:02d}",
                                   rank=rank, graph_bitmask=graph_mask,
                                   k=graph_mask.bit_count(), count=int(counts[index]),
                                   probability_within_stratum=float(counts[index] / n),
                                   edges_one_based=json.dumps(graph_edges(graph_mask), ensure_ascii=False)))
    write_csv("eta_quartile_summaries.csv", summary)
    write_csv("eta_quartile_top_graphs.csv", graphs)
    return low_cut, high_cut, summary


def style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "pdf.fonttype": 42,
                         "savefig.bbox": "tight"})


def plot_traces(chains):
    style()
    colors = ("#254b70", "#b34e38", "#4a806b")
    for chain, filename, span in (
        (chains[0], "trace_chain_01.pdf", (0, len(chains[0].k))),
        (chains[0], "focus_240000_350000.pdf", (240_000, 350_000)),
    ):
        start, end = span
        focused = filename.startswith("focus")
        fig, axes = plt.subplots(4 if focused else 3, 1,
                                 figsize=(9, 7.7 if focused else 6.5), sharex=True,
                                 constrained_layout=True)
        x = np.arange(start, end)
        series = [(chain.k, "Off-diagonal edges $k$", colors[0]),
                  (chain.c, "Reciprocal pairs $c(G)$", colors[1]),
                  (chain.eta, "Stability margin $\\eta$", colors[2])]
        if focused:
            flags = pair_flags(chain.mask)
            leading_pair = int(np.argmax(flags[270_000:330_000].mean(axis=0)))
            i, j = PAIRS[leading_pair]
            series.append((flags[:, leading_pair].astype(int),
                           f"Pair ${i + 1}\\leftrightarrow{j + 1}$", "#946c21"))
        for axis, (values, ylabel, color) in zip(axes, series):
            axis.plot(x, values[start:end], lw=.35, color=color, rasterized=True)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.18)
            if focused:
                axis.axvspan(270_000, 330_000, color="#cfa82f", alpha=.12)
        for axis in (axes[0], axes[1], *(axes[3:] if focused else [])):
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[-1].set_xlabel("Retained iteration after burn-in (zero-based)")
        axes[-1].set_xlim(start, end)
        fig.savefig(HERE / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(5, 3, figsize=(12, 10), sharex=True,
                             constrained_layout=True)
    for row, chain in enumerate(chains):
        x = np.arange(0, len(chain.k), 25)
        for col, (values, title, color) in enumerate(zip(
            (chain.k, chain.c, chain.eta), ("$k$", "$c(G)$", "$\\eta$"), colors,
        )):
            ax = axes[row, col]
            ax.plot(x, values[::25], lw=.45, color=color, rasterized=True)
            ax.grid(alpha=.15)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"Chain {chain.number}")
            if col < 2:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if row == 4:
                ax.set_xlabel("Retained iteration")
    fig.savefig(HERE / "trace_all_chains.pdf", dpi=180)
    plt.close(fig)


def plot_distributions(chains, pair_rows):
    style()
    c = np.concatenate([ch.c for ch in chains])
    k = np.concatenate([ch.k for ch in chains])
    values, counts = np.unique(c, return_counts=True)
    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    ax.bar(values, counts / len(c), color="#315f86", width=.75)
    ax.set(xlabel="Number of reciprocal pairs $c(G)$", ylabel="Fraction of retained draws")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=.2)
    fig.savefig(HERE / "reciprocal_count_distribution.pdf")
    plt.close(fig)

    matrix = np.full((P, P), np.nan)
    for row in pair_rows:
        if row["chain"] == "pooled":
            i, j = row["vertex_i"] - 1, row["vertex_j"] - 1
            matrix[i, j] = matrix[j, i] = row["p_both"]
    fig, ax = plt.subplots(figsize=(5.7, 4.5), constrained_layout=True)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f1f1f1")
    image = ax.imshow(matrix, cmap=cmap, norm=Normalize(0, np.nanmax(matrix)))
    ax.set_xticks(range(P), [str(i + 1) for i in range(P)])
    ax.set_yticks(range(P), [str(i + 1) for i in range(P)])
    ax.set(xlabel="Vertex", ylabel="Vertex")
    for i in range(P):
        for j in range(P):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if matrix[i, j] > .5 * np.nanmax(matrix) else "black")
    fig.colorbar(image, ax=ax, label="Posterior reciprocal-pair inclusion")
    fig.savefig(HERE / "reciprocal_pair_heatmap.pdf")
    plt.close(fig)

    table = pd.crosstab(k, c, normalize="index")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bottom = np.zeros(len(table))
    palette = plt.cm.viridis(np.linspace(.15, .85, len(table.columns)))
    for color, col in zip(palette, table.columns):
        probability = table[col].to_numpy()
        ax.bar(table.index, probability, bottom=bottom, color=color,
               label=f"$c={col}$", width=.85)
        bottom += probability
    ax.set(xlabel="Number of off-diagonal edges $k$", ylabel="$P(c(G)=c\\mid k)$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Reciprocal pairs", ncol=min(4, len(table.columns)), fontsize=8)
    fig.savefig(HERE / "k_c_relationship.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for ax, variable, ylabel in zip(axes, ("eta", "ll"),
                                    ("Stability margin $\\eta$", "Total log-likelihood")):
        selected_counts = [int(value) for value in values if np.sum(c == value) >= 100]
        data = [np.concatenate([getattr(ch, variable)[ch.c == value] for ch in chains])
                for value in selected_counts]
        ax.boxplot(data, tick_labels=[str(value) for value in selected_counts],
                   showfliers=False, medianprops={"color": "#a44331"})
        ax.set(xlabel="Reciprocal pairs $c(G)$", ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(HERE / "continuous_parameters_by_c.pdf")
    plt.close(fig)


def render_report(chains, diagnostics, pair_rows, intervals, automatic,
                  transitions, comparisons, low_cut, high_cut, strata, move_rows):
    pooled_pairs = sorted((row for row in pair_rows if row["chain"] == "pooled"),
                          key=lambda row: row["p_both"], reverse=True)
    first = [row for row in intervals if row["chain"] == "chain_01" and row["interval_type"] == "exploratory"]
    auto_other = sorted((row for row in automatic if row["chain"] != "chain_01"),
                        key=lambda row: row["length"], reverse=True)
    top = pooled_pairs[0]
    pair = f"{top['vertex_i']}↔{top['vertex_j']}"
    top_transitions = [row for row in transitions if row["pair"] == pair]
    pair_dwell = max(top_transitions, key=lambda row: row["residence_max"])
    comparison = comparisons[0]
    broad_occupancy = pair_flags(chains[0].mask[270_000:330_000]).mean(axis=0)
    broad_pair_index = int(np.argmax(broad_occupancy))
    broad_pair = PAIRS[broad_pair_index]
    broad_pair_fraction = float(broad_occupancy[broad_pair_index])
    long_pair = next(row for row in transitions if row["chain"] == "chain_01" and row["pair"] == pair_label(broad_pair))
    high_01 = max((row for row in automatic if row["chain"] == "chain_01"),
                  key=lambda row: max(0, min(row["end"], 330_000) - max(row["start"], 270_000)))
    high_pair_fraction = float(pair_flags(chains[0].mask[high_01["start"]:high_01["end"]])[:, broad_pair_index].mean())
    birth = next(row for row in move_rows if row["chain"] == "pooled" and row["move"] == "birth")
    death = next(row for row in move_rows if row["chain"] == "pooled" and row["move"] == "death")
    top_other = []
    for row in auto_other[:3]:
        chain = chains[int(row["chain"].split("_")[1]) - 1]
        occupancy = pair_flags(chain.mask[row["start"]:row["end"]]).mean(axis=0)
        index = int(np.argmax(occupancy))
        top_other.append(f"{row['chain']} [{row['start']:,}, {row['end']:,}) "
                         f"({row['length']:,} draws; {pair_label(PAIRS[index])} present "
                         f"{100 * occupancy[index]:.1f}% of draws; mean η={row['mean_eta']:.3f})")
    lines = [
        "# Reciprocal pairs and mixing in the base general-graph run", "",
        "Five independent G2, n=800, λP=5 chains were analyzed from their 500,000 retained draws each. "
        "All positions below are zero-based *retained* indices, with half-open intervals; vertices in figures and tables are one-based. "
        "The samples were not regenerated. The stored log-likelihood is the total log-likelihood up to a constant.", "",
        "## Main findings", "",
        f"Across 2,500,000 retained draws, the mean reciprocal-pair count is {diagnostics[-1]['mean_c']:.3f}; "
        f"{100 * diagnostics[-1]['p_any_pair']:.1f}% of draws contain at least one reciprocal pair. "
        f"The pooled correlation of k and c(G) is {diagnostics[-1]['corr_k_c']:.3f}. "
        f"Bulk ESS for c(G) is {diagnostics[-1]['ess_bulk_c']:.1f} and rank-normalized split R-hat is "
        f"{diagnostics[-1]['rhat_rank_c']:.3f}. These are diagnostic estimates, not independent sample counts.", "",
        "The most frequent reciprocal pairs are " + ", ".join(
            f"{row['vertex_i']}↔{row['vertex_j']} ({100 * row['p_both']:.2f}%)"
            for row in pooled_pairs[:4]) + ". See `reciprocal_pair_heatmap.pdf` and `edge_and_pair_inclusion.csv`.", "",
        "First-chain exploratory windows (these were supplied for inspection, not inferred change points):", "",
        "| Retained indices | Mean k | Mean c(G) | P(c>0) | Mean η | Mean total log-likelihood |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in first:
        lines.append(f"| {row['start']:,}–{row['end']:,} | {row['mean_k']:.3f} | "
                     f"{row['mean_c']:.3f} | {row['p_any_pair']:.3f} | "
                     f"{row['eta_mean']:.3f} | {row['log_likelihood_mean']:.3f} |")
    lines += [
        "", f"In chain 1's 270,000–330,000 interval, {pair_label(broad_pair)} occurs in "
        f"{100 * broad_pair_fraction:.1f}% of draws. Its longest uninterrupted residence is "
        f"[{long_pair['longest_start']:,}, {long_pair['longest_end']:,}), or "
        f"{long_pair['residence_max']:,} draws, overlapping the detected high-k stretch "
        f"[{high_01['start']:,}, {high_01['end']:,}), where it occurs in "
        f"{100 * high_pair_fraction:.1f}% of draws. The detailed graphs and pair frequencies are in "
        "`regime_top_graphs.csv` and `regime_pairs.csv`. See `trace_chain_01.pdf` and "
        "`focus_240000_350000.pdf`.", "",
        "The other chains were screened with a 5,000-draw rolling mean of k exceeding that chain's "
        "median k by 0.8, merging gaps of at most 1,000 draws and keeping stretches of at least "
        "5,000 draws. This is an exploratory rule, not a formal change-point procedure. "
        "The longest such stretches outside chain 1 are: " + "; ".join(top_other)
        + ". The five-chain trace figure plots every 25th saved draw for display only. "
        "See `trace_all_chains.pdf` and `automatic_high_k_regimes.csv`.", "",
        "## Transitions and continuous parameters", "",
        "The saved rows record the selected move and whether it was accepted, but not the identity of a "
        "rejected proposed edge or its proposed parameter values. Thus `move_acceptance.csv` gives "
        "overall birth/death proposal acceptance; it cannot give an acceptance rate *conditional on "
        "attempting to complete or break a particular pair*. `pair_transition_residence.csv` counts "
        "only observed, accepted pair completions/breaks and includes residence durations. "
        f"For the most frequent pair {pair}, the longest observed continuous residence is "
        f"{pair_dwell['residence_max']:,} draws in {pair_dwell['chain']}. "
        f"Across all retained draws, {100 * birth['acceptance_rate']:.1f}% of birth proposals "
        f"and {100 * death['acceptance_rate']:.1f}% of death proposals were accepted. "
        f"Chain 1 nevertheless has {long_pair['completed']} observed completions and "
        f"{long_pair['broken']} breaks of {pair_label(broad_pair)}; its median residence is "
        f"{long_pair['residence_median']:.0f} draws against a maximum of "
        f"{long_pair['residence_max']:,}. Many short visits coexist with rare long ones. "
        "`transitions_near_regime_boundaries.csv` lists pair transitions within 2,000 draws of "
        "selected regime boundaries.", "",
        f"When c(G)=0 versus c(G)≥1, the pooled *conditional mean* drift matrices differ by "
        f"{comparison['frobenius_mean_M']:.3f} in Frobenius norm, and the pooled conditional mean "
        f"stationary covariance matrices by {comparison['frobenius_mean_Sigma']:.3f}. "
        f"Their mean η values are {comparison['mean_eta_a']:.3f} and {comparison['mean_eta_b']:.3f}; "
        f"mean total log-likelihoods are {comparison['mean_log_likelihood_a']:.3f} and "
        f"{comparison['mean_log_likelihood_b']:.3f}. These summaries mix over different graphs and "
        "chains, so small mean-matrix distances do not establish draw-wise observational equivalence. "
        "The group matrices and the most common pair's comparison are in `conditional_mean_matrices.csv` "
        "and `matrix_group_comparisons.csv`. Conditional η and likelihood distributions are shown "
        "in `continuous_parameters_by_c.pdf`; their quantiles are in `continuous_by_reciprocal_count.csv`.", "",
        f"Pooled η quartile cutoffs are {low_cut:.4f} and {high_cut:.4f}. "
        "`eta_quartile_top_graphs.csv` lists common graphs in low- and high-η draws for each chain; "
        "these are descriptive partitions of autocorrelated samples.", "",
        "## Interpretation and limits", "",
        "Reciprocal pairs coincide with the detected elevated-k stretches, often with one pair "
        "dominating an entire stretch. This supplies empirical motivation to test a simultaneous "
        "cycle birth/death proposal. Frequent pair entries and exits mean these results do not "
        "support a universal one-edge transition barrier. They also do **not** establish that the "
        "intermediate one-edge graph is an acceptance barrier: rejected target edges and their "
        "likelihoods were not stored. Similar likelihoods across structural groups also do not by "
        "themselves prove structural non-identifiability. Slow mixing, posterior multimodality, "
        "and observational equivalence remain distinct explanations. The between-chain discrepancies "
        "and low ESS mean pooled frequencies should be interpreted cautiously.", "",
        "## Reproduction", "",
        "From the repository root, run "
        "`python3 plots/gclm/simulation_study/base_general_reciprocal_pair/analyze_reciprocal_pairs.py`. "
        "No simulation output is changed. All figures are PDFs; all numerical summaries are CSVs.", "",
    ]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    sums = {key: MatrixSums() for key in ("all", "c0", "cpos", *PAIRS)}
    chains = [read_chain(number, seed, sums)
              for number, seed in enumerate(CASE.chain_seeds, start=1)]
    diagnostics, pair_rows, move_rows = save_core_statistics(chains)
    save_conditional_statistics(chains)
    intervals, automatic = save_regimes(chains)
    transitions = save_transitions(chains, automatic)
    comparisons = save_matrix_comparisons(sums, pair_rows)
    low_cut, high_cut, strata = save_eta_strata(chains)
    plot_traces(chains)
    plot_distributions(chains, pair_rows)
    render_report(chains, diagnostics, pair_rows, intervals, automatic,
                  transitions, comparisons, low_cut, high_cut, strata, move_rows)
    print(f"Saved results to {HERE}", flush=True)


if __name__ == "__main__":
    main()
