"""
Minimal NSGA-II for multi-objective minimization on R^n (typically two objectives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class NSGA2Result:
    pareto_params: np.ndarray
    pareto_F: np.ndarray


def _dominates(v: np.ndarray, w: np.ndarray) -> bool:
    tol = 1e-14
    return bool(np.all(v <= w + tol) and np.any(v + tol < w))


def fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    n = int(F.shape[0])
    n_dominated = np.zeros(n, dtype=np.int32)
    dom_set: List[set[int]] = [set() for _ in range(n)]
    first: List[int] = []

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            fp, fq = F[p], F[q]
            if _dominates(fp, fq):
                dom_set[p].add(q)
            elif _dominates(fq, fp):
                n_dominated[p] += 1
        if n_dominated[p] == 0:
            first.append(p)

    fronts: List[np.ndarray] = [np.asarray(first, dtype=np.int64)]
    ix = 0
    while True:
        nx: List[int] = []
        for p in fronts[ix].tolist():
            for q in dom_set[int(p)]:
                n_dominated[q] -= 1
                if n_dominated[q] == 0:
                    nx.append(q)
        if not nx:
            break
        fronts.append(np.asarray(nx, dtype=np.int64))
        ix += 1

    return fronts


def crowding_distance(F_front: np.ndarray) -> np.ndarray:
    rows = int(F_front.shape[0])
    d = np.zeros(rows, dtype=np.float64)
    if rows <= 2:
        d[:] = np.inf
        return d
    m = int(F_front.shape[1])
    for j in range(m):
        order = np.argsort(F_front[:, j])
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        f_min = float(F_front[order[0], j])
        f_max = float(F_front[order[-1], j])
        if f_max <= f_min + 1e-30:
            continue
        for k in range(1, rows - 1):
            d[order[k]] += (
                float(F_front[order[k + 1], j]) - float(F_front[order[k - 1], j])
            ) / (f_max - f_min)
    return d


def environmental_selection(pop: np.ndarray, Fscores: np.ndarray, target_n: int) -> Tuple[np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(Fscores)
    chosen: List[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= target_n:
            chosen.extend(int(x) for x in front)
            continue

        remainder = target_n - len(chosen)
        Fi = Fscores[front.astype(np.int64)]
        dist = crowding_distance(Fi)
        order = np.argsort(-dist)
        pick = front[order[:remainder]].astype(int)
        chosen.extend(int(x) for x in pick)
        break

    ci = np.asarray(chosen[:target_n], dtype=np.int64)
    return pop[ci], Fscores[ci]


def _sbx(pa: float, pb: float, xl: float, xu: float, rng: np.random.Generator, eta: float) -> Tuple[float, float]:
    if abs(pa - pb) < 1e-14:
        return pa, pb
    pa_i = min(pa, pb)
    pb_i = max(pa, pb)
    if rng.random() > 0.5:
        return pa_i, pb_i
    u = float(rng.random())
    beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    c1 = 0.5 * ((pa_i + pb_i) - beta * (pb_i - pa_i))
    c2 = 0.5 * ((pa_i + pb_i) + beta * (pb_i - pa_i))
    c1 = min(max(c1, xl), xu)
    c2 = min(max(c2, xl), xu)
    return c1, c2


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, xl: np.ndarray, xu: np.ndarray, rng: np.random.Generator, eta: float = 15.0):
    n = len(p1)
    c1 = np.empty(n, dtype=np.float64)
    c2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        cc1, cc2 = _sbx(p1[i], p2[i], float(xl[i]), float(xu[i]), rng, eta)
        c1[i] = cc1
        c2[i] = cc2
    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
    eta: float = 20.0,
):
    """Deb polynomial mutation."""
    n = len(x)
    pm = 1.0 / n
    out = x.copy().astype(np.float64)
    for i in range(n):
        if rng.random() >= pm:
            continue
        y = out[i]
        yl = float(xl[i])
        yu = float(xu[i])
        dy = yu - yl
        if dy <= 0:
            continue
        r = float(rng.random())
        dq = abs((yu - y) / dy)
        dqhi = abs((y - yl) / dy)
        if r < 0.5:
            val = dq ** (eta + 1)
            delta = float((2 * r + (1 - 2 * r) * val) ** (1 / (eta + 1)) - 1)
        else:
            val = dqhi ** (eta + 1)
            delta = float(1 - (2 * (1 - r) + 2 * (r - 0.5) * val) ** (1 / (eta + 1)))

        yi = float(y + delta * dy)
        out[i] = min(max(yi, yl), yu)

    return out


def tournament(pick_idx: np.ndarray, Fscores: np.ndarray, rng: np.random.Generator) -> int:
    a, b = int(rng.integers(0, len(pick_idx))), int(rng.integers(0, len(pick_idx)))
    ia, ib = int(pick_idx[a]), int(pick_idx[b])
    Fa, Fb = Fscores[ia], Fscores[ib]
    if _dominates(Fa, Fb):
        return ia
    if _dominates(Fb, Fa):
        return ib
    return ia if rng.random() < 0.5 else ib


def run_nsga2(
    evaluate_population: Callable[[np.ndarray], np.ndarray],
    xl: np.ndarray,
    xu: np.ndarray,
    population_size: int,
    n_generations: int,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = 42,
):
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)
    n_var = int(xl.shape[0])
    rng = rng if rng is not None else np.random.default_rng(seed)

    rng_u = rng.random((population_size, n_var))
    pop = xl + rng_u * (xu - xl)

    ids = np.arange(population_size, dtype=np.int64)
    Fscores = evaluate_population(pop)

    for _ in range(n_generations):
        offspring_children: List[np.ndarray] = []
        while len(offspring_children) < population_size:
            i_a = tournament(ids, Fscores, rng)
            i_b = tournament(ids, Fscores, rng)
            ch1, ch2 = sbx_crossover(pop[i_a], pop[i_b], xl, xu, rng)
            offspring_children.append(ch1)
            offspring_children.append(ch2)
        raw_off = offspring_children[:population_size]
        off = np.asarray(raw_off, dtype=np.float64)
        mutated_pop = np.empty_like(off)
        for j in range(int(population_size)):
            mutated_pop[j] = polynomial_mutation(off[j], xl, xu, rng)

        offspring_F = evaluate_population(mutated_pop)

        comb_pop = np.vstack([pop, mutated_pop])
        comb_F = np.vstack([Fscores, offspring_F])
        pop, Fscores = environmental_selection(comb_pop, comb_F, population_size)

    fronts = fast_non_dominated_sort(Fscores)
    f0 = fronts[0].astype(np.int64) if fronts else np.array([], dtype=np.int64)

    pareto_params = pop[f0] if len(f0) else np.zeros((0, n_var))
    pareto_F = Fscores[f0] if len(f0) else np.zeros((0, Fscores.shape[1]))
    return NSGA2Result(pareto_params=pareto_params, pareto_F=pareto_F)
