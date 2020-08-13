import itertools as it
import heapq
import subprocess
import math
import operator
from collections import defaultdict
from collections import namedtuple
from functools import lru_cache
from functools import reduce
from random import choice
from random import sample
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar

import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from mapdata import roads_200, roads_0


Vertex = Tuple[float, float]
Edge = Tuple[Vertex, Vertex]
Route = Sequence[Edge]
Solution = Sequence[Edge]


T = TypeVar('T')
def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def just_letters(df: pd.DataFrame, letters: str) -> pd.DataFrame:
    name = df.EZIRDNMLBL.str.lower()
    masks = []
    for letter in letters:
        masks.append(name.str.startswith(letter))
        mask = reduce(operator.or_, masks)
        return df[mask]


@lru_cache(maxsize=1024*1024)
def nav(orig: Vertex, dest: Vertex) -> Optional[Route]:

    def h(vertex: Vertex) -> float:
        return distance(vertex, dest)

    open: Set[Vertex] = {orig, }
    came_from: Dict[Vertex, Vertex] = dict()
    g_score: Dict[Vertex, float] = defaultdict(lambda: 10 ** 9)
    f_score: Dict[Vertex, float] = defaultdict(lambda: 10 ** 9)

    g_score[orig] = 0
    f_score[orig] = h(orig)

    while open:
        current = min(open, key=f_score.get)
        if current == dest:
            v = current
            path = [v]
            while v in came_from:
                v = came_from[v]
                path.append(v)
            return tuple(pairwise(reversed(path)))

        open.remove(current)

        for neighbour in graph[current]:
            maybe_g_score = g_score[current] + length((current, neighbour))
            if maybe_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = maybe_g_score
                f_score[neighbour] = maybe_g_score + h(neighbour)
                open.add(neighbour)

    return None


def flatten(iterables: Iterable[Iterable]) -> Iterable:
    for iterable in iterables:
        yield from iterable


def solution_to_route(sol: Solution) -> Optional[Route]:
    subroutes = []
    sol = [home] + list(sol) + [home]
    wps = list(flatten(sol))
    for origin, dest in pairwise(wps):
        route = nav(origin, dest)
        if route is None:
            print(f'No route from {origin} to {dest}')
            return None
        subroutes.append(route)
    return tuple(flatten(subroutes))


def distance(orig: Vertex, dest: Vertex) -> float:
    (x1, y1), (x2, y2) = orig, dest
    return math.hypot(x2 - x1, y2 - y1)


def length(edge: Edge) -> float:
    orig, dest = edge
    return distance(orig, dest)


@lru_cache(maxsize=1024 * 1024)
def route_length(route: Route) -> float:
    return sum(map(length, route))


def plot_edges(ax, edges: Sequence[Edge], *args, **kwargs):
    for (x1, y1), (x2, y2) in edges:
        ax.plot((x1, x2), (y1, y2), *args, **kwargs, solid_capstyle='round')


def get_edges(geoms) -> Iterable[Edge]:
    for geom in geoms:
        if geom.geom_type == 'LineString':
            yield geom.coords[0], geom.coords[-1]
        elif geom.geom_type == 'MultiLineString':
            yield from get_edges(geom.geoms)


def build_graph(edges: Sequence[Edge]) -> Mapping[Vertex, Sequence[Vertex]]:
    _g = defaultdict(list)
    for head, tail in edges:
        _g[head].append(tail)
        _g[tail].append(head)
    return _g


def score(solution: Solution) -> float:
    route = solution_to_route(solution)
    if route is None:
        return 10**9
    return route_length(route)


def randoswap(sol: Solution, swaps: int = 1) -> Solution:
    l = len(sol)
    sol = list(sol)
    for _ in range(swaps):
        swap_a = choice(range(l))
        swap_b = choice(range(l))
        sol[swap_a], sol[swap_b] = sol[swap_b], sol[swap_a]
    return tuple(sol)


def randoscrobble(sol: Solution, scrobbles: int = 1) -> Solution:
    l = len(sol)
    for scrobble in range(scrobbles):
        breaks: Set[int] = set()
        while len(breaks) < 2:
            breaks.add(choice(range(l)))
        a, b = sorted(breaks)
        sublists = sample((sol[:a], sol[a:b], sol[b:]), k=3)
        sol = tuple(sublists[0]) + tuple(sublists[1]) + tuple(sublists[2])
    return sol


def generation(prev: Sequence[Solution]) -> Sequence[Solution]:
    cream = prev[:3]
    mutants = []
    for c in cream:
        for mutant in range(3):
            mutants.append(randoswap(c[::], swaps=mutant + 1))
            mutants.append(randoscrobble(c[::], scrobbles=mutant + 1))

    pool = set(mutants).union(cream)
    return sorted(pool, key=score)


just = just_letters(roads_0, "w")
edges = list(get_edges(roads_200.geometry))
goals = list(get_edges(just.geometry))
edges = edges + goals  # 100m truncation changes some edges at the boundary
graph = build_graph(edges)
home_ = (
    (977497.1692172779, -4302403.591923077),
    (977544.9539988617, -4302206.452253673)
)
home = min(edges, key=lambda e: length((e[0], home_[0])))

reach = {}
# any goal unreachable from all five witnesses is removed as entirely
# unreachable.
witnesses = list(sample(goals, k=5))
for goal in goals:
    reach[goal] = -1  # account for self
    for witness in witnesses:
        if nav(goal[0], witness[1]):
            reach[goal] += 1


removable = [g for g, r in reach.items() if r < 1]
print(f'Removing {len(removable)} goals because they are unreachable')
goals = [g for g, r in reach.items() if r >= 1]
sols: Sequence[Solution] = [tuple(sample(goals, k=len(goals))) for _ in range(10)]
checkpoints = [0.0]

subprocess.run('rm ./tmp/route*png', shell=True)
max_gens = 500
status_count = 5
_status_gens = [max_gens * 1.25 ** x for x in range(-status_count, 0)]
status_gens = [0] + [int(round(s, 0)) for s in _status_gens] + [max_gens]
for gen in range(max_gens + 1):
    if gen in status_gens:
        best = sols[0]

        # print('        ', distance.cache_info())
        # print('        ', nav.cache_info())
        km = round(score(best) / 1000, 2)
        print(str(gen).rjust(6), km)
        checkpoints.append(score(best))
        fig, ax = plt.subplots(dpi=100, figsize=(10, 10))
        roads_200.plot(ax=ax, color='#009999', alpha=0.25)
        plot_edges(ax, [home] + goals + [home], lw=4, color='red')
        best_route = solution_to_route(best)
        if best_route:
            plot_edges(ax, best_route, color='orange')
        ax.set(xticks=[], yticks=[])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set(title=f"gen {gen!s:>6}: {km:0.2f}km")
        fig.set_facecolor('white')
        fig.savefig(f"./tmp/route-{gen:05d}.png")
        plt.close()
        if not best_route:
            break
    sols = generation(sols)

# subprocess.run('convert -delay 50 -loop 0 ./tmp/route*png ./tmp/route.gif',
#                shell=True)
print('ok')
