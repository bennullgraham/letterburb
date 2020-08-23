import argparse
import heapq
import itertools as it
import math
import operator
import re
import subprocess
import sys
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
from typing import Union

import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import CRS
from pyproj import Transformer

from mapdata import get_mapdata


def latlon_to_metres(lat, lon):
    # epsg 4326 is lat/lon; 3112 is "geoscience australia lambert" which has
    # metres as its unit.
    transformer = Transformer.from_crs(4326, 3112)
    return transformer.transform(lat, lon)


def latlon(val):
    if not re.match(r"^[-\d\.]+,[-\d\.]+$", val):
        raise ValueError("r{val} should be 12.34,56.78 lat lon pair")
    lat, lon = map(float, val.split(","))

    # switch to epsg 3112 which is used everywhere else
    return latlon_to_metres(lat, lon)


parser = argparse.ArgumentParser()
parser.add_argument("postcode", type=int, help="3xxx postcode to ride in")
parser.add_argument("letter", type=str, help="ride streets beginning with this letter")
parser.add_argument(
    "home", type=latlon, help="lat,lon pair; where to start/finish rides"
)
args = parser.parse_args()

Vertex = Tuple[float, float]
Edge = Tuple[Vertex, Vertex]
Route = Sequence[Edge]
Solution = Sequence[Edge]
T = TypeVar("T")


def midpoint(edge: Edge) -> Vertex:
    return (
        (edge[0][0] + edge[1][0]) / 2,
        (edge[0][1] + edge[1][1]) / 2,
    )


@lru_cache(maxsize=1024 * 1024)
def distance(orig: Vertex, dest: Vertex) -> float:
    return math.hypot(dest[0] - orig[0], dest[1] - orig[1])


@lru_cache(maxsize=1024 * 1024)
def length(edge: Edge) -> float:
    return math.hypot(edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])


@lru_cache(maxsize=1024 * 1024)
def route_length(route: Route) -> float:
    return sum(map(length, route))


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def flatten(iterables: Iterable[Iterable]) -> Iterable:
    for iterable in iterables:
        yield from iterable


def just_letters(df: pd.DataFrame, letters: str) -> pd.DataFrame:
    """
    Filter road dataframe down to roads starting with one of `letters`.
    """
    name = df.EZIRDNMLBL.str.lower()
    masks = []
    for letter in letters:
        masks.append(name.str.startswith(letter))
        mask = reduce(operator.or_, masks)
    return df[mask]


def group_by_contiguity(edges: Sequence[Edge]) -> List[List[Edge]]:
    """
    Group together sequential edges which share vertices.

    Contiguous edges will only appear in the same group if they are input in
    the right order. The "_2" version of this function forces the same group.

    TODO this is probably biased where two branches of edges fork from a common
    edge-trunk... one contiguous block will be root+branch, and the other just
    branch.
    """
    if not edges:
        return [[]]
    head = [edges[0]]
    contigs = [head]
    for (phead, ptail), edge in pairwise(edges):
        if phead not in edge and ptail not in edge:
            head = []
            contigs.append(head)
        head.append(edge)
    return contigs


def group_by_contiguity_2(edges: Sequence[Edge]) -> List[List[Edge]]:
    """
    Group together sequential edges which share vertices.
    """
    contigs: List[List[Edge]] = []

    for edge in edges:
        head, tail = edge
        revedge = (tail, head)
        for contig in contigs:
            chead = contig[0][0]
            ctail = contig[-1][1]
            if head == chead:
                contig.insert(0, edge)
                break
            elif head == ctail:
                contig.append(edge)
                break
            elif tail == chead:
                contig.insert(0, revedge)
                break
            elif tail == ctail:
                contig.append(revedge)
                break
        else:
            contigs.append([edge])

    # TODO I think I can still get separate groups here depending on the input
    # ordering. Maybe a second pass to stick the chead/ctails together if they
    # match?

    return contigs


@lru_cache(maxsize=1024 * 1024)
def nav(orig: Vertex, dest: Vertex) -> Optional[Route]:
    """
    Return route from orig to dest, or None if impossible.

    Uses A* to walk the delightfully global `graph` when routing. The returned
    route is a series of contiguous edges like:

        [(A, B), (B, C), (C, D), ... ]
    """

    @lru_cache(maxsize=1024 * 1024)
    def h(vertex: Vertex) -> float:
        """
        Heuristic function is just the crow-distance from dest.
        """
        return distance(vertex, dest)

    came_from: Dict[Vertex, Vertex] = dict()
    g_score: Dict[Vertex, float] = defaultdict(lambda: 10 ** 9)
    f_score: Dict[Vertex, float] = defaultdict(lambda: 10 ** 9)

    g_score[orig] = 0
    f_score[orig] = h(orig)

    open: List[Tuple[float, Vertex]] = []
    heapq.heappush(open, (f_score[orig], orig))

    while open:
        _, current = heapq.heappop(open)
        if current == dest:
            v = current
            path = [v]
            while v in came_from:
                v = came_from[v]
                path.append(v)
            return tuple(pairwise(reversed(path)))

        for neighbour in graph[current]:
            maybe_g_score = g_score[current] + distance(current, neighbour)
            if maybe_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = maybe_g_score
                f_score[neighbour] = maybe_g_score + h(neighbour)
                # TODO check if dupe
                heapq.heappush(open, (f_score[neighbour], neighbour))

    return None


@lru_cache(maxsize=1024 * 1024)
def solution_to_route(sol: Solution) -> Optional[Route]:
    """
    Plot a route between a given ordering of goal-roads
    """
    subroutes = []
    sol = [home] + list(sol) + [home]
    wps = list(flatten(sol))
    for origin, dest in pairwise(wps):
        route = nav(origin, dest)
        if route is None:
            print(f"No route from {origin} to {dest}")
            return None
        subroutes.append(route)
    return tuple(flatten(subroutes))


def plot_edges(ax, edges: Sequence[Edge], offset=0, *args, **kwargs):
    """
    Draw edges nicely on axis.

    `offset` displaces the drawn edge gradually from (-offset, -offset) to
    (offset, offset). This sort of helps when trying to read routes which visit
    the same road twice, but mostly a double-visit occurs in a proximate parts
    of the route so the offset is too similar to help.
    """
    l = len(edges)
    auto_color = "color" not in kwargs
    # for n, edge in enumerate(edges):
    # TODO edge_geom_map
    for n, ((x1, y1), (x2, y2)) in enumerate(edges):
        r = n / l
        o = (2 * r * offset) - offset
        if auto_color:
            kwargs["color"] = mpl.cm.rainbow(r)
        ax.plot(
            (x1 + o, x2 + o), (y1 + o, y2 + o), *args, **kwargs, solid_capstyle="round"
        )


edge_geom_map = {}


def get_edges(geoms) -> Iterable[Edge]:
    """
    Turn a geopandas geometry column into Edges.

    This discards detail about the shape of road segments; we just preserve the
    beginning and end vertex. This is all you need for routing.
    """
    for geom in geoms:
        if geom.geom_type == "LineString":
            edge = geom.coords[0], geom.coords[-1]
            edge_geom_map[edge] = geom
            yield edge
        elif geom.geom_type == "MultiLineString":
            yield from get_edges(geom.geoms)


def build_graph(edges: Sequence[Edge]) -> Mapping[Vertex, Set[Vertex]]:
    """
    Build a map saying where you can go from each vertex.
    """
    _g = defaultdict(set)
    for head, tail in edges:
        _g[head].add(tail)
        _g[tail].add(head)
    return _g


@lru_cache(maxsize=1024 * 1024)
def score(solution: Solution) -> float:
    route = solution_to_route(solution)
    if route is None:
        return 10 ** 9
    return route_length(route)


def singleswap(sol: Solution, swaps: int = 1) -> Solution:
    l = len(sol)
    sol = list(sol)
    for _ in range(swaps):
        swap_a = choice(range(l))
        swap_b = choice(range(l))
        sol[swap_a], sol[swap_b] = sol[swap_b], sol[swap_a]
    return tuple(sol)


def chunkswap(sol: Solution, swaps: int = 1) -> Solution:
    l = len(sol)
    for scrobble in range(swaps):
        breaks: Set[int] = set()
        while len(breaks) < 3:
            breaks.add(choice(range(l)))
        a, b, c = sorted(breaks)
        sublists = sample((sol[:a], sol[a:b], sol[b:c], sol[c:]), k=4)
        # mypy complains about using the callable `tuple` directly here...
        # wrapping in a lambda fixes. shrug
        sol = reduce(operator.add, map(lambda x: tuple(x), sublists))
    return sol


def contigswap(sol: Solution, swaps: int = 1) -> Solution:
    contigs = group_by_contiguity(sol)
    l = len(contigs)
    for _ in range(swaps):
        swap_a = choice(range(l))
        swap_b = choice(range(l))
        contigs[swap_a], contigs[swap_b] = contigs[swap_b], contigs[swap_a]
    return tuple(flatten(contigs))


def contigrev(sol: Solution, revs: int = 1) -> Solution:
    contigs = group_by_contiguity(sol)
    l = len(contigs)
    for n in range(revs):
        idx = choice(range(l))
        contigs[idx] = [(t, h) for h, t in reversed(contigs[idx])]
    return tuple(flatten(contigs))


def generation(prev: Sequence[Solution]) -> Sequence[Solution]:
    cream = prev[:3]
    mutants = []
    for c in cream:
        for m in (1, 2, 3, 4, 5):
            mutants.append(singleswap(c[::], m))
            mutants.append(chunkswap(c[::], m))
            mutants.append(contigswap(c[::], m))
            mutants.append(contigrev(c[::], m))

    pool = set(mutants).union(cream)
    return sorted(pool, key=score)


mapdata = get_mapdata(postcode=args.postcode)
roads_0 = mapdata["roads_0"]
roads_buffer = mapdata["roads_buffer"]
postcode = mapdata["postcode"]

# remove monash freeway. there is a column in the frame that could remove
# freeways, maintenance tracks etc. should do it that way instead.
roads_0 = roads_0[~roads_0.EZIRDNMLBL.str.lower().str.contains("monash")]

# goals are all segments of roads beginning with args.letter
just = just_letters(roads_0, args.letter)
goals = list(get_edges(just.geometry))

# edges are the roads we can ride on, including some outside the postcode. in a
# postcode with concave bits this means you can cut across another postcode to
# make a shorter route.
edges = list(get_edges(roads_buffer.geometry))
edges = edges + goals  # truncation changes some edges at the boundary

# map of origin vertex to possible destination vertices
graph = build_graph(edges)

#

# home_ = (
#     (977497.1692172779, -4302403.591923077),
#     (977544.9539988617, -4302206.452253673)
# )
# home is the edge closest to the argued home latlon
home = min(edges, key=lambda e: length((midpoint(e), args.home)))


# any goal unreachable from all five witnesses is removed as entirely
# unreachable.
if len(goals) > 5:
    reach = {}
    witnesses = list(sample(goals, k=5))
    for goal in goals:
        reach[goal] = -1  # account for self
        for witness in witnesses:
            if nav(goal[0], witness[1]):
                reach[goal] += 1
    removable = [g for g, r in reach.items() if r < 1]
    print(f"Removing {len(removable)} goals because they are unreachable")
    goals = [g for g, r in reach.items() if r >= 1]
else:
    print("Not enough roads to reliably check reachability")


contigs = group_by_contiguity_2(goals)
goals_ = []
for contig in contigs:
    l = route_length(tuple(contig))
    if l < 20:
        print(
            f"Removing a goal road with {len(contig)} segments because the total length is only {l:0.2f}m"
        )
    else:
        goals_.extend(contig)
goals = goals_


if not goals:
    print(f"No roads begining with {args.letter}")
    sys.exit(1)


def plot_sol(sol: Solution, gen: int):
    km = round(score(sol) / 1000, 2)
    print(str(gen + 1).rjust(6), km)
    fig, ax = plt.subplots(dpi=100, figsize=(20, 20))
    roads_buffer.plot(ax=ax, color="#333333")
    postcode.buffer(500).difference(postcode).plot(
        ax=ax, color="#333333", alpha=0.65, lw=5, hatch="/"
    )
    plot_edges(ax, [home] + goals + [home], lw=8, color="white")
    plot_edges(ax, [home] + goals + [home], lw=5, color="black")
    route = solution_to_route(sol)
    if route:
        plot_edges(ax, route, offset=0, lw=2)
    ax.set(xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"“{args.letter}” gen {gen}: {km:0.2f}km", color="white")
    ax.set_facecolor("#000000")
    fig.set_facecolor("#000000")
    fig.set_tight_layout(True)
    fig.savefig(f"./tmp/{args.letter}/route-{gen:05d}.png")
    plt.close()


subprocess.run(f"mkdir -p  ./tmp/{args.letter}/", shell=True)
subprocess.run(f"rm ./tmp/{args.letter}/route*png", shell=True)


def solve(goals: Sequence[Edge], max_gens: int = 5000, plot: bool = True) -> Solution:
    sols: Sequence[Solution] = [goals]
    prev = best = sols[0]
    plot_rate = 0.0
    try:
        for gen in range(max_gens):
            best = sols[0]
            if score(best) < score(prev) and plot_rate < 5:
                plot_rate += gen ** 0.85
                if plot:
                    plot_sol(best, gen)
                best_route = solution_to_route(best)
                if not best_route:
                    break
                prev = best
            else:
                plot_rate = max(0, plot_rate - 1)
            sols = generation(sols)
        plot_sol(best, gen)
        return best
    except KeyboardInterrupt:
        print("")
        print("Quitting. Best result so far:")
        plot_sol(best, gen)
        return best


contigs_ = group_by_contiguity_2(goals)
contigs = []
max_contig_len = 10
for c in contigs_:
    while len(c) > max_contig_len:
        contigs.append(c[:max_contig_len])
        c = c[max_contig_len:]
    contigs.append(c)


standins = {}
for contig in contigs:
    mid = len(contig) // 2
    standins[contig[mid]] = contig
print(f"Pre-solving {len(standins)} standins of {len(goals)} goal roads for 200 gens")
pre_best = solve(tuple(standins.keys()), max_gens=200, plot=False)

goals = []
for standin in pre_best:
    if standin not in standins:
        standin = tuple(reversed(standin))
    goals.extend(standins[standin])


print(
    f"Solving {len(goals)} goal roads for 10,000 generations. "
    f"Ctrl-C to write best so far and quit."
)
best = solve(tuple(goals), max_gens=10000)
