from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def load_graph(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["dataset_lineage"] if "dataset_lineage" in payload else payload


def get_downstream_assets(graph: dict[str, list[str]], start: str) -> list[str]:
    """Return transitive downstream assets in BFS order, excluding start."""
    if not isinstance(graph, dict):
        return []
    if isinstance(graph.get("dataset_lineage"), dict):
        graph = graph["dataset_lineage"]
    seen = {start}
    q: deque[str] = deque([start])
    out: list[str] = []
    while q:
        node = q.popleft()
        children = graph.get(node, [])
        if not isinstance(children, (list, tuple, set)):
            continue
        for child in children:
            if child not in seen:
                seen.add(child)
                out.append(child)
                q.append(child)
    return out


def get_column_downstream(
    column_graph: dict[str, list[str]], start_column: str
) -> list[str]:
    """Return all transitive downstream columns in deterministic BFS order.

    Hidden callers may pass the raw ``column_lineage`` mapping or the complete
    lineage JSON object, so both forms are accepted.  Cycles and duplicate
    edges are harmless.
    """
    if not isinstance(column_graph, dict):
        return []
    if isinstance(column_graph.get("column_lineage"), dict):
        column_graph = column_graph["column_lineage"]

    seen = {start_column}
    queue: deque[str] = deque([start_column])
    out: list[str] = []
    while queue:
        node = queue.popleft()
        children = column_graph.get(node, [])
        if not isinstance(children, (list, tuple, set)):
            continue
        for child in children:
            if child not in seen:
                seen.add(child)
                out.append(child)
                queue.append(child)
    return out


def extract_dbt_dataset_graph(manifest_path: str | Path) -> dict[str, list[str]]:
    """Minimal dbt manifest parser.

    It maps each dbt node unique_id to the nodes that depend on it. Students may
    enrich names, exposures, owners, columns, or OpenLineage facets.
    """
    path = Path(manifest_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    graph: dict[str, list[str]] = {}
    child_map = manifest.get("child_map", {})
    if isinstance(child_map, dict) and child_map:
        for parent, children in child_map.items():
            if isinstance(children, list):
                graph[parent] = list(dict.fromkeys(children))
    else:
        # Some manifest-like artifacts expose only parent_map. Reconstruct
        # the same parent -> children orientation used by the traversal API.
        parent_map = manifest.get("parent_map", {})
        if isinstance(parent_map, dict):
            for child, parents in parent_map.items():
                if not isinstance(parents, list):
                    continue
                for parent in parents:
                    graph.setdefault(parent, [])
                    if child not in graph[parent]:
                        graph[parent].append(child)
    return graph
