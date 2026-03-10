"""
Semantic hierarchy: bridge detection and .ologic YAML emission.

This module takes clustered entities (from HAC / embed_and_cluster) and:
1. Detects bridge nodes between clusters via a Leiden-inspired community
   connectivity algorithm (pure-Python, zero new dependencies).
2. Emits valid .ologic YAML following the Ordinal ontology rules.

Implements:
    REQ-NKGL-005 — bridge detection (Leiden-inspired, pure-Python)
    REQ-NKGL-006 — .ologic bridge node emission
    REQ-NKGL-007 — valid .ologic YAML emission

Usage (simple — entity strings only):
    from nous import OllamaEmbedder
    from nous.semantic_cluster import embed_and_cluster
    from nous.hierarchy import BridgeDetector, OlogicEmitter

    embedder = OllamaEmbedder()
    cluster_result = await embed_and_cluster(entities, embedder)

    detector = BridgeDetector()
    bridges = detector.detect_from_cluster_result(cluster_result)

    emitter = OlogicEmitter()
    yaml_str = emitter.emit_from_cluster_result(cluster_result, bridges)

Usage (enriched — with requirements + node types):
    from nous.hierarchy import HierarchyEntity, build_ologic_enriched

    entities = [
        HierarchyEntity(id="e0", text="...", requirements=["REQ-NKGL-002"], node_type="ai"),
        ...
    ]
    bridges, yaml_str = build_ologic_enriched(entities, cluster_result)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from nous.vectors import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HierarchyEntity — enriched input type (extends plain entity strings with
# requirements tracing and node type annotations for the .ologic emitter)
# ---------------------------------------------------------------------------

REQ_PATTERN = re.compile(r"^REQ-[A-Z0-9]+-\d+$")


@dataclass
class HierarchyEntity:
    """
    An entity enriched with requirements tracing and node type for .ologic emission.

    When using embed_and_cluster() directly, entities are plain strings. Use
    HierarchyEntity when you need requirements traceability (REQ-NKGL-002) and
    semantic node type annotations in the emitted YAML.

    Attributes:
        id:           Unique identifier within the run (e.g. "req-0", "ent-auth").
        text:         The human-readable text (requirement description or entity name).
        embedding:    Float vector from OllamaEmbedder. Empty list if NullEmbedder.
        requirements: Requirement IDs this entity implements (e.g. ["REQ-NKGL-002"]).
        node_type:    Ordinal node type for the emitted YAML node (default: "process").
        metadata:     Arbitrary extra data (source file, line number, etc.)
    """

    id: str
    text: str
    embedding: list[float] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    node_type: str = "process"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityCluster:
    """
    A group of semantically related entities produced by HAC clustering.

    Attributes:
        id:                Unique string cluster identifier (e.g. "cluster-0").
        entities:          HierarchyEntity objects belonging to this cluster.
        centroid:          Mean embedding vector for the cluster. May be empty.
        label:             Human-readable label — most central entity text.
        intra_similarity:  Average pairwise cosine similarity within cluster [0,1].
    """

    id: str
    entities: list[HierarchyEntity]
    centroid: list[float] = field(default_factory=list)
    label: str = ""
    intra_similarity: float = 1.0

    def display_label(self) -> str:
        """Human-readable label: explicit label or first entity text truncated."""
        if self.label:
            return self.label
        if self.entities:
            return self.entities[0].text[:40]
        return self.id

    @property
    def machine_id(self) -> str:
        return f"machine-{self.id}"

    @property
    def factory_id(self) -> str:
        return f"factory-{self.id}"


@dataclass
class ClusterResult:
    """
    Full output of embed_and_cluster().

    This is the interface contract between embedder-and-cluster (Nu) and
    bridge-and-emit (Omicron). Both roles own this type — it lives here
    in hierarchy.py as the canonical definition.

    Attributes:
        clusters:              All clusters from HAC.
        inter_cluster_edges:   (cluster_id_a, cluster_id_b, similarity) tuples
                               representing centroid-to-centroid cosine similarity.
                               similarity is in [0, 1].
        model_name:            Embedding model used (for audit/logging).
    """

    clusters: list[EntityCluster]
    inter_cluster_edges: list[tuple[str, str, float]] = field(default_factory=list)
    model_name: str = "unknown"


# ---------------------------------------------------------------------------
# Bridge types
# ---------------------------------------------------------------------------


@dataclass
class Bridge:
    """
    A detected bridge between two clusters — becomes a gateway node in .ologic.

    Attributes:
        id:           Node ID in emitted YAML (always prefixed "bridge-").
        cluster_a_id: Source cluster (string, e.g. "0" from cluster_id int).
        cluster_b_id: Target cluster.
        similarity:   Inter-cluster similarity score [0, 1].
        requirements: Requirement IDs this bridge implements (REQ-NKGL-006).
        title:        Human-readable bridge label.
        is_cut_edge:  True if this edge is a Tarjan bridge (cut edge).
    """

    id: str
    cluster_a_id: str
    cluster_b_id: str
    similarity: float
    requirements: list[str] = field(default_factory=lambda: ["REQ-NKGL-006"])
    title: str = ""
    is_cut_edge: bool = False

    def __post_init__(self) -> None:
        if not self.title:
            self.title = f"Bridge: cluster-{self.cluster_a_id} → cluster-{self.cluster_b_id}"


# ---------------------------------------------------------------------------
# Inter-cluster edge derivation from ClusterResult
# ---------------------------------------------------------------------------


def _derive_inter_cluster_edges(
    cluster_result: ClusterResult,
    similarity_threshold: float = 0.0,
) -> list[tuple[str, str, float]]:
    """
    Derive inter-cluster similarity edges from a ClusterResult.

    If inter_cluster_edges is already populated (Nu computed them), use those.
    Otherwise, fall back to computing centroid-to-centroid cosine similarity.

    Returns:
        List of (cluster_id_a, cluster_id_b, similarity) tuples.
    """
    # If Nu already computed edges, use them directly
    if cluster_result.inter_cluster_edges:
        return [
            (a, b, sim)
            for a, b, sim in cluster_result.inter_cluster_edges
            if sim >= similarity_threshold
        ]

    # Fallback: compute from centroids
    clusters = cluster_result.clusters
    if len(clusters) < 2:
        return []

    edges = []
    seen: set[tuple[str, str]] = set()

    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            if i >= j:
                continue
            key = (min(ca.id, cb.id), max(ca.id, cb.id))
            if key in seen:
                continue
            seen.add(key)

            if ca.centroid and cb.centroid:
                sim = cosine_similarity(ca.centroid, cb.centroid)
                sim = max(0.0, min(1.0, sim))
            else:
                sim = 0.5

            if sim >= similarity_threshold:
                edges.append((ca.id, cb.id, sim))

    return edges


# ---------------------------------------------------------------------------
# Bridge detection — Leiden-inspired
# ---------------------------------------------------------------------------


class BridgeDetector:
    """
    Detects bridge nodes between clusters using a Leiden-inspired algorithm.

    The full Leiden algorithm requires igraph/leidenalg. Since Nous has zero
    new dependencies, we implement the connectivity-critical edge detection
    step in pure Python:

    1. Build a weighted graph from inter-cluster centroid similarities
       (threshold-filtered).
    2. Find edges whose removal disconnects the graph (bridge / cut edges)
       via DFS-based bridge detection (Tarjan's bridge algorithm).
    3. All edges above threshold are candidates; bridges (cut edges) get
       elevated importance. Non-bridge edges also generate gateway nodes since
       they represent meaningful semantic overlap.

    The result is a list of Bridge objects — one per significant inter-cluster
    connection. The emitter turns each Bridge into a standalone gateway node.

    Args:
        similarity_threshold: Minimum inter-cluster similarity to consider as a
                              bridge candidate. Default 0.35.
        max_bridges:          Cap on total bridges to avoid overwhelming diagrams.
                              Default 20.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.35,
        max_bridges: int = 20,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_bridges = max_bridges

    def detect_from_cluster_result(self, cluster_result: ClusterResult) -> list[Bridge]:
        """
        Detect bridges directly from a ClusterResult.

        Uses pre-computed inter_cluster_edges if available, otherwise derives
        them from centroid cosine similarities. Then applies Tarjan's bridge
        detection algorithm.

        Returns:
            List of Bridge objects, sorted by similarity descending.
        """
        edges = _derive_inter_cluster_edges(
            cluster_result,
            similarity_threshold=self.similarity_threshold,
        )
        cluster_labels = {c.id: c.display_label() for c in cluster_result.clusters}
        return self._build_bridges(edges, cluster_labels)

    def detect(
        self,
        inter_cluster_edges: list[tuple[str, str, float]],
        cluster_labels: dict[str, str] | None = None,
    ) -> list[Bridge]:
        """
        Detect bridges from explicit inter-cluster edge list.

        This is the lower-level API — use when you've pre-computed edge weights
        (e.g. from a custom similarity measure).

        Args:
            inter_cluster_edges: List of (cluster_id_a, cluster_id_b, similarity) tuples.
            cluster_labels:      Optional mapping from cluster_id_str → human label.

        Returns:
            List of Bridge objects, sorted by similarity descending.
        """
        candidate_edges = [
            (a, b, sim)
            for a, b, sim in inter_cluster_edges
            if sim >= self.similarity_threshold
        ]
        return self._build_bridges(candidate_edges, cluster_labels or {})

    def _build_bridges(
        self,
        candidate_edges: list[tuple[str, str, float]],
        cluster_labels: dict[str, str],
    ) -> list[Bridge]:
        """Core bridge detection: Tarjan + Bridge construction."""
        if not candidate_edges:
            logger.debug("BridgeDetector: no edges above threshold %.2f", self.similarity_threshold)
            return []

        # Build adjacency for Tarjan's bridge detection
        node_ids = set()
        for a, b, _ in candidate_edges:
            node_ids.add(a)
            node_ids.add(b)

        adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in node_ids}
        for a, b, sim in candidate_edges:
            adj[a].append((b, sim))
            adj[b].append((a, sim))

        # Detect cut edges (bridges) via Tarjan's algorithm
        cut_edges = self._tarjan_bridges(adj)
        cut_edge_set = {(min(a, b), max(a, b)) for a, b in cut_edges}

        # Sort by: cut-edge first, then similarity descending
        def edge_key(e: tuple[str, str, float]) -> tuple[int, float]:
            a, b, sim = e
            is_cut = (min(a, b), max(a, b)) in cut_edge_set
            return (0 if is_cut else 1, -sim)

        sorted_edges = sorted(candidate_edges, key=edge_key)

        bridges: list[Bridge] = []
        seen: set[tuple[str, str]] = set()

        for a, b, sim in sorted_edges:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)

            bridge_id = f"bridge-{a}-{b}"
            label_a = cluster_labels.get(a, f"cluster-{a}")
            label_b = cluster_labels.get(b, f"cluster-{b}")
            # Truncate for readability
            la = (label_a[:22] + "...") if len(label_a) > 25 else label_a
            lb = (label_b[:22] + "...") if len(label_b) > 25 else label_b
            title = f"Bridge: {la} → {lb}"
            is_cut = (min(a, b), max(a, b)) in cut_edge_set

            bridges.append(
                Bridge(
                    id=bridge_id,
                    cluster_a_id=a,
                    cluster_b_id=b,
                    similarity=sim,
                    requirements=["REQ-NKGL-006"],
                    title=title,
                    is_cut_edge=is_cut,
                )
            )

            if len(bridges) >= self.max_bridges:
                break

        logger.info(
            "BridgeDetector: detected %d bridges (%d cut-edges) from %d candidates",
            len(bridges),
            len(cut_edges),
            len(candidate_edges),
        )
        return bridges

    # ------------------------------------------------------------------
    # Tarjan's bridge-finding algorithm (pure Python)
    # ------------------------------------------------------------------

    def _tarjan_bridges(
        self, adj: dict[str, list[tuple[str, float]]]
    ) -> list[tuple[str, str]]:
        """
        Find all bridge edges (cut edges) in an undirected graph via Tarjan's algorithm.

        A bridge is an edge whose removal disconnects the graph (or a component).
        These are the most semantically critical inter-cluster connections.

        Returns:
            List of (node_a, node_b) tuples for each bridge edge.
        """
        disc: dict[str, int] = {}
        low: dict[str, int] = {}
        visited: set[str] = set()
        bridges: list[tuple[str, str]] = []
        timer = [0]

        def dfs(u: str, parent: str | None) -> None:
            visited.add(u)
            disc[u] = low[u] = timer[0]
            timer[0] += 1

            for v, _ in adj.get(u, []):
                if v not in visited:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    # Bridge condition: low[v] > disc[u]
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent:
                    low[u] = min(low[u], disc[v])

        for node in adj:
            if node not in visited:
                dfs(node, None)

        return bridges


# ---------------------------------------------------------------------------
# .ologic YAML emitter
# ---------------------------------------------------------------------------

# Valid Ordinal node types (diagramming mode)
VALID_NODE_TYPES = frozenset(
    [
        "static", "source", "decision", "ai", "server", "database", "api",
        "cloud", "container", "queue", "cache", "gateway", "firewall",
        "loadbalancer", "user", "monitor", "input", "output", "process",
        "worker", "orchestrator", "oracle",
    ]
)

_DEFAULT_NODE_TYPE = "process"


class OlogicEmitter:
    """
    Emits valid .ologic YAML from a ClusterResult and detected bridges.

    Follows the Ordinal ontology rules exactly:
    - Nodes connected via outputs: within a cluster form 1 machine per cluster
    - Bridge nodes are standalone at factory level (inputs+outputs → machines)
    - Structure depth is determined by bridge presence:

      | Clusters | Bridges | Output root  |
      |----------|---------|--------------|
      | ≤1       | 0       | machines:    |
      | 2+       | 0       | factories:   |
      | 2+       | 1+      | networks:    |

    The emitted YAML can be parsed and rendered by any .ologic-compatible tool
    (Shellworks, Thoughtorio, etc.)

    Args:
        network_id:    Top-level network node ID. Default "semantic-hierarchy".
        indent:        YAML indentation spaces. Default 2.
        max_title_len: Maximum length for node titles. Default 60.
    """

    def __init__(
        self,
        network_id: str = "semantic-hierarchy",
        indent: int = 2,
        max_title_len: int = 60,
    ) -> None:
        self.network_id = network_id
        self.indent = indent
        self.max_title_len = max_title_len

    def emit_from_cluster_result(
        self,
        cluster_result: ClusterResult,
        bridges: list[Bridge],
    ) -> str:
        """
        Emit .ologic YAML from a ClusterResult (canonical entry point).

        Args:
            cluster_result: Output from embed_and_cluster().
            bridges:        Detected bridges from BridgeDetector.

        Returns:
            YAML string — valid .ologic format.
        """
        return self._emit_core(
            clusters=cluster_result.clusters,
            bridges=bridges,
        )

    def emit(
        self,
        clusters: list[EntityCluster],
        bridges: list[Bridge],
    ) -> str:
        """
        Emit .ologic YAML from a list of EntityCluster objects and bridges.

        Lower-level API for when you construct clusters manually.

        Args:
            clusters: List of EntityCluster objects.
            bridges:  Detected bridges.

        Returns:
            YAML string — valid .ologic format.
        """
        return self._emit_core(clusters=clusters, bridges=bridges)

    # ------------------------------------------------------------------
    # Core emission logic
    # ------------------------------------------------------------------

    def _emit_core(
        self,
        clusters: list[EntityCluster],
        bridges: list[Bridge],
    ) -> str:
        n_clusters = len(clusters)
        n_bridges = len(bridges)

        lines: list[str] = []
        lines.append("logic:")
        lines.append(f"{self._i(1)}mode: diagramming")
        lines.append(f"{self._i(1)}version: '2.0'")

        if n_clusters == 0:
            lines.append(f"{self._i(1)}machines: []")
            return "\n".join(lines) + "\n"

        if n_clusters == 1 and n_bridges == 0:
            lines.append(f"{self._i(1)}machines:")
            self._emit_machine(lines, clusters[0], depth=2)

        elif n_bridges == 0:
            lines.append(f"{self._i(1)}factories:")
            for cluster in clusters:
                self._emit_factory_no_bridges(lines, cluster, depth=2)

        else:
            lines.append(f"{self._i(1)}networks:")
            self._emit_network(lines, clusters, bridges, depth=2)

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Structure-level emitters
    # ------------------------------------------------------------------

    def _machine_id(self, cluster: EntityCluster) -> str:
        return f"machine-{cluster.id}"

    def _factory_id(self, cluster: EntityCluster) -> str:
        return f"factory-{cluster.id}"

    def _emit_machine(
        self,
        lines: list[str],
        cluster: EntityCluster,
        depth: int,
    ) -> None:
        lines.append(f"{self._i(depth)}- id: {self._machine_id(cluster)}")
        if cluster.label:
            lines.append(f"{self._i(depth+1)}title: {self._safe_str(cluster.label)}")
        lines.append(f"{self._i(depth+1)}nodes:")
        self._emit_entity_nodes(lines, cluster.entities, depth + 2)

    def _emit_factory_no_bridges(
        self,
        lines: list[str],
        cluster: EntityCluster,
        depth: int,
    ) -> None:
        lines.append(f"{self._i(depth)}- id: {self._factory_id(cluster)}")
        if cluster.label:
            lines.append(f"{self._i(depth+1)}title: {self._safe_str(cluster.label)}")
        lines.append(f"{self._i(depth+1)}machines:")
        self._emit_machine(lines, cluster, depth + 2)

    def _emit_network(
        self,
        lines: list[str],
        clusters: list[EntityCluster],
        bridges: list[Bridge],
        depth: int,
    ) -> None:
        lines.append(f"{self._i(depth)}- id: {self.network_id}")
        lines.append(f"{self._i(depth+1)}title: Semantic Hierarchy")
        lines.append(f"{self._i(depth+1)}requirements: [REQ-NKGL-007]")

        # Group bridges by cluster involvement
        bridges_by_cluster: dict[str, list[Bridge]] = {}
        for b in bridges:
            bridges_by_cluster.setdefault(b.cluster_a_id, []).append(b)
            bridges_by_cluster.setdefault(b.cluster_b_id, []).append(b)

        # One factory per cluster
        lines.append(f"{self._i(depth+1)}factories:")
        for cluster in clusters:
            self._emit_factory_with_bridges(
                lines,
                cluster,
                bridges_by_cluster.get(cluster.id, []),
                depth + 2,
            )

        # Network-level standalone nodes (cross-factory bridge gateways)
        # Per Ordinal rules: node at network level with inputs:[factory-id] →
        # triggers network detection.
        net_bridge_nodes = self._build_network_bridge_nodes(clusters, bridges)
        if net_bridge_nodes:
            lines.append(f"{self._i(depth+1)}nodes:")
            for node in net_bridge_nodes:
                self._emit_node_dict(lines, node, depth + 2)

    def _emit_factory_with_bridges(
        self,
        lines: list[str],
        cluster: EntityCluster,
        bridges: list[Bridge],
        depth: int,
    ) -> None:
        """
        Emit a factory for a cluster, with bridge gateway nodes at the factory level.

        Bridge nodes are standalone siblings to machines:, NOT inside the machine.
        This is what the Ordinal ontology requires — a node at factory level with
        inputs: [machine-id] is what triggers factory detection.
        """
        lines.append(f"{self._i(depth)}- id: {self._factory_id(cluster)}")
        lines.append(f"{self._i(depth+1)}title: {self._safe_str(cluster.display_label())}")
        lines.append(f"{self._i(depth+1)}requirements: [REQ-NKGL-005]")

        lines.append(f"{self._i(depth+1)}machines:")
        self._emit_machine(lines, cluster, depth + 2)

        # Bridge gateway nodes at factory level (sibling to machines:)
        factory_bridge_nodes = self._build_factory_bridge_nodes(cluster, bridges)
        if factory_bridge_nodes:
            lines.append(f"{self._i(depth+1)}nodes:")
            for node in factory_bridge_nodes:
                self._emit_node_dict(lines, node, depth + 2)

    # ------------------------------------------------------------------
    # Node-level emitters
    # ------------------------------------------------------------------

    def _emit_entity_nodes(
        self,
        lines: list[str],
        entities: list[HierarchyEntity],
        depth: int,
    ) -> None:
        """
        Emit entity nodes, chained via outputs: to form one machine (connected component).

        Entity[0] → Entity[1] → ... → Entity[N-1]: sequential chain ensures all
        nodes are in one connected component (one machine per cluster).
        """
        for i, entity in enumerate(entities):
            node_type = (
                entity.node_type
                if entity.node_type in VALID_NODE_TYPES
                else _DEFAULT_NODE_TYPE
            )
            node_id = f"entity-{entity.id}"
            title = self._truncate(entity.text, self.max_title_len)
            requirements = entity.requirements

            lines.append(f"{self._i(depth)}- id: {node_id}")
            lines.append(f"{self._i(depth+1)}type: {node_type}")
            lines.append(f"{self._i(depth+1)}title: {self._safe_str(title)}")

            if requirements:
                lines.append(
                    f"{self._i(depth+1)}requirements: {self._yaml_list(requirements)}"
                )

            # Chain to next entity (creates machine via Union-Find connectivity)
            if i < len(entities) - 1:
                next_entity = entities[i + 1]
                lines.append(f"{self._i(depth+1)}outputs: [entity-{next_entity.id}]")

    def _emit_node_dict(
        self,
        lines: list[str],
        node: dict[str, Any],
        depth: int,
    ) -> None:
        lines.append(f"{self._i(depth)}- id: {node['id']}")
        lines.append(f"{self._i(depth+1)}type: {node['type']}")
        lines.append(f"{self._i(depth+1)}title: {self._safe_str(node['title'])}")
        if node.get("inputs"):
            lines.append(f"{self._i(depth+1)}inputs: {self._yaml_list(node['inputs'])}")
        if node.get("outputs"):
            lines.append(f"{self._i(depth+1)}outputs: {self._yaml_list(node['outputs'])}")
        if node.get("requirements"):
            lines.append(
                f"{self._i(depth+1)}requirements: {self._yaml_list(node['requirements'])}"
            )

    # ------------------------------------------------------------------
    # Bridge node builders
    # ------------------------------------------------------------------

    def _build_factory_bridge_nodes(
        self,
        cluster: EntityCluster,
        bridges: list[Bridge],
    ) -> list[dict[str, Any]]:
        """
        Build gateway node dicts for bridges where this cluster is the source (cluster_a).

        These sit at the factory level — standalone, NOT inside any machine.
        Per Ordinal rules: inputs: [machine-X] on a standalone factory-level node
        creates a cross-machine connection → triggers factory detection.
        """
        nodes = []
        for bridge in bridges:
            if bridge.cluster_a_id != cluster.id:
                continue
            source_machine = self._machine_id(cluster)
            target_machine = f"machine-{bridge.cluster_b_id}"
            nodes.append({
                "id": bridge.id,
                "type": "gateway",
                "title": bridge.title,
                "inputs": [source_machine],
                "outputs": [target_machine],
                "requirements": bridge.requirements,
            })
        return nodes

    def _build_network_bridge_nodes(
        self,
        clusters: list[EntityCluster],
        bridges: list[Bridge],
    ) -> list[dict[str, Any]]:
        """
        Build network-level gateway nodes for cross-factory connections.

        Per Ordinal rules: node at network level with inputs:[factory-id] triggers
        network detection. One network-level gateway per bridge.
        """
        cluster_id_strs = {c.id for c in clusters}
        nodes = []
        for bridge in bridges:
            if bridge.cluster_a_id not in cluster_id_strs:
                continue
            if bridge.cluster_b_id not in cluster_id_strs:
                continue
            factory_a = f"factory-{bridge.cluster_a_id}"
            factory_b = f"factory-{bridge.cluster_b_id}"
            nodes.append({
                "id": f"net-{bridge.id}",
                "type": "gateway",
                "title": f"Net: {bridge.title}",
                "inputs": [factory_a],
                "outputs": [factory_b],
                "requirements": ["REQ-NKGL-007"],
            })
        return nodes

    # ------------------------------------------------------------------
    # YAML formatting helpers
    # ------------------------------------------------------------------

    def _i(self, depth: int) -> str:
        return " " * (self.indent * depth)

    def _safe_str(self, value: str) -> str:
        """Wrap in quotes if the value contains YAML-special characters."""
        if not value:
            return '""'
        needs_quotes = any(c in value for c in (":", "#", "'", "[", "]", "{", "}", "&", "*"))
        if needs_quotes:
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        return value

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def _yaml_list(self, items: list[str]) -> str:
        return "[" + ", ".join(str(i) for i in items) + "]"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def build_ologic_from_cluster_result(
    cluster_result: ClusterResult,
    similarity_threshold: float = 0.35,
    max_bridges: int = 20,
    network_id: str = "semantic-hierarchy",
) -> tuple[list[Bridge], str]:
    """
    Convenience wrapper: detect bridges + emit .ologic YAML from a ClusterResult.

    This is the main entry point for the full pipeline:
        embed_and_cluster() → build_ologic_from_cluster_result()

    Args:
        cluster_result:       Output from embed_and_cluster() — ClusterResult with
                              EntityCluster objects containing HierarchyEntity lists.
        similarity_threshold: Minimum inter-cluster similarity to emit a bridge.
        max_bridges:          Maximum number of bridge gateway nodes in the YAML.
        network_id:           Top-level network node ID in emitted YAML.

    Returns:
        Tuple of (bridges, yaml_string).

    Example:
        from nous import OllamaEmbedder
        from nous.hierarchy import HierarchyEntity, build_ologic_from_cluster_result
        from nous.semantic_cluster import embed_and_cluster

        embedder = OllamaEmbedder()
        entities = [
            HierarchyEntity(id="e0", text="auth", requirements=["REQ-NKGL-002"]),
            HierarchyEntity(id="e1", text="storage", requirements=["REQ-NKGL-002"]),
        ]
        result = await embed_and_cluster(entities, embedder)
        bridges, yaml_str = build_ologic_from_cluster_result(result)
        print(yaml_str)
    """
    detector = BridgeDetector(
        similarity_threshold=similarity_threshold,
        max_bridges=max_bridges,
    )
    bridges = detector.detect_from_cluster_result(cluster_result)

    emitter = OlogicEmitter(network_id=network_id)
    yaml_str = emitter.emit_from_cluster_result(cluster_result, bridges)

    return bridges, yaml_str


# Alias for convenience
build_ologic = build_ologic_from_cluster_result
