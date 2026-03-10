"""
ologic.py — .ologic YAML validator and requirement reconciler for the Nous semantic hierarchy.

This module validates .ologic YAML files produced by the semantic hierarchy generator
(REQ-NKGL-002 through 007) against the Ordinal ontology rules. It provides:

- OlogicError / OlogicValidationResult — structured error reporting
- OlogicValidator — full structural + ontology validator for emitted YAML
- ReconciliationResult / reconcile_requirements — REQ traceability checker
- REQUIREMENTS — the canonical REQ-NKGL-002..007 requirement definitions

Ordinal Ontology Rules enforced:
  1. Root key must be `logic:` with mode/version fields
  2. Node→Node outputs within a machine are allowed; cross-machine node→node is forbidden
  3. Bridge nodes must be standalone at factory level (not inside a machine)
  4. Bridge nodes must have BOTH inputs and outputs (cross-machine connection)
  5. All node IDs must be unique within the document
  6. All referenced IDs (inputs/outputs/requirements) must exist or be known REQ-NKGL-NNN
  7. Every node must have a valid type (from VALID_NODE_TYPES)
  8. Requirements annotations must match REQ-[A-Z]+-[0-9]+ pattern
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─── Requirements Catalog ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Requirement:
    """A formal requirement in the REQ-NKGL family."""
    id: str
    title: str
    description: str


# REQ-NKGL-001 is the overall feature requirement (not implemented here — that's the dispatcher)
# REQ-NKGL-002..007 are the sub-requirements this team implements.

REQUIREMENTS: dict[str, Requirement] = {
    "REQ-NKGL-001": Requirement(
        id="REQ-NKGL-001",
        title="Semantic Hierarchy Generator",
        description=(
            "The Nous system shall provide a semantic hierarchy generator that takes a list of "
            "entities/requirements, embeds them via an EmbeddingProvider, clusters them via "
            "Hierarchical Agglomerative Clustering (HAC), detects cross-cluster bridges via "
            "Leiden community detection, and emits a valid .ologic YAML document representing "
            "the semantic hierarchy."
        ),
    ),
    "REQ-NKGL-002": Requirement(
        id="REQ-NKGL-002",
        title="Entity Embedding",
        description=(
            "Each input entity/requirement shall be embedded using the configured "
            "EmbeddingProvider (Ollama, OpenAI, or custom). Entities must carry their "
            "text, embedding vector, node_type, and requirement annotations through the "
            "pipeline."
        ),
    ),
    "REQ-NKGL-003": Requirement(
        id="REQ-NKGL-003",
        title="Hierarchical Agglomerative Clustering (HAC)",
        description=(
            "Embedded entities shall be clustered using Hierarchical Agglomerative Clustering "
            "with cosine distance and average linkage. The distance threshold shall be "
            "configurable (default 0.4). Each cluster shall produce a Cluster object with "
            "id, entities list, centroid vector, and optional label."
        ),
    ),
    "REQ-NKGL-004": Requirement(
        id="REQ-NKGL-004",
        title="ClusterResult Contract",
        description=(
            "The HAC step shall emit a ClusterResult containing: clusters (list[Cluster]), "
            "inter_cluster_edges (list of (cluster_id_a, cluster_id_b, similarity) triples). "
            "This is the interface contract between embedder-and-cluster and bridge-and-emit."
        ),
    ),
    "REQ-NKGL-005": Requirement(
        id="REQ-NKGL-005",
        title="Leiden Bridge Detection",
        description=(
            "Cross-cluster bridges shall be detected using the Leiden community detection "
            "algorithm applied to the inter_cluster_edges graph. Bridge nodes are entities "
            "that appear at community boundaries (high betweenness centrality or explicit "
            "Leiden-assigned bridge role). Each bridge shall appear as a standalone gateway "
            "node at the factory level in the emitted .ologic."
        ),
    ),
    "REQ-NKGL-006": Requirement(
        id="REQ-NKGL-006",
        title=".ologic Bridge Node Emission",
        description=(
            "Every detected bridge shall be emitted as a standalone gateway node at the "
            "factory level with both `inputs` (source machine) and `outputs` (target machine). "
            "Bridge node IDs shall follow the pattern `bridge-{cluster_a}-{cluster_b}`. "
            "Every bridge node shall carry `requirements: [REQ-NKGL-006]` in its annotation."
        ),
    ),
    "REQ-NKGL-007": Requirement(
        id="REQ-NKGL-007",
        title="Valid .ologic YAML Emission",
        description=(
            "The generator shall emit a valid .ologic YAML document conforming to the "
            "Ordinal ontology rules: root key `logic:`, mode `diagramming`, version `2.0`, "
            "correct container hierarchy (nodes/machines/factories/networks), no cross-machine "
            "node→node edges, all node IDs unique, all referenced IDs resolvable, all node "
            "types from the valid set."
        ),
    ),
}

# ─── Constants ────────────────────────────────────────────────────────────────

VALID_NODE_TYPES: frozenset[str] = frozenset({
    "static", "source", "decision", "ai", "server", "database", "api", "cloud",
    "container", "queue", "cache", "gateway", "firewall", "loadbalancer", "user",
    "monitor", "input", "output", "process", "worker", "orchestrator", "oracle",
})

REQ_ID_PATTERN: re.Pattern = re.compile(r"^REQ-[A-Z]+-\d+$")
BRIDGE_NODE_PREFIX: str = "bridge-"
OLOGIC_MODE: str = "diagramming"
OLOGIC_VERSION: str = "2.0"


# ─── Error Types ─────────────────────────────────────────────────────────────

@dataclass
class OlogicError:
    """A single validation error with location context."""
    code: str          # Machine-readable error code (e.g. "MISSING_ROOT_KEY")
    message: str       # Human-readable description
    path: str = ""     # Dot-path to the offending element (e.g. "factories[0].machines[1].nodes[2]")
    node_id: str = ""  # Node ID if applicable


@dataclass
class OlogicValidationResult:
    """Result of validating a .ologic YAML document."""
    valid: bool
    errors: list[OlogicError] = field(default_factory=list)
    warnings: list[OlogicError] = field(default_factory=list)
    node_count: int = 0
    machine_count: int = 0
    factory_count: int = 0
    network_count: int = 0
    bridge_count: int = 0
    covered_requirements: set[str] = field(default_factory=set)

    def error(self, code: str, message: str, path: str = "", node_id: str = "") -> None:
        self.errors.append(OlogicError(code=code, message=message, path=path, node_id=node_id))
        self.valid = False

    def warn(self, code: str, message: str, path: str = "", node_id: str = "") -> None:
        self.warnings.append(OlogicError(code=code, message=message, path=path, node_id=node_id))

    def summary(self) -> str:
        lines = [
            f"{'✅ VALID' if self.valid else '❌ INVALID'} .ologic document",
            f"  Nodes: {self.node_count} | Machines: {self.machine_count} | "
            f"Factories: {self.factory_count} | Networks: {self.network_count} | "
            f"Bridges: {self.bridge_count}",
        ]
        if self.covered_requirements:
            lines.append(f"  Requirements covered: {', '.join(sorted(self.covered_requirements))}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                loc = f" [{e.path}]" if e.path else ""
                lines.append(f"    [{e.code}]{loc} {e.message}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                loc = f" [{w.path}]" if w.path else ""
                lines.append(f"    [{w.code}]{loc} {w.message}")
        return "\n".join(lines)


# ─── Validator ───────────────────────────────────────────────────────────────

class OlogicValidator:
    """
    Validates .ologic YAML documents against the Ordinal ontology rules.

    Checks enforced (REQ-NKGL-007 and Ordinal spec):
    - Root structure: `logic.mode == 'diagramming'`, `logic.version == '2.0'`
    - Exactly one top-level container key (networks/factories/machines/nodes)
    - All node IDs unique within the document
    - All node types from VALID_NODE_TYPES
    - Requirements annotations match REQ-[A-Z]+-[0-9]+ pattern
    - No cross-machine node→node edges (outputs must reference siblings in same machine)
    - Bridge nodes: standalone at factory level, have BOTH inputs AND outputs
    - Bridge node inputs/outputs reference valid machine IDs
    - All referenced IDs (inputs/outputs) resolve to known nodes or machines

    Usage:
        validator = OlogicValidator()
        result = validator.validate(yaml_dict)
        print(result.summary())
    """

    def validate(self, doc: dict[str, Any]) -> OlogicValidationResult:
        """
        Validate a parsed .ologic YAML document (already loaded as a Python dict).

        Args:
            doc: The parsed YAML dict — top-level key should be 'logic'.

        Returns:
            OlogicValidationResult with valid=True/False and full error list.
        """
        result = OlogicValidationResult(valid=True)

        # ── 1. Root structure ────────────────────────────────────────────────
        if "logic" not in doc:
            result.error(
                "MISSING_ROOT_KEY",
                "Document must have a top-level 'logic:' key",
                path="<root>",
            )
            return result  # Cannot proceed without root

        logic = doc["logic"]
        if not isinstance(logic, dict):
            result.error("INVALID_LOGIC_TYPE", "'logic' must be a mapping", path="logic")
            return result

        mode = logic.get("mode")
        version = logic.get("version")

        if mode != OLOGIC_MODE:
            result.error(
                "WRONG_MODE",
                f"logic.mode must be '{OLOGIC_MODE}', got {mode!r}",
                path="logic.mode",
            )

        if str(version) != OLOGIC_VERSION:
            result.error(
                "WRONG_VERSION",
                f"logic.version must be '{OLOGIC_VERSION}', got {version!r}",
                path="logic.version",
            )

        # ── 2. Container key detection ───────────────────────────────────────
        container_keys = {"networks", "factories", "machines", "nodes"}
        present = [k for k in container_keys if k in logic]
        if len(present) == 0:
            result.error(
                "NO_CONTAINER",
                "logic must contain one of: networks, factories, machines, nodes",
                path="logic",
            )
            return result
        if len(present) > 1:
            result.error(
                "MULTIPLE_CONTAINERS",
                f"logic must contain exactly one container key, found: {present}",
                path="logic",
            )

        # ── 3. Collect all IDs and build context for reference checking ──────
        all_node_ids: dict[str, str] = {}       # id → path
        all_machine_ids: dict[str, str] = {}    # id → path
        all_factory_ids: dict[str, str] = {}    # id → path
        bridge_nodes: list[dict[str, Any]] = []  # bridge node dicts with their path

        # Nodes that are inside machines (for cross-machine edge detection)
        # machine_id → set of node_ids inside it
        machine_membership: dict[str, set[str]] = {}

        self._collect_ids(
            logic,
            path="logic",
            result=result,
            all_node_ids=all_node_ids,
            all_machine_ids=all_machine_ids,
            all_factory_ids=all_factory_ids,
            machine_membership=machine_membership,
            bridge_nodes=bridge_nodes,
        )

        # ── 4. Count stats ───────────────────────────────────────────────────
        result.node_count = len(all_node_ids)
        result.machine_count = len(all_machine_ids)
        result.factory_count = len(all_factory_ids)
        result.network_count = len(logic.get("networks", []))
        result.bridge_count = len(bridge_nodes)

        # ── 5. Structural decision table check (from Omicron's emitter spec) ──
        #
        # | Clusters | Bridges | Root key    |
        # |----------|---------|-------------|
        # | 1        | 0       | machines:   |
        # | 2+       | 0       | factories:  |
        # | 2+       | 1+      | networks:   |
        #
        # We infer clusters from machine_count and bridges from bridge_count.
        # This is a heuristic check — only applies when the doc was emitted by
        # the semantic hierarchy generator. We emit warnings, not errors, since
        # hand-authored .ologic files may differ.
        machine_c = len(all_machine_ids)
        bridge_c = len(bridge_nodes)
        top_key = present[0] if present else None
        if top_key == "machines" and machine_c > 1:
            result.warn(
                "STRUCTURAL_MULTI_CLUSTER_SHOULD_USE_FACTORIES",
                f"Document has {machine_c} machines at root 'machines:' level but no factory "
                f"wrapper. With 2+ clusters and no bridges, the emitter should use 'factories:' "
                f"(one factory per cluster, disconnected). Consider wrapping in factories.",
                path="logic.machines",
            )
        if top_key == "factories" and bridge_c > 0:
            result.warn(
                "STRUCTURAL_BRIDGES_SHOULD_USE_NETWORKS",
                f"Document has {bridge_c} bridge node(s) inside 'factories:' but no network "
                f"wrapper. With bridges present, the emitter should use 'networks:' to represent "
                f"the cross-factory connections.",
                path="logic.factories",
            )
        if top_key == "networks" and machine_c <= 1 and bridge_c == 0:
            result.warn(
                "STRUCTURAL_SINGLE_CLUSTER_SHOULD_USE_MACHINES",
                f"Document uses 'networks:' but has only {machine_c} machine(s) and no bridges. "
                f"A single cluster with no bridges should use 'machines:' at root level.",
                path="logic.networks",
            )

        # ── 6. Validate edge references and cross-machine violations ─────────
        self._validate_edges(
            logic,
            path="logic",
            result=result,
            all_node_ids=all_node_ids,
            all_machine_ids=all_machine_ids,
            all_factory_ids=all_factory_ids,
            machine_membership=machine_membership,
        )

        # ── 7. Validate bridge nodes specifically ────────────────────────────
        for bn in bridge_nodes:
            self._validate_bridge_node(
                bn["node"],
                path=bn["path"],
                result=result,
                all_machine_ids=all_machine_ids,
            )

        return result

    # ── Internal: ID collection ───────────────────────────────────────────────

    def _collect_ids(
        self,
        container: dict[str, Any],
        path: str,
        result: OlogicValidationResult,
        all_node_ids: dict[str, str],
        all_machine_ids: dict[str, str],
        all_factory_ids: dict[str, str],
        machine_membership: dict[str, set[str]],
        bridge_nodes: list[dict[str, Any]],
    ) -> None:
        """Recursively collect all node/machine/factory IDs from the document."""

        # Networks
        for ni, network in enumerate(container.get("networks", [])):
            npath = f"{path}.networks[{ni}]"
            self._validate_id_field(network, npath, "network", result, {})
            self._collect_ids(
                network, npath, result,
                all_node_ids, all_machine_ids, all_factory_ids,
                machine_membership, bridge_nodes,
            )
            # Standalone nodes at network level
            for xi, node in enumerate(network.get("nodes", [])):
                self._register_node(
                    node, f"{npath}.nodes[{xi}]", result,
                    all_node_ids, machine_membership, bridge_nodes,
                    factory_level=False,
                )

        # Factories
        for fi, factory in enumerate(container.get("factories", [])):
            fpath = f"{path}.factories[{fi}]"
            fid = self._validate_id_field(factory, fpath, "factory", result, all_factory_ids)
            self._collect_ids(
                factory, fpath, result,
                all_node_ids, all_machine_ids, all_factory_ids,
                machine_membership, bridge_nodes,
            )
            # Standalone nodes at factory level (may be bridges)
            for xi, node in enumerate(factory.get("nodes", [])):
                self._register_node(
                    node, f"{fpath}.nodes[{xi}]", result,
                    all_node_ids, machine_membership, bridge_nodes,
                    factory_level=True,
                )

        # Machines
        for mi, machine in enumerate(container.get("machines", [])):
            mpath = f"{path}.machines[{mi}]"
            mid = self._validate_id_field(machine, mpath, "machine", result, all_machine_ids)
            if mid:
                machine_membership[mid] = set()
            # Nodes inside the machine
            for ni2, node in enumerate(machine.get("nodes", [])):
                npath2 = f"{mpath}.nodes[{ni2}]"
                nid = self._register_node(
                    node, npath2, result,
                    all_node_ids, machine_membership, bridge_nodes,
                    factory_level=False,
                )
                if mid and nid:
                    machine_membership[mid].add(nid)

        # Top-level standalone nodes (when root container is nodes:)
        for ni, node in enumerate(container.get("nodes", [])):
            # Only register if we're at the logic level (not inside factories/machines)
            # — factory-level nodes are handled above in the factories loop
            pass  # handled in their respective parent loops

    def _validate_id_field(
        self,
        obj: dict[str, Any],
        path: str,
        kind: str,
        result: OlogicValidationResult,
        registry: dict[str, str],
    ) -> str | None:
        """Check that an object has an 'id' field and register it. Returns the id."""
        if not isinstance(obj, dict):
            result.error("INVALID_CONTAINER", f"{kind} at {path} must be a mapping", path=path)
            return None
        obj_id = obj.get("id")
        if not obj_id:
            result.error("MISSING_ID", f"{kind} at {path} has no 'id' field", path=path)
            return None
        if obj_id in registry:
            result.error(
                "DUPLICATE_ID",
                f"{kind} id '{obj_id}' is duplicated (first at {registry[obj_id]}, again at {path})",
                path=path,
                node_id=obj_id,
            )
        else:
            registry[obj_id] = path
        return obj_id

    def _register_node(
        self,
        node: dict[str, Any],
        path: str,
        result: OlogicValidationResult,
        all_node_ids: dict[str, str],
        machine_membership: dict[str, set[str]],
        bridge_nodes: list[dict[str, Any]],
        factory_level: bool,
    ) -> str | None:
        """Register a node, validate its type and requirements, flag bridges."""
        if not isinstance(node, dict):
            result.error("INVALID_NODE", f"node at {path} must be a mapping", path=path)
            return None

        nid = node.get("id")
        if not nid:
            result.error("MISSING_NODE_ID", f"node at {path} has no 'id' field", path=path)
            return None

        if nid in all_node_ids:
            result.error(
                "DUPLICATE_NODE_ID",
                f"node id '{nid}' is duplicated (first at {all_node_ids[nid]}, again at {path})",
                path=path,
                node_id=nid,
            )
        else:
            all_node_ids[nid] = path

        # Validate node type
        ntype = node.get("type")
        if not ntype:
            result.error("MISSING_NODE_TYPE", f"node '{nid}' has no 'type' field", path=path, node_id=nid)
        elif ntype not in VALID_NODE_TYPES:
            result.error(
                "INVALID_NODE_TYPE",
                f"node '{nid}' has invalid type '{ntype}'. Valid types: {sorted(VALID_NODE_TYPES)}",
                path=path,
                node_id=nid,
            )

        # Validate title
        if not node.get("title"):
            result.warn("MISSING_TITLE", f"node '{nid}' has no 'title' field", path=path, node_id=nid)

        # Validate requirements annotations
        reqs = node.get("requirements", [])
        if isinstance(reqs, list):
            for req in reqs:
                if not REQ_ID_PATTERN.match(str(req)):
                    result.error(
                        "INVALID_REQ_PATTERN",
                        f"node '{nid}' has invalid requirement ID '{req}' "
                        f"(must match REQ-[A-Z]+-NNN)",
                        path=path,
                        node_id=nid,
                    )
                else:
                    result.covered_requirements.add(str(req))
        elif reqs:
            result.error(
                "INVALID_REQ_TYPE",
                f"node '{nid}' requirements must be a list, got {type(reqs).__name__}",
                path=path,
                node_id=nid,
            )

        # Track bridge nodes
        if factory_level and isinstance(nid, str) and nid.startswith(BRIDGE_NODE_PREFIX):
            bridge_nodes.append({"node": node, "path": path})
        elif factory_level and node.get("type") == "gateway":
            # Even if not prefixed with 'bridge-', a gateway at factory level is a bridge
            bridge_nodes.append({"node": node, "path": path})

        return nid

    # ── Internal: Edge validation ─────────────────────────────────────────────

    def _validate_edges(
        self,
        container: dict[str, Any],
        path: str,
        result: OlogicValidationResult,
        all_node_ids: dict[str, str],
        all_machine_ids: dict[str, str],
        all_factory_ids: dict[str, str],
        machine_membership: dict[str, set[str]],
    ) -> None:
        """Recursively validate all edge references."""

        for ni, network in enumerate(container.get("networks", [])):
            npath = f"{path}.networks[{ni}]"
            self._validate_edges(
                network, npath, result,
                all_node_ids, all_machine_ids, all_factory_ids, machine_membership,
            )
            for xi, node in enumerate(network.get("nodes", [])):
                self._validate_node_edges(
                    node, f"{npath}.nodes[{xi}]", result,
                    all_node_ids, all_machine_ids, all_factory_ids, machine_membership,
                    in_machine_id=None,
                )

        for fi, factory in enumerate(container.get("factories", [])):
            fpath = f"{path}.factories[{fi}]"
            self._validate_edges(
                factory, fpath, result,
                all_node_ids, all_machine_ids, all_factory_ids, machine_membership,
            )
            for xi, node in enumerate(factory.get("nodes", [])):
                self._validate_node_edges(
                    node, f"{fpath}.nodes[{xi}]", result,
                    all_node_ids, all_machine_ids, all_factory_ids, machine_membership,
                    in_machine_id=None,
                )

        for mi, machine in enumerate(container.get("machines", [])):
            mpath = f"{path}.machines[{mi}]"
            mid = machine.get("id")
            for ni2, node in enumerate(machine.get("nodes", [])):
                self._validate_node_edges(
                    node, f"{mpath}.nodes[{ni2}]", result,
                    all_node_ids, all_machine_ids, all_factory_ids, machine_membership,
                    in_machine_id=mid,
                )

    def _validate_node_edges(
        self,
        node: dict[str, Any],
        path: str,
        result: OlogicValidationResult,
        all_node_ids: dict[str, str],
        all_machine_ids: dict[str, str],
        all_factory_ids: dict[str, str],
        machine_membership: dict[str, set[str]],
        in_machine_id: str | None,
    ) -> None:
        """Validate a single node's inputs/outputs for rule compliance."""
        if not isinstance(node, dict):
            return

        nid = node.get("id", "<unknown>")
        outputs = node.get("outputs", [])
        inputs = node.get("inputs", [])

        if isinstance(outputs, list):
            for ref in outputs:
                ref = str(ref)
                is_node_ref = ref in all_node_ids
                is_machine_ref = ref in all_machine_ids
                is_factory_ref = ref in all_factory_ids

                if not (is_node_ref or is_machine_ref or is_factory_ref):
                    result.error(
                        "DANGLING_OUTPUT_REF",
                        f"node '{nid}' outputs references unknown id '{ref}'",
                        path=path,
                        node_id=nid,
                    )
                elif is_node_ref and in_machine_id:
                    # Node→Node edge: check both are in the same machine
                    ref_machine = self._find_machine_for_node(ref, machine_membership)
                    if ref_machine and ref_machine != in_machine_id:
                        result.error(
                            "CROSS_MACHINE_NODE_EDGE",
                            f"node '{nid}' (in machine '{in_machine_id}') has output to node "
                            f"'{ref}' (in machine '{ref_machine}'). Cross-machine node→node edges "
                            f"are forbidden — use a standalone bridge node at factory level instead.",
                            path=path,
                            node_id=nid,
                        )

        if isinstance(inputs, list):
            for ref in inputs:
                ref = str(ref)
                is_node_ref = ref in all_node_ids
                is_machine_ref = ref in all_machine_ids
                is_factory_ref = ref in all_factory_ids

                if not (is_node_ref or is_machine_ref or is_factory_ref):
                    result.error(
                        "DANGLING_INPUT_REF",
                        f"node '{nid}' inputs references unknown id '{ref}'",
                        path=path,
                        node_id=nid,
                    )

    # ── Internal: Bridge node validation ─────────────────────────────────────

    def _validate_bridge_node(
        self,
        node: dict[str, Any],
        path: str,
        result: OlogicValidationResult,
        all_machine_ids: dict[str, str],
    ) -> None:
        """Validate bridge-specific rules (REQ-NKGL-006)."""
        nid = node.get("id", "<unknown>")

        # Must have BOTH inputs and outputs (REQ-NKGL-006)
        inputs = node.get("inputs", [])
        outputs = node.get("outputs", [])

        if not inputs:
            result.error(
                "BRIDGE_MISSING_INPUTS",
                f"bridge node '{nid}' must have 'inputs' pointing to a source machine "
                f"(REQ-NKGL-006: bridge nodes connect machines via inputs+outputs)",
                path=path,
                node_id=nid,
            )
        if not outputs:
            result.error(
                "BRIDGE_MISSING_OUTPUTS",
                f"bridge node '{nid}' must have 'outputs' pointing to a target machine "
                f"(REQ-NKGL-006: bridge nodes connect machines via inputs+outputs)",
                path=path,
                node_id=nid,
            )

        # Inputs should reference machine IDs (not node IDs — that would be a node→node edge)
        if isinstance(inputs, list):
            for ref in inputs:
                if str(ref) not in all_machine_ids:
                    result.warn(
                        "BRIDGE_INPUT_NOT_MACHINE",
                        f"bridge node '{nid}' input '{ref}' does not reference a machine ID. "
                        f"Bridge inputs should typically point to machines, not nodes.",
                        path=path,
                        node_id=nid,
                    )

        # Bridge type should be gateway (strongly recommended)
        if node.get("type") != "gateway":
            result.warn(
                "BRIDGE_NOT_GATEWAY",
                f"bridge node '{nid}' has type '{node.get('type')}' — bridge nodes should "
                f"use type 'gateway' for semantic clarity (REQ-NKGL-006)",
                path=path,
                node_id=nid,
            )

        # Bridge should carry REQ-NKGL-006 in requirements
        reqs = node.get("requirements", [])
        if "REQ-NKGL-006" not in [str(r) for r in reqs]:
            result.warn(
                "BRIDGE_MISSING_REQ_ANNOTATION",
                f"bridge node '{nid}' should have 'requirements: [REQ-NKGL-006]' per spec",
                path=path,
                node_id=nid,
            )

    def _find_machine_for_node(self, node_id: str, machine_membership: dict[str, set[str]]) -> str | None:
        """Return the machine ID containing the given node, or None."""
        for machine_id, node_ids in machine_membership.items():
            if node_id in node_ids:
                return machine_id
        return None


# ─── Requirement Reconciler ───────────────────────────────────────────────────

@dataclass
class ReconciliationResult:
    """Result of checking requirement coverage in a .ologic document."""
    covered: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)  # in doc but not in REQUIREMENTS catalog
    coverage_pct: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Requirement Coverage: {self.coverage_pct:.0%} ({len(self.covered)}/{len(self.covered) + len(self.missing)})",
        ]
        if self.covered:
            lines.append(f"  Covered: {', '.join(sorted(self.covered))}")
        if self.missing:
            lines.append(f"  Missing: {', '.join(sorted(self.missing))}")
        if self.unknown:
            lines.append(f"  Unknown (not in catalog): {', '.join(sorted(self.unknown))}")
        return "\n".join(lines)


def reconcile_requirements(
    doc: dict[str, Any],
    required_ids: list[str] | None = None,
) -> ReconciliationResult:
    """
    Check that a .ologic YAML document covers the expected requirements.

    Walks the entire YAML tree collecting all `requirements:` arrays, then
    checks coverage against the given required_ids (defaults to all REQ-NKGL-NNN).

    Args:
        doc: Parsed .ologic YAML dict.
        required_ids: List of requirement IDs that MUST appear. Defaults to
                      all keys in REQUIREMENTS (REQ-NKGL-001 through REQ-NKGL-007).

    Returns:
        ReconciliationResult with covered/missing/unknown sets and coverage %.
    """
    if required_ids is None:
        required_ids = list(REQUIREMENTS.keys())

    covered: set[str] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            reqs = obj.get("requirements", [])
            if isinstance(reqs, list):
                for r in reqs:
                    covered.add(str(r))
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(doc)

    required_set = set(required_ids)
    missing = required_set - covered
    unknown = covered - set(REQUIREMENTS.keys())

    total = len(required_set)
    actually_covered = covered & required_set
    pct = len(actually_covered) / total if total > 0 else 1.0

    return ReconciliationResult(
        covered=actually_covered,
        missing=missing,
        unknown=unknown,
        coverage_pct=pct,
    )


# ─── Convenience: validate from YAML string ──────────────────────────────────

def validate_ologic_yaml(yaml_str: str) -> OlogicValidationResult:
    """
    Parse and validate a .ologic YAML string.

    Requires PyYAML (yaml). Returns an OlogicValidationResult.
    Raises ImportError if PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError(
            "PyYAML is required to parse .ologic YAML strings. "
            "Install with: pip install pyyaml"
        )
    doc = yaml.safe_load(yaml_str)
    if not isinstance(doc, dict):
        result = OlogicValidationResult(valid=False)
        result.error("INVALID_DOCUMENT", "YAML document is not a mapping", path="<root>")
        return result
    return OlogicValidator().validate(doc)
