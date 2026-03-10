"""
Tests for ologic.py — .ologic YAML validator and requirement reconciler.

Tests cover:
- OlogicValidator: valid docs, structural violations, bridge rules, ref resolution
- reconcile_requirements: coverage detection, missing reqs, unknown reqs
- REQUIREMENTS catalog: all REQ-NKGL-002..007 defined
- Integration: full pipeline fixture (entities → cluster → bridge → emit → validate)
- Structural decision table: machines/factories/networks selection rules

The validator tests use hand-crafted YAML dicts (no live Ollama/scipy needed).
The pipeline integration test uses mock embedder + mock cluster output to verify
the validator accepts what the emitter will produce.
"""

from __future__ import annotations

import pytest
from nous.ologic import (
    REQUIREMENTS,
    VALID_NODE_TYPES,
    OlogicError,
    OlogicValidationResult,
    OlogicValidator,
    ReconciliationResult,
    reconcile_requirements,
)


# ─── Fixtures: valid .ologic documents ───────────────────────────────────────


def _minimal_machines_doc() -> dict:
    """Single cluster, no bridges → machines: root."""
    return {
        "logic": {
            "mode": "diagramming",
            "version": "2.0",
            "machines": [
                {
                    "id": "machine-cluster-0",
                    "nodes": [
                        {
                            "id": "entity-embed-text",
                            "type": "process",
                            "title": "Embed Text",
                            "requirements": ["REQ-NKGL-002"],
                            "outputs": ["entity-hac-cluster"],
                        },
                        {
                            "id": "entity-hac-cluster",
                            "type": "process",
                            "title": "HAC Cluster",
                            "requirements": ["REQ-NKGL-003"],
                        },
                    ],
                }
            ],
        }
    }


def _factories_doc_no_bridges() -> dict:
    """Two disconnected clusters, no bridges → factories: root."""
    return {
        "logic": {
            "mode": "diagramming",
            "version": "2.0",
            "factories": [
                {
                    "id": "cluster-0",
                    "machines": [
                        {
                            "id": "machine-cluster-0",
                            "nodes": [
                                {
                                    "id": "entity-a",
                                    "type": "source",
                                    "title": "Entity A",
                                    "requirements": ["REQ-NKGL-002"],
                                    "outputs": ["entity-b"],
                                },
                                {
                                    "id": "entity-b",
                                    "type": "process",
                                    "title": "Entity B",
                                    "requirements": ["REQ-NKGL-003"],
                                },
                            ],
                        }
                    ],
                },
                {
                    "id": "cluster-1",
                    "machines": [
                        {
                            "id": "machine-cluster-1",
                            "nodes": [
                                {
                                    "id": "entity-c",
                                    "type": "database",
                                    "title": "Entity C",
                                    "requirements": ["REQ-NKGL-004"],
                                    "outputs": ["entity-d"],
                                },
                                {
                                    "id": "entity-d",
                                    "type": "process",
                                    "title": "Entity D",
                                    "requirements": ["REQ-NKGL-003"],
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    }


def _networks_doc_with_bridges() -> dict:
    """Two clusters with a bridge → networks: root (full Omicron-spec output shape)."""
    return {
        "logic": {
            "mode": "diagramming",
            "version": "2.0",
            "networks": [
                {
                    "id": "semantic-hierarchy",
                    "factories": [
                        {
                            "id": "cluster-0",
                            "machines": [
                                {
                                    "id": "machine-cluster-0",
                                    "nodes": [
                                        {
                                            "id": "entity-embed",
                                            "type": "process",
                                            "title": "Embed",
                                            "requirements": ["REQ-NKGL-002"],
                                            "outputs": ["entity-cluster"],
                                        },
                                        {
                                            "id": "entity-cluster",
                                            "type": "process",
                                            "title": "Cluster",
                                            "requirements": ["REQ-NKGL-003"],
                                        },
                                    ],
                                }
                            ],
                            "nodes": [
                                {
                                    "id": "bridge-cluster-0-cluster-1",
                                    "type": "gateway",
                                    "title": "Bridge: Embed → Bridge",
                                    "inputs": ["machine-cluster-0"],
                                    "outputs": ["machine-cluster-1"],
                                    "requirements": ["REQ-NKGL-006"],
                                }
                            ],
                        },
                        {
                            "id": "cluster-1",
                            "machines": [
                                {
                                    "id": "machine-cluster-1",
                                    "nodes": [
                                        {
                                            "id": "entity-leiden",
                                            "type": "process",
                                            "title": "Leiden",
                                            "requirements": ["REQ-NKGL-005"],
                                            "outputs": ["entity-emit"],
                                        },
                                        {
                                            "id": "entity-emit",
                                            "type": "output",
                                            "title": "Emit .ologic",
                                            "requirements": ["REQ-NKGL-007"],
                                        },
                                    ],
                                }
                            ],
                            "nodes": [],
                        },
                    ],
                }
            ],
        }
    }


# ─── Requirements Catalog Tests ───────────────────────────────────────────────


class TestRequirementsCatalog:
    def test_all_nkgl_reqs_defined(self):
        """REQ-NKGL-001 through REQ-NKGL-007 must all be in the catalog."""
        for i in range(1, 8):
            req_id = f"REQ-NKGL-00{i}"
            assert req_id in REQUIREMENTS, f"{req_id} missing from REQUIREMENTS catalog"

    def test_requirement_fields_populated(self):
        """Each requirement must have non-empty id, title, description."""
        for req_id, req in REQUIREMENTS.items():
            assert req.id == req_id, f"{req_id}.id mismatch"
            assert req.title, f"{req_id} has empty title"
            assert req.description, f"{req_id} has empty description"
            assert len(req.description) > 20, f"{req_id} description too short (likely placeholder)"

    def test_valid_node_types_not_empty(self):
        """VALID_NODE_TYPES must include at least the Ordinal spec types."""
        required_types = {"process", "source", "database", "gateway", "output", "input", "monitor"}
        assert required_types.issubset(VALID_NODE_TYPES), (
            f"Missing types from VALID_NODE_TYPES: {required_types - VALID_NODE_TYPES}"
        )


# ─── OlogicValidator: Root Structure Tests ────────────────────────────────────


class TestValidatorRootStructure:
    def setup_method(self):
        self.v = OlogicValidator()

    def test_valid_machines_doc(self):
        result = self.v.validate(_minimal_machines_doc())
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_valid_factories_doc(self):
        result = self.v.validate(_factories_doc_no_bridges())
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_valid_networks_doc_with_bridges(self):
        result = self.v.validate(_networks_doc_with_bridges())
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_missing_logic_root(self):
        result = self.v.validate({"not_logic": {}})
        assert not result.valid
        assert any(e.code == "MISSING_ROOT_KEY" for e in result.errors)

    def test_wrong_mode(self):
        doc = _minimal_machines_doc()
        doc["logic"]["mode"] = "wrong"
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "WRONG_MODE" for e in result.errors)

    def test_wrong_version(self):
        doc = _minimal_machines_doc()
        doc["logic"]["version"] = "1.0"
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "WRONG_VERSION" for e in result.errors)

    def test_version_as_float_still_valid(self):
        """version: 2.0 (float) and version: '2.0' (string) both accepted."""
        doc = _minimal_machines_doc()
        doc["logic"]["version"] = 2.0  # YAML may parse as float
        result = self.v.validate(doc)
        assert result.valid, f"Float version 2.0 should be accepted, errors: {result.errors}"

    def test_no_container_key(self):
        result = self.v.validate({"logic": {"mode": "diagramming", "version": "2.0"}})
        assert not result.valid
        assert any(e.code == "NO_CONTAINER" for e in result.errors)

    def test_multiple_container_keys(self):
        doc = _minimal_machines_doc()
        doc["logic"]["factories"] = []
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "MULTIPLE_CONTAINERS" for e in result.errors)


# ─── OlogicValidator: Node Tests ─────────────────────────────────────────────


class TestValidatorNodes:
    def setup_method(self):
        self.v = OlogicValidator()

    def test_missing_node_id(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0].pop("id")
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "MISSING_NODE_ID" for e in result.errors)

    def test_duplicate_node_id(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][1]["id"] = "entity-embed-text"  # duplicate
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "DUPLICATE_NODE_ID" for e in result.errors)

    def test_invalid_node_type(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0]["type"] = "robot"
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "INVALID_NODE_TYPE" for e in result.errors)

    def test_missing_node_type(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0].pop("type")
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "MISSING_NODE_TYPE" for e in result.errors)

    def test_missing_title_is_warning_not_error(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0].pop("title")
        result = self.v.validate(doc)
        # Still valid — title is a warning only
        assert result.valid
        assert any(w.code == "MISSING_TITLE" for w in result.warnings)

    def test_invalid_req_pattern(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0]["requirements"] = ["BADREQ-999"]
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "INVALID_REQ_PATTERN" for e in result.errors)

    def test_valid_req_patterns(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0]["requirements"] = [
            "REQ-NKGL-002", "REQ-NKGL-003", "REQ-FOO-001"
        ]
        result = self.v.validate(doc)
        assert result.valid
        assert "REQ-NKGL-002" in result.covered_requirements
        assert "REQ-NKGL-003" in result.covered_requirements
        assert "REQ-FOO-001" in result.covered_requirements

    def test_all_valid_node_types_accepted(self):
        """Every type in VALID_NODE_TYPES should pass validation."""
        for ntype in VALID_NODE_TYPES:
            doc = {
                "logic": {
                    "mode": "diagramming",
                    "version": "2.0",
                    "machines": [
                        {
                            "id": "m0",
                            "nodes": [
                                {"id": "n0", "type": ntype, "title": f"Test {ntype}"}
                            ],
                        }
                    ],
                }
            }
            result = self.v.validate(doc)
            assert result.valid, (
                f"Type '{ntype}' should be valid, got errors: {result.errors}"
            )


# ─── OlogicValidator: Edge / Reference Tests ─────────────────────────────────


class TestValidatorEdges:
    def setup_method(self):
        self.v = OlogicValidator()

    def test_valid_intra_machine_outputs(self):
        """node→node edges within same machine are valid."""
        result = self.v.validate(_minimal_machines_doc())
        assert result.valid

    def test_dangling_output_ref(self):
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0]["outputs"] = ["nonexistent-id"]
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "DANGLING_OUTPUT_REF" for e in result.errors)

    def test_dangling_input_ref(self):
        doc = _networks_doc_with_bridges()
        # Corrupt the bridge's inputs
        bridge = doc["logic"]["networks"][0]["factories"][0]["nodes"][0]
        bridge["inputs"] = ["nonexistent-machine"]
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "DANGLING_INPUT_REF" for e in result.errors)

    def test_cross_machine_node_edge_forbidden(self):
        """node in machine-0 outputting to node in machine-1 must be flagged."""
        doc = _factories_doc_no_bridges()
        # Make entity-a in machine-cluster-0 output to entity-c in machine-cluster-1
        doc["logic"]["factories"][0]["machines"][0]["nodes"][0]["outputs"] = ["entity-c"]
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "CROSS_MACHINE_NODE_EDGE" for e in result.errors)

    def test_machine_to_machine_via_bridge_is_valid(self):
        """Bridge node's outputs pointing to a machine ID is allowed."""
        result = self.v.validate(_networks_doc_with_bridges())
        assert result.valid, f"Networks-with-bridges should be valid, errors: {result.errors}"

    def test_outputs_to_machine_id_valid(self):
        """A standalone factory-level node can output to a machine id."""
        doc = _factories_doc_no_bridges()
        # Add a standalone node at factory[0] level that outputs to machine-cluster-1
        doc["logic"]["factories"][0]["nodes"] = [
            {
                "id": "bridge-cluster-0-cluster-1",
                "type": "gateway",
                "title": "Bridge",
                "inputs": ["machine-cluster-0"],
                "outputs": ["machine-cluster-1"],
                "requirements": ["REQ-NKGL-006"],
            }
        ]
        result = self.v.validate(doc)
        assert result.valid, f"Bridge to machine-cluster-1 should be valid, errors: {result.errors}"


# ─── OlogicValidator: Bridge Node Tests ──────────────────────────────────────


class TestValidatorBridgeNodes:
    def setup_method(self):
        self.v = OlogicValidator()

    def test_bridge_missing_inputs_is_error(self):
        doc = _networks_doc_with_bridges()
        bridge = doc["logic"]["networks"][0]["factories"][0]["nodes"][0]
        bridge.pop("inputs")
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "BRIDGE_MISSING_INPUTS" for e in result.errors)

    def test_bridge_missing_outputs_is_error(self):
        doc = _networks_doc_with_bridges()
        bridge = doc["logic"]["networks"][0]["factories"][0]["nodes"][0]
        bridge.pop("outputs")
        result = self.v.validate(doc)
        assert not result.valid
        assert any(e.code == "BRIDGE_MISSING_OUTPUTS" for e in result.errors)

    def test_bridge_not_gateway_type_is_warning(self):
        doc = _networks_doc_with_bridges()
        bridge = doc["logic"]["networks"][0]["factories"][0]["nodes"][0]
        bridge["type"] = "process"  # not gateway
        result = self.v.validate(doc)
        # Type mismatch is a warning, not error (validator is lenient on bridge type)
        assert result.valid
        assert any(w.code == "BRIDGE_NOT_GATEWAY" for w in result.warnings)

    def test_bridge_missing_req_annotation_is_warning(self):
        doc = _networks_doc_with_bridges()
        bridge = doc["logic"]["networks"][0]["factories"][0]["nodes"][0]
        bridge["requirements"] = ["REQ-NKGL-007"]  # missing REQ-NKGL-006
        result = self.v.validate(doc)
        assert result.valid
        assert any(w.code == "BRIDGE_MISSING_REQ_ANNOTATION" for w in result.warnings)

    def test_bridge_with_correct_spec_no_warnings(self):
        """A perfectly-spec'd bridge node should have zero warnings."""
        result = self.v.validate(_networks_doc_with_bridges())
        bridge_warnings = [
            w for w in result.warnings
            if "bridge" in w.code.lower()
        ]
        assert not bridge_warnings, f"Unexpected bridge warnings: {bridge_warnings}"

    def test_bridge_count_tracked(self):
        result = self.v.validate(_networks_doc_with_bridges())
        assert result.bridge_count == 1

    def test_no_bridges_in_machines_doc(self):
        result = self.v.validate(_minimal_machines_doc())
        assert result.bridge_count == 0


# ─── OlogicValidator: Stats Tracking ─────────────────────────────────────────


class TestValidatorStats:
    def setup_method(self):
        self.v = OlogicValidator()

    def test_machines_doc_stats(self):
        result = self.v.validate(_minimal_machines_doc())
        assert result.node_count == 2
        assert result.machine_count == 1
        assert result.factory_count == 0
        assert result.bridge_count == 0

    def test_factories_doc_stats(self):
        result = self.v.validate(_factories_doc_no_bridges())
        assert result.node_count == 4
        assert result.machine_count == 2
        assert result.factory_count == 2
        assert result.bridge_count == 0

    def test_networks_doc_stats(self):
        result = self.v.validate(_networks_doc_with_bridges())
        assert result.node_count == 5  # 4 entity nodes + 1 bridge
        assert result.machine_count == 2
        assert result.factory_count == 2
        assert result.bridge_count == 1

    def test_requirements_covered_collected(self):
        result = self.v.validate(_networks_doc_with_bridges())
        assert "REQ-NKGL-002" in result.covered_requirements
        assert "REQ-NKGL-003" in result.covered_requirements
        assert "REQ-NKGL-005" in result.covered_requirements
        assert "REQ-NKGL-006" in result.covered_requirements
        assert "REQ-NKGL-007" in result.covered_requirements


# ─── OlogicValidator: Structural Decision Table Tests ────────────────────────


class TestStructuralDecisionTable:
    """
    Validate the structural decision table from Omicron's emitter spec:

    | Clusters | Bridges | Root key   |
    |----------|---------|------------|
    | 1        | 0       | machines:  |
    | 2+       | 0       | factories: |
    | 2+       | 1+      | networks:  |

    Violations emit warnings (not errors) since the validator is lenient for
    hand-authored docs.
    """

    def setup_method(self):
        self.v = OlogicValidator()

    def test_two_machines_at_root_warns(self):
        """machines: with 2 machines should warn to use factories:"""
        doc = {
            "logic": {
                "mode": "diagramming",
                "version": "2.0",
                "machines": [
                    {
                        "id": "m0",
                        "nodes": [{"id": "n0", "type": "process", "title": "N0"}],
                    },
                    {
                        "id": "m1",
                        "nodes": [{"id": "n1", "type": "process", "title": "N1"}],
                    },
                ],
            }
        }
        result = self.v.validate(doc)
        assert result.valid  # warning only
        assert any(
            w.code == "STRUCTURAL_MULTI_CLUSTER_SHOULD_USE_FACTORIES"
            for w in result.warnings
        )

    def test_factories_with_bridge_warns(self):
        """factories: with bridge nodes should warn to use networks:"""
        doc = _factories_doc_no_bridges()
        # Add a bridge at factory level
        doc["logic"]["factories"][0]["nodes"] = [
            {
                "id": "bridge-cluster-0-cluster-1",
                "type": "gateway",
                "title": "Bridge",
                "inputs": ["machine-cluster-0"],
                "outputs": ["machine-cluster-1"],
                "requirements": ["REQ-NKGL-006"],
            }
        ]
        result = self.v.validate(doc)
        assert result.valid
        assert any(
            w.code == "STRUCTURAL_BRIDGES_SHOULD_USE_NETWORKS"
            for w in result.warnings
        )

    def test_correct_machines_single_cluster_no_warning(self):
        """Single cluster with machines: root → no structural warning."""
        result = self.v.validate(_minimal_machines_doc())
        structural_warnings = [
            w for w in result.warnings if w.code.startswith("STRUCTURAL_")
        ]
        assert not structural_warnings, f"Unexpected structural warnings: {structural_warnings}"

    def test_correct_factories_two_clusters_no_bridges_no_warning(self):
        """Two disconnected clusters with factories: → no structural warning."""
        result = self.v.validate(_factories_doc_no_bridges())
        structural_warnings = [
            w for w in result.warnings if w.code.startswith("STRUCTURAL_")
        ]
        assert not structural_warnings, f"Unexpected structural warnings: {structural_warnings}"

    def test_correct_networks_with_bridge_no_warning(self):
        """Two clusters bridged with networks: → no structural warning."""
        result = self.v.validate(_networks_doc_with_bridges())
        structural_warnings = [
            w for w in result.warnings if w.code.startswith("STRUCTURAL_")
        ]
        assert not structural_warnings, f"Unexpected structural warnings: {structural_warnings}"


# ─── Reconcile Requirements Tests ────────────────────────────────────────────


class TestReconcileRequirements:
    def test_full_coverage(self):
        """A doc that annotates REQ-NKGL-002 through 007 shows full coverage."""
        # Build a doc that explicitly annotates all six requirements
        doc = {
            "logic": {
                "mode": "diagramming",
                "version": "2.0",
                "networks": [
                    {
                        "id": "semantic-hierarchy",
                        "factories": [
                            {
                                "id": "cluster-0",
                                "machines": [
                                    {
                                        "id": "machine-cluster-0",
                                        "nodes": [
                                            {
                                                "id": "entity-embed",
                                                "type": "process",
                                                "title": "Embed",
                                                # REQ-002: entity embedding
                                                "requirements": ["REQ-NKGL-002"],
                                                "outputs": ["entity-hac"],
                                            },
                                            {
                                                "id": "entity-hac",
                                                "type": "process",
                                                "title": "HAC",
                                                # REQ-003: HAC clustering, REQ-004: ClusterResult
                                                "requirements": ["REQ-NKGL-003", "REQ-NKGL-004"],
                                            },
                                        ],
                                    }
                                ],
                                "nodes": [
                                    {
                                        "id": "bridge-cluster-0-cluster-1",
                                        "type": "gateway",
                                        "title": "Bridge",
                                        "inputs": ["machine-cluster-0"],
                                        "outputs": ["machine-cluster-1"],
                                        # REQ-005: Leiden, REQ-006: bridge emission
                                        "requirements": ["REQ-NKGL-005", "REQ-NKGL-006"],
                                    }
                                ],
                            },
                            {
                                "id": "cluster-1",
                                "machines": [
                                    {
                                        "id": "machine-cluster-1",
                                        "nodes": [
                                            {
                                                "id": "entity-emit",
                                                "type": "output",
                                                "title": "Emit .ologic",
                                                # REQ-007: valid .ologic emission
                                                "requirements": ["REQ-NKGL-007"],
                                            }
                                        ],
                                    }
                                ],
                                "nodes": [],
                            },
                        ],
                    }
                ],
            }
        }
        result = reconcile_requirements(
            doc,
            required_ids=[f"REQ-NKGL-00{i}" for i in range(2, 8)],
        )
        assert result.missing == set(), f"Missing reqs: {result.missing}"
        assert result.coverage_pct == 1.0

    def test_partial_coverage(self):
        """machines doc only covers 002 and 003."""
        result = reconcile_requirements(
            _minimal_machines_doc(),
            required_ids=[f"REQ-NKGL-00{i}" for i in range(2, 8)],
        )
        assert "REQ-NKGL-002" in result.covered
        assert "REQ-NKGL-003" in result.covered
        assert "REQ-NKGL-004" in result.missing
        assert result.coverage_pct < 1.0

    def test_unknown_requirements_detected(self):
        """Requirements in doc but not in REQUIREMENTS catalog go to unknown."""
        doc = _minimal_machines_doc()
        doc["logic"]["machines"][0]["nodes"][0]["requirements"] = ["REQ-CUSTOM-999"]
        result = reconcile_requirements(doc, required_ids=["REQ-NKGL-002"])
        assert "REQ-CUSTOM-999" in result.unknown

    def test_empty_doc_all_missing(self):
        doc = {"logic": {"mode": "diagramming", "version": "2.0", "machines": []}}
        result = reconcile_requirements(
            doc, required_ids=["REQ-NKGL-002", "REQ-NKGL-003"]
        )
        assert result.covered == set()
        assert result.missing == {"REQ-NKGL-002", "REQ-NKGL-003"}
        assert result.coverage_pct == 0.0

    def test_default_required_ids_is_full_catalog(self):
        """reconcile_requirements() with no required_ids checks all REQUIREMENTS keys."""
        result = reconcile_requirements(_networks_doc_with_bridges())
        # REQ-NKGL-001 is not in our fixture — should be missing
        assert "REQ-NKGL-001" in result.missing

    def test_summary_format(self):
        result = reconcile_requirements(
            _networks_doc_with_bridges(),
            required_ids=["REQ-NKGL-002", "REQ-NKGL-999"],
        )
        summary = result.summary()
        assert "Coverage" in summary

    def test_nested_requirements_walked(self):
        """reconcile_requirements walks nested structures."""
        doc = _networks_doc_with_bridges()
        result = reconcile_requirements(doc, required_ids=["REQ-NKGL-006"])
        assert "REQ-NKGL-006" in result.covered


# ─── OlogicValidationResult: Summary Format ──────────────────────────────────


class TestValidationResultSummary:
    def test_valid_doc_summary_shows_valid(self):
        v = OlogicValidator()
        result = v.validate(_minimal_machines_doc())
        summary = result.summary()
        assert "VALID" in summary
        assert "INVALID" not in summary

    def test_invalid_doc_summary_shows_invalid(self):
        v = OlogicValidator()
        result = v.validate({"not_logic": {}})
        summary = result.summary()
        assert "INVALID" in summary

    def test_summary_includes_error_codes(self):
        v = OlogicValidator()
        doc = _minimal_machines_doc()
        doc["logic"]["mode"] = "wrong"
        result = v.validate(doc)
        summary = result.summary()
        assert "WRONG_MODE" in summary

    def test_summary_includes_counts(self):
        v = OlogicValidator()
        result = v.validate(_networks_doc_with_bridges())
        summary = result.summary()
        assert "Nodes:" in summary
        assert "Machines:" in summary
        assert "Bridges:" in summary


# ─── Integration: Full Pipeline Simulation ───────────────────────────────────


class TestPipelineIntegration:
    """
    Simulate the full semantic hierarchy pipeline:
      entities → (mock embed) → (mock cluster) → (mock bridge) → emitted YAML → validate

    This tests that what the emitter produces (per Omicron's spec) passes the validator.
    No live Ollama or scipy required — we build the expected output structure directly.
    """

    def _build_pipeline_output(
        self,
        n_clusters: int,
        n_bridges: int,
    ) -> dict:
        """
        Build a plausible .ologic YAML doc as the emitter (hierarchy.py) would produce.

        Args:
            n_clusters: Number of HAC clusters (machines).
            n_bridges: Number of Leiden bridges between adjacent clusters.

        Returns:
            A dict suitable for OlogicValidator.validate().
        """
        if n_clusters == 1 and n_bridges == 0:
            # machines: root
            return {
                "logic": {
                    "mode": "diagramming",
                    "version": "2.0",
                    "machines": [
                        {
                            "id": "machine-cluster-0",
                            "nodes": [
                                {
                                    "id": "entity-0-0",
                                    "type": "source",
                                    "title": "Entity 0-0",
                                    "requirements": ["REQ-NKGL-002"],
                                    "outputs": ["entity-0-1"],
                                },
                                {
                                    "id": "entity-0-1",
                                    "type": "process",
                                    "title": "Entity 0-1",
                                    "requirements": ["REQ-NKGL-003"],
                                },
                            ],
                        }
                    ],
                }
            }

        elif n_clusters >= 2 and n_bridges == 0:
            # factories: root — disconnected clusters
            factories = []
            for ci in range(n_clusters):
                factories.append(
                    {
                        "id": f"cluster-{ci}",
                        "machines": [
                            {
                                "id": f"machine-cluster-{ci}",
                                "nodes": [
                                    {
                                        "id": f"entity-{ci}-0",
                                        "type": "source",
                                        "title": f"Entity {ci}-0",
                                        "requirements": ["REQ-NKGL-002"],
                                        "outputs": [f"entity-{ci}-1"],
                                    },
                                    {
                                        "id": f"entity-{ci}-1",
                                        "type": "process",
                                        "title": f"Entity {ci}-1",
                                        "requirements": ["REQ-NKGL-003"],
                                    },
                                ],
                            }
                        ],
                    }
                )
            return {
                "logic": {
                    "mode": "diagramming",
                    "version": "2.0",
                    "factories": factories,
                }
            }

        else:
            # networks: root — clusters + bridges
            factories = []
            for ci in range(n_clusters):
                bridge_nodes = []
                # Add bridge from ci to ci+1 if within bridge count
                if ci < n_bridges and ci + 1 < n_clusters:
                    bridge_nodes.append(
                        {
                            "id": f"bridge-cluster-{ci}-cluster-{ci + 1}",
                            "type": "gateway",
                            "title": f"Bridge: cluster-{ci} → cluster-{ci + 1}",
                            "inputs": [f"machine-cluster-{ci}"],
                            "outputs": [f"machine-cluster-{ci + 1}"],
                            "requirements": ["REQ-NKGL-006"],
                        }
                    )
                factories.append(
                    {
                        "id": f"cluster-{ci}",
                        "machines": [
                            {
                                "id": f"machine-cluster-{ci}",
                                "nodes": [
                                    {
                                        "id": f"entity-{ci}-0",
                                        "type": "source",
                                        "title": f"Entity {ci}-0",
                                        "requirements": ["REQ-NKGL-002"],
                                        "outputs": [f"entity-{ci}-1"],
                                    },
                                    {
                                        "id": f"entity-{ci}-1",
                                        "type": "process",
                                        "title": f"Entity {ci}-1",
                                        "requirements": ["REQ-NKGL-005"],
                                    },
                                ],
                            }
                        ],
                        "nodes": bridge_nodes,
                    }
                )
            return {
                "logic": {
                    "mode": "diagramming",
                    "version": "2.0",
                    "networks": [
                        {
                            "id": "semantic-hierarchy",
                            "factories": factories,
                        }
                    ],
                }
            }

    def test_single_cluster_no_bridge(self):
        doc = self._build_pipeline_output(n_clusters=1, n_bridges=0)
        result = OlogicValidator().validate(doc)
        assert result.valid, result.summary()
        assert result.machine_count == 1
        assert result.bridge_count == 0

    def test_two_clusters_no_bridges(self):
        doc = self._build_pipeline_output(n_clusters=2, n_bridges=0)
        result = OlogicValidator().validate(doc)
        assert result.valid, result.summary()
        assert result.machine_count == 2
        assert result.factory_count == 2
        assert result.bridge_count == 0

    def test_three_clusters_two_bridges(self):
        doc = self._build_pipeline_output(n_clusters=3, n_bridges=2)
        result = OlogicValidator().validate(doc)
        assert result.valid, result.summary()
        assert result.machine_count == 3
        assert result.bridge_count == 2

    def test_five_clusters_four_bridges(self):
        """Larger pipeline — chain of 5 clusters with 4 bridges."""
        doc = self._build_pipeline_output(n_clusters=5, n_bridges=4)
        result = OlogicValidator().validate(doc)
        assert result.valid, result.summary()
        assert result.machine_count == 5
        assert result.bridge_count == 4

    def test_emitter_output_requirement_coverage(self):
        """Pipeline output should cover REQ-NKGL-002, 003/005, 006."""
        doc = self._build_pipeline_output(n_clusters=3, n_bridges=2)
        result = OlogicValidator().validate(doc)
        assert "REQ-NKGL-002" in result.covered_requirements
        assert "REQ-NKGL-006" in result.covered_requirements

    def test_pipeline_reconcile_with_emitter_output(self):
        """reconcile_requirements on pipeline output should show coverage for 002, 005, 006."""
        doc = self._build_pipeline_output(n_clusters=3, n_bridges=2)
        recon = reconcile_requirements(
            doc, required_ids=["REQ-NKGL-002", "REQ-NKGL-005", "REQ-NKGL-006"]
        )
        assert recon.missing == set(), f"Missing: {recon.missing}"
        assert recon.coverage_pct == 1.0
