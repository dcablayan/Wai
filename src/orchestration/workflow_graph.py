"""Incremental workflow graph for Wai Ultra coordination."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.orchestration.protocol import CoordinationAction, CoordinationMessage


@dataclass
class WorkflowNode:
    """One executed action/result pair in an Ultra workflow."""

    turn_id: int
    expert_id: str
    role: str
    subtask_kind: str
    status: str
    parallel_group: str | None = None
    latency_ms: float = 0.0


@dataclass
class WorkflowEdge:
    """Directed edge between two turns."""

    source_turn_id: int
    target_turn_id: int
    edge_type: str


@dataclass
class WorkflowGraph:
    """Serializable graph of dependencies, access edges, and accepted output."""

    nodes: list[WorkflowNode] = field(default_factory=list)
    dependency_edges: list[WorkflowEdge] = field(default_factory=list)
    access_edges: list[WorkflowEdge] = field(default_factory=list)
    parallel_groups: dict[str, list[int]] = field(default_factory=dict)
    child_workflows: list[dict[str, Any]] = field(default_factory=list)
    final_accepted_node: int | None = None

    def add_turn(self, action: CoordinationAction, message: CoordinationMessage) -> None:
        self.nodes.append(
            WorkflowNode(
                turn_id=action.turn_id,
                expert_id=action.expert_id,
                role=action.role.value,
                subtask_kind=action.subtask_kind.value,
                status=message.status.value,
                parallel_group=action.parallel_group,
                latency_ms=float(message.latency_ms),
            )
        )
        for prior_turn in action.access_list:
            self.access_edges.append(
                WorkflowEdge(
                    source_turn_id=int(prior_turn),
                    target_turn_id=action.turn_id,
                    edge_type="access",
                )
            )
            self.dependency_edges.append(
                WorkflowEdge(
                    source_turn_id=int(prior_turn),
                    target_turn_id=action.turn_id,
                    edge_type="depends_on_visible_output",
                )
            )
        if action.parallel_group:
            self.parallel_groups.setdefault(action.parallel_group, []).append(action.turn_id)

    def mark_accepted(self, turn_id: int) -> None:
        self.final_accepted_node = int(turn_id)

    def add_child_workflow(
        self,
        *,
        parent_verifier_turn_id: int,
        child_graph: dict[str, Any],
        child_depth: int,
    ) -> None:
        self.child_workflows.append({
            "parent_verifier_turn_id": int(parent_verifier_turn_id),
            "child_depth": int(child_depth),
            "graph": child_graph,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "dependency_edges": [asdict(edge) for edge in self.dependency_edges],
            "access_edges": [asdict(edge) for edge in self.access_edges],
            "parallel_groups": dict(self.parallel_groups),
            "child_workflows": list(self.child_workflows),
            "final_accepted_node": self.final_accepted_node,
        }
