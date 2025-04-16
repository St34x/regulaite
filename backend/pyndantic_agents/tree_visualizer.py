# plugins/regul_aite/backend/pyndantic_agents/tree_visualizer.py
import json
import base64
import io
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from .tree_reasoning import DecisionTree, DecisionNode
from .decision_trees import get_available_trees

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# Add this router to the pyndantic_agents/router.py file
router = APIRouter(prefix="/tree_visualizer", tags=["tree_visualizer"])

class NodeInfo(BaseModel):
    """Node information for visualization"""
    id: str
    description: str
    is_leaf: bool = False
    is_probabilistic: bool = False
    children: Dict[str, str] = {}
    visited: bool = False
    confidence: Optional[float] = None
    decision: Optional[str] = None
    metadata: Dict[str, Any] = {}

class TreeVisualization(BaseModel):
    """Tree visualization data structure"""
    name: str
    description: str
    root_id: str
    nodes: Dict[str, NodeInfo]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    decision_path: List[Dict[str, Any]] = []

def generate_tree_visualization(
    tree: DecisionTree,
    decision_path: Optional[List[Dict[str, Any]]] = None,
    output_format: str = "json"
) -> Union[TreeVisualization, str, bytes]:
    """
    Generate a visualization of a decision tree.

    Args:
        tree: The decision tree to visualize
        decision_path: Optional list of decision steps taken through the tree
        output_format: Output format ('json', 'mermaid', 'dot', 'svg', 'png')

    Returns:
        Visualization in the requested format
    """
    if decision_path is None:
        decision_path = []

    # Map of visited nodes and decisions
    visited_nodes = {}
    for step in decision_path:
        node_id = step.get("node_id")
        if node_id:
            visited_nodes[node_id] = {
                "decision": step.get("decision"),
                "confidence": step.get("confidence")
            }

    # Generate appropriate format
    if output_format == "json":
        return _generate_json_visualization(tree, visited_nodes, decision_path)
    elif output_format in ["dot", "svg", "png"]:
        return _generate_graphviz_visualization(tree, visited_nodes, decision_path, output_format)
    elif output_format == "mermaid":
        return _generate_mermaid_diagram(tree, visited_nodes)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def _generate_json_visualization(
    tree: DecisionTree,
    visited_nodes: Dict[str, Dict[str, Any]],
    decision_path: List[Dict[str, Any]]
) -> TreeVisualization:
    """Generate JSON visualization data"""
    # Create node info dictionary
    nodes = {}
    for node_id, node in tree.nodes.items():
        visit_info = visited_nodes.get(node_id, {})

        nodes[node_id] = NodeInfo(
            id=node_id,
            description=node.description,
            is_leaf=node.is_leaf,
            is_probabilistic=node.is_probabilistic,
            children=node.children,
            visited=node_id in visited_nodes,
            confidence=visit_info.get("confidence"),
            decision=visit_info.get("decision"),
            metadata={
                "domain": node.domain,
                "requires_context": node.requires_context,
                "confidence_threshold": node.confidence_threshold,
                "fallback_node_id": node.fallback_node_id
            }
        )

    # Create edges list
    edges = []
    for node_id, node in tree.nodes.items():
        for decision, child_id in node.children.items():
            # Calculate edge properties based on decision path
            is_path_edge = False
            confidence = None

            for i, step in enumerate(decision_path):
                if step.get("node_id") == node_id and step.get("decision") == decision:
                    is_path_edge = True
                    confidence = step.get("confidence")
                    break

            edge_style = "solid" if is_path_edge else "dashed"
            edge_width = 2 if is_path_edge else 1

            edges.append({
                "source": node_id,
                "target": child_id,
                "label": decision,
                "visited": is_path_edge,
                "confidence": confidence,
                "style": edge_style,
                "width": edge_width
            })

        # Add fallback edge if present
        if node.fallback_node_id:
            edges.append({
                "source": node_id,
                "target": node.fallback_node_id,
                "label": "fallback",
                "style": "dotted",
                "width": 1
            })

    return TreeVisualization(
        name=tree.name,
        description=tree.description,
        root_id=tree.root_node_id,
        nodes=nodes,
        edges=edges,
        metadata={
            "domain": tree.domain,
            "version": tree.version,
            "max_paths_to_explore": tree.max_paths_to_explore
        },
        decision_path=decision_path
    )

def _generate_graphviz_visualization(
    tree: DecisionTree,
    visited_nodes: Dict[str, Dict[str, Any]],
    decision_path: List[Dict[str, Any]],
    output_format: str = "dot"
) -> Union[str, bytes]:
    """Generate GraphViz visualization"""
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("GraphViz is not available. Install with 'pip install graphviz'")

    # Create digraph
    dot = graphviz.Digraph(
        name=tree.id,
        comment=tree.description,
        format=output_format if output_format != "dot" else None
    )

    # Set graph attributes
    dot.attr('graph',
        rankdir='TB',
        fontname='Arial',
        label=f'"{tree.name}\\n{tree.description}"',
        labelloc='t',
        concentrate='true'
    )

    # Set default node and edge attributes
    dot.attr('node',
        shape='box',
        style='filled',
        fontname='Arial',
        margin='0.2,0.1'
    )
    dot.attr('edge',
        fontname='Arial',
        fontsize='10'
    )

    # Add nodes
    for node_id, node in tree.nodes.items():
        # Determine node style based on type and visit status
        node_shape = 'ellipse' if node.is_leaf else 'box'
        node_style = 'filled,rounded'

        # Set color based on visit status
        fillcolor = '#f5f5f5'  # default light gray
        fontcolor = '#000000'  # default black
        penwidth = '1'

        # Check if node was visited
        if node_id in visited_nodes:
            visit_info = visited_nodes[node_id]
            confidence = visit_info.get("confidence", 0.5)

            # Color based on confidence
            if confidence > 0.8:
                fillcolor = '#d4f1d4'  # light green
            elif confidence > 0.5:
                fillcolor = '#d4e6f1'  # light blue
            else:
                fillcolor = '#f9e6e6'  # light red

            penwidth = '2'

        # Special styling for probabilistic nodes
        if node.is_probabilistic:
            node_style += ',dashed'

        # Special styling for leaf nodes
        if node.is_leaf:
            fillcolor = '#f1e8b8'  # light yellow
            node_shape = 'ellipse'

        # Create label with description
        label_parts = [f'"{node.description}"']

        # Add node to graph
        dot.node(
            node_id,
            label='\n'.join(label_parts),
            shape=node_shape,
            style=node_style,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            penwidth=penwidth
        )

    # Add edges
    for node_id, node in tree.nodes.items():
        for decision, child_id in node.children.items():
            # Determine if this edge is part of the decision path
            is_path_edge = False
            confidence = None

            for step in decision_path:
                if step.get("node_id") == node_id and step.get("decision") == decision:
                    is_path_edge = True
                    confidence = step.get("confidence")
                    break

            # Set edge properties
            edge_style = 'solid' if is_path_edge else 'dashed'
            edge_color = '#4285F4' if is_path_edge else '#888888'  # blue if visited, gray otherwise
            edge_penwidth = '2.0' if is_path_edge else '1.0'

            # Add confidence to label if available
            edge_label = decision
            if is_path_edge and confidence is not None:
                conf_pct = int(confidence * 100)
                edge_label = f"{decision} ({conf_pct}%)"

            dot.edge(
                node_id,
                child_id,
                label=edge_label,
                style=edge_style,
                color=edge_color,
                penwidth=edge_penwidth
            )

        # Add fallback edge if present
        if node.fallback_node_id:
            dot.edge(
                node_id,
                node.fallback_node_id,
                label="fallback",
                style="dotted",
                color="#FF0000",  # red for fallback
                penwidth="1.0"
            )

    # Return in requested format
    if output_format == "dot":
        return dot.source
    else:
        # For SVG and PNG, render and return bytes
        return dot.pipe()

def _generate_mermaid_diagram(
    tree: DecisionTree,
    visited_nodes: Dict[str, Dict[str, Any]]
) -> str:
    """Generate a Mermaid.js diagram of a decision tree"""
    # Start the diagram
    diagram = ["flowchart TD"]

    # Add title
    diagram.append(f'    title["{tree.name}"]')
    diagram.append('    style title fill:#f9f9f9,stroke:#333,stroke-width:1px')

    # Define nodes
    for node_id, node in tree.nodes.items():
        # Define node style based on type and visit status
        style = "style " + node_id + " "

        # Check if node was visited
        if node_id in visited_nodes:
            visit_info = visited_nodes[node_id]
            confidence = visit_info.get("confidence", 0.5)

            # Color based on confidence
            if confidence > 0.8:
                style += "fill:#d4f1d4,stroke:#28a745,stroke-width:2px"  # green
            elif confidence > 0.5:
                style += "fill:#d4e6f1,stroke:#0077be,stroke-width:2px"  # blue
            else:
                style += "fill:#f9e6e6,stroke:#dc3545,stroke-width:2px"  # red
        else:
            style += "fill:#f5f5f5,stroke:#333,stroke-width:1px"  # gray

        # Special styling for node types
        if node.is_leaf:
            style += ",border-radius:10px"
        if node.is_probabilistic:
            style += ",stroke-dasharray:5 5"

        # Create node label
        node_shape = "(" if node.is_leaf else "["
        node_shape_end = ")" if node.is_leaf else "]"

        # Add probability if available
        prob_text = ""
        if node_id in visited_nodes:
            conf = visited_nodes[node_id].get("confidence")
            if conf is not None:
                prob_text = f"<br>Confidence: {conf:.2f}"

        # Add node definition
        diagram.append(f'    {node_id}{node_shape}"{node.description}{prob_text}"{node_shape_end}')
        diagram.append(f'    {style}')

    # Define edges
    for node_id, node in tree.nodes.items():
        for decision, child_id in node.children.items():
            # Check if this edge is part of decision path
            is_path_edge = False
            for step in visited_nodes.values():
                if step.get("decision") == decision:
                    is_path_edge = True
                    break

            # Style edge based on path status
            if is_path_edge:
                diagram.append(f'    {node_id} -->|"{decision}"| {child_id}')
            else:
                diagram.append(f'    {node_id} -.->|"{decision}"| {child_id}')

        # Add fallback edge if present
        if node.fallback_node_id:
            diagram.append(f'    {node_id} -.->|"fallback"| {node.fallback_node_id}')

    return "\n".join(diagram)

@router.get("/trees")
async def list_trees():
    """List all available decision trees"""
    try:
        trees = get_available_trees()
        tree_list = [
            {
                "id": tree_id,
                "name": tree.name,
                "description": tree.description,
                "node_count": len(tree.nodes),
                "domain": tree.domain,
                "version": tree.version
            }
            for tree_id, tree in trees.items()
        ]

        return {"trees": tree_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing trees: {str(e)}")

@router.get("/tree/{tree_id}/visualization")
async def get_tree_visualization(tree_id: str, format: str = "json"):
    """Get visualization data for a specific tree"""
    try:
        # Get the requested tree
        trees = get_available_trees()
        if tree_id in trees:
            tree = trees[tree_id]
        else:
            raise HTTPException(status_code=404, detail=f"Tree '{tree_id}' not found")

        # Generate visualization in requested format
        viz = generate_tree_visualization(tree, output_format=format)

        if format == "json":
            return viz
        elif format == "mermaid":
            return {"mermaid": viz}
        elif format == "svg":
            return Response(content=viz, media_type="image/svg+xml")
        elif format == "png":
            return Response(content=viz, media_type="image/png")
        elif format == "dot":
            return {"dot": viz}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@router.post("/visualize")
async def visualize_with_path(
    tree: Dict[str, Any],
    decision_path: List[Dict[str, Any]] = [],
    format: str = "json"
):
    """Visualize a tree with a decision path"""
    try:
        # Convert dictionary to DecisionTree
        from .tree_reasoning import DecisionTree, DecisionNode

        nodes = {}
        for node_id, node_data in tree.get("nodes", {}).items():
            nodes[node_id] = DecisionNode(
                id=node_data.get("id", node_id),
                description=node_data.get("description", ""),
                prompt=node_data.get("prompt", ""),
                children=node_data.get("children", {}),
                is_leaf=node_data.get("is_leaf", False),
                is_probabilistic=node_data.get("is_probabilistic", False),
                action=node_data.get("action"),
                response_template=node_data.get("response_template"),
                confidence_threshold=node_data.get("confidence_threshold", 0.7),
                domain=node_data.get("domain"),
                fallback_node_id=node_data.get("fallback_node_id")
            )

        tree_obj = DecisionTree(
            id=tree.get("id", "custom"),
            name=tree.get("name", "Custom Tree"),
            description=tree.get("description", ""),
            root_node_id=tree.get("root_node_id", "root"),
            nodes=nodes,
            version=tree.get("version", "1.0"),
            domain=tree.get("domain"),
            metadata=tree.get("metadata", {})
        )

        # Generate visualization
        viz = generate_tree_visualization(tree_obj, decision_path, format)

        if format == "json":
            return viz
        elif format == "mermaid":
            return {"mermaid": viz}
        elif format == "svg":
            return Response(content=viz, media_type="image/svg+xml")
        elif format == "png":
            return Response(content=viz, media_type="image/png")
        elif format == "dot":
            return {"dot": viz}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")
