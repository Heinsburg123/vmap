"""
dag_viz.py
----------
Visualise the ancestor DAG of one or more RVs from pangolin.ir.

DFS walks rv.parents fully — Index nodes are included as real nodes,
nothing is unwrapped or skipped.

Usage (notebook):
    from dag_viz import draw_dag
    draw_dag(my_rv)                        # single RV
    draw_dag([rv_a, rv_b], label_fn=...)   # multiple roots, custom labels

Usage (plain script):
    from dag_viz import draw_dag_matplotlib
    draw_dag_matplotlib(my_rv)
"""

from collections import deque

# ── colour scheme ──────────────────────────────────────────────────────────────
_OP_COLORS = {
    "Constant":  "#3498db",   # blue
    "Normal":    "#2ecc71",   # green
    "Bernoulli": "#e67e22",   # orange
    "Beta":      "#9b59b6",   # purple
    "Gamma":     "#1abc9c",   # teal
    "VMap":      "#e74c3c",   # red
    "Index":     "#95a5a6",   # grey
}
_DEFAULT_COLOR = "#f39c12"    # yellow-orange for unknown ops


# ── helpers ────────────────────────────────────────────────────────────────────

def _node_id(rv):
    """Stable string id using the internal _n counter."""
    return str(rv._n)


def _default_label(rv):
    """Human-readable label: op name + shape + node id.
    For Constant nodes, also shows the value."""
    shape_str = str(rv.shape) if rv.shape else "()"
    if rv.op.name == "Constant":
        val = rv.op.value
        try:
            import numpy as np
            arr = np.asarray(val)
            if arr.size <= 6:
                val_str = str(arr.tolist())
            else:
                val_str = f"[{arr.flat[0]:.3g}...] shape={arr.shape}"
        except Exception:
            val_str = str(val)
        return f"Constant\n{val_str}\n#{rv._n}"
    return f"{rv.op.name}\n{shape_str}\n#{rv._n}"


def _color(rv):
    return _OP_COLORS.get(rv.op.name, _DEFAULT_COLOR)


# ── DFS collector ──────────────────────────────────────────────────────────────

def collect_dag(rvs):
    """
    DFS upward from *rvs* (a single RV or a list of RVs).
    Every node including Index nodes is kept as-is — nothing is unwrapped.

    Returns
    -------
    nodes : set of RV
    edges : list of (parent_rv, child_rv)   — directed parent → child
    """
    if not isinstance(rvs, (list, tuple)):
        rvs = [rvs]

    nodes    = set()
    edges    = []
    edge_set = set()
    visited  = set()

    def dfs(rv):
        for p in rv.parents:
            edge_key = (_node_id(p), _node_id(rv))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append((p, rv))

            nodes.add(p)

            if p not in visited:
                visited.add(p)
                dfs(p)

    for rv in rvs:
        nodes.add(rv)
        visited.add(rv)
        dfs(rv)

    return nodes, edges


# ── pyvis (notebook) ──────────────────────────────────────────────────────────

def draw_dag(
    rvs,
    label_fn=None,
    height="700px",
    dark=True,
):
    """
    Draw the full ancestor DAG inline in a Jupyter notebook using pyvis.

    Parameters
    ----------
    rvs      : RV or list[RV]   – the target RV(s) to trace upward from
    label_fn : callable or None – label_fn(rv) -> str; defaults to op+shape+id
    height   : str              – iframe height e.g. '700px'
    dark     : bool             – dark background theme
    """
    try:
        from pyvis.network import Network
        from IPython.display import display, HTML
    except ImportError:
        raise ImportError("pip install pyvis  (and run in a Jupyter notebook)")

    label_fn = label_fn or _default_label
    nodes, edges = collect_dag(rvs)

    bg = "#1a1a2e" if dark else "#ffffff"
    fc = "white"   if dark else "black"

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor=bg,
        font_color=fc,
        notebook=True,
        cdn_resources="in_line",
    )

    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "levelSeparation": 110,
          "nodeSpacing": 150,
          "treeSpacing": 180
        }
      },
      "physics": { "enabled": false },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "smooth": { "type": "cubicBezier", "forceDirection": "vertical" },
        "color": { "color": "#aaaaaa" }
      },
      "nodes": {
        "shape": "box",
        "borderWidth": 2,
        "font": { "size": 12, "face": "monospace", "multi": true }
      }
    }
    """)

    for rv in nodes:
        c = _color(rv)
        net.add_node(
            _node_id(rv),
            label=label_fn(rv),
            color={"background": c, "border": "white",
                   "highlight": {"background": "white", "border": c}},
            font={"color": fc, "size": 12},
            shape="box",
        )

    for parent, child in edges:
        net.add_edge(_node_id(parent), _node_id(child))

    display(HTML(net.generate_html()))


# ── matplotlib (plain script / fallback) ──────────────────────────────────────

def draw_dag_matplotlib(
    rvs,
    label_fn=None,
    figsize=(18, 12),
    x_spacing=3.5,
    y_spacing=3.0,
):
    """
    Draw the full ancestor DAG with matplotlib (works in scripts and notebooks).

    Parameters
    ----------
    rvs       : RV or list[RV]
    label_fn  : callable or None
    figsize   : tuple
    x_spacing : float  – horizontal gap between nodes in the same layer
    y_spacing : float  – vertical gap between layers
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import collections
    except ImportError:
        raise ImportError("pip install networkx matplotlib")

    label_fn = label_fn or _default_label
    nodes, edges = collect_dag(rvs)

    G = nx.DiGraph()
    for rv in nodes:
        G.add_node(_node_id(rv), rv=rv)
    for parent, child in edges:
        G.add_edge(_node_id(parent), _node_id(child))

    # ── layered layout (longest-path layering) ────────────────────────────────
    layer = {}
    for nid in nx.topological_sort(G):
        preds = list(G.predecessors(nid))
        layer[nid] = max((layer[p] + 1 for p in preds), default=0)

    layers = collections.defaultdict(list)
    for nid, l in layer.items():
        layers[l].append(nid)

    pos = {}
    for l, nids in layers.items():
        n = len(nids)
        for i, nid in enumerate(nids):
            x = (i - (n - 1) / 2.0) * x_spacing
            pos[nid] = (x, -l * y_spacing)

    # ── draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos,
        arrows=True, arrowsize=15,
        edge_color="#555555", width=1.2,
        connectionstyle="arc3,rad=0.1",
        node_size=0, ax=ax,
    )

    for nid, (x, y) in pos.items():
        rv = G.nodes[nid]["rv"]
        ax.text(
            x, y, label_fn(rv),
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="white",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=_color(rv),
                edgecolor="white",
                linewidth=1.5,
            ),
        )

    # ── legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    seen_ops = {G.nodes[nid]["rv"].op.name for nid in G.nodes()}
    legend = [
        Patch(color=_OP_COLORS.get(op, _DEFAULT_COLOR), label=op)
        for op in sorted(seen_ops)
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title("Ancestor DAG", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)