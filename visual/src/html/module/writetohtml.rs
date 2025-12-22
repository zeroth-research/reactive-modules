use std::collections::{HashMap, HashSet};
use std::fmt;

use super::{DefaultDescriptor, DescriptionContext, Descriptor};

use base::{Atom, Module, Term};

use serde::Serialize;
use std::fs::File;
use std::io::Write as IOWrite;

#[derive(Serialize)]
struct Node {
    data: NodeData,
    #[serde(skip_serializing_if = "Option::is_none")]
    classes: Option<String>,
}

#[derive(Serialize)]
struct NodeData {
    id: String,
    label: String,
    description: String,
    parent: Option<String>,
}

#[derive(Serialize)]
struct Edge {
    data: EdgeData,
}

#[derive(Serialize)]
struct EdgeData {
    id: String,
    source: String,
    target: String,
    label: String,
    description: String,
}

#[derive(Serialize)]
struct Graph {
    elements: Vec<serde_json::Value>,
}

fn module_to_graph<T, I, D: Descriptor<T, I>>(module: &Module<T, I>, descr: &D) -> Graph
where
    T: fmt::Display,
    I: fmt::Display,
{
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    // what wire is written by what node
    let mut wire_written_by: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut wire_read_by: HashMap<usize, Vec<usize>> = HashMap::new();

    let module_name = "<name not found>";

    nodes.push(Node {
        data: NodeData {
            id: format!("module.{module_name}"),
            label: format!("Module {module_name}"),
            description: descr.describe_module(module, DescriptionContext::Standalone),
            parent: Some(module_name.to_string()),
        },
        classes: Some("module".into()),
    });

    for (atom_id, atom) in module.atoms().iter().enumerate() {
        // Add a node for the whole atom
        nodes.push(Node {
            data: NodeData {
                id: format!("atom.{atom_id}"),
                label: format!("Atom {atom_id}"), // TODO
                description: descr.describe_atom(atom, DescriptionContext::Standalone),
                parent: Some(format!("module.{module_name}")),
            },
            classes: Some("atom".into()),
        });

        // Add a node for the atom init terms and update terms
        let atom_id_init = format!("atom.{atom_id}.init");
        nodes.push(Node {
            data: NodeData {
                id: atom_id_init.clone(),
                label: descr.describe_atom_section(atom, "init", DescriptionContext::Node),
                description: descr.describe_atom_section(
                    atom,
                    "init",
                    DescriptionContext::Standalone,
                ),
                parent: Some(format!("atom.{atom_id}")),
            },
            classes: Some("atom-init".into()),
        });
        let atom_id_update = format!("atom.{atom_id}.update");
        nodes.push(Node {
            data: NodeData {
                id: atom_id_update.clone(),
                label: descr.describe_atom_section(atom, "update", DescriptionContext::Node),
                description: descr.describe_atom_section(
                    atom,
                    "update",
                    DescriptionContext::Standalone,
                ),
                parent: Some(format!("atom.{atom_id}")),
            },
            classes: Some("atom-update".into()),
        });
        // add an edge between init and update to order them
        edges.push(Edge {
            data: EdgeData {
                id: format!("order.atom.{atom_id}"),
                source: atom_id_init.clone(),
                target: atom_id_update.clone(),
                label: "".into(),
                description: "".into(),
            },
        });

        // Add the atom init terms
        for term in atom.init() {
            let id = nodes.len();
            nodes.push(Node {
                data: NodeData {
                    id: format!("term.{id}"),
                    label: descr.describe_term(term, DescriptionContext::Node),
                    description: descr.describe_term(term, DescriptionContext::Standalone),
                    parent: Some(atom_id_init.clone()),
                },
                classes: Some("term-init".into()),
            });

            // gather information for creating edges
            for wire in term.write().ids() {
                wire_written_by.entry(wire).or_default().push(id);
            }
            for wire in term.read().ids() {
                wire_read_by.entry(wire).or_default().push(id);
            }
        }

        // Add the atom update terms
        for term in atom.update() {
            let id = nodes.len();
            nodes.push(Node {
                data: NodeData {
                    id: format!("term.{id}"),
                    label: descr.describe_term(term, DescriptionContext::Node),
                    description: descr.describe_term(term, DescriptionContext::Standalone),
                    parent: Some(atom_id_update.clone()),
                },
                classes: Some("term-update".into()),
            });

            // gather information for creating edges
            for wire in term.write().ids() {
                wire_written_by.entry(wire).or_default().push(id);
            }
            for wire in term.read().ids() {
                wire_read_by.entry(wire).or_default().push(id);
            }
        }
    }

    let wires: HashSet<usize> =
        HashSet::from_iter(wire_written_by.keys().chain(wire_read_by.keys()).copied());

    // map wire ids to nodes of already existing input values
    let mut input_nodes: HashMap<usize, String> = HashMap::new();
    for wire in wires {
        let wire_name = descr.describe_wire_id(wire, DescriptionContext::Edge);
        let wire_descr = descr.describe_wire(wire, DescriptionContext::Edge);

        if let (Some(srcs), Some(dests)) = (wire_written_by.get(&wire), wire_read_by.get(&wire)) {
            for src in srcs {
                for dst in dests {
                    edges.push(Edge {
                        data: EdgeData {
                            id: format!("wire.{}.{}.{}", wire, *src, *dst),
                            source: format!("term.{}", *src),
                            target: format!("term.{}", *dst),
                            label: wire_name.clone(),
                            description: wire_descr.clone(),
                        },
                    });
                }
            }
        } else if let Some(srcs) = wire_written_by.get(&wire) {
            // wires only written to, ie., interface output
            for src in srcs {
                let nd = &nodes[*src];
                let parent = nd.data.parent.as_ref().unwrap().clone();

                // create new "output" node
                let id = nodes.len();
                let id_str = format!("output.{id}");
                nodes.push(Node {
                    data: NodeData {
                        id: id_str.clone(),
                        label: descr.describe_wire_id(wire, DescriptionContext::Node),
                        description: descr.describe_output(wire),
                        // TODO: we could set the parent to be the parent of this parent
                        parent: Some(parent),
                    },
                    classes: Some("output".into()),
                });
                edges.push(Edge {
                    data: EdgeData {
                        id: format!("wire.{}.{}.", wire, *src),
                        source: format!("term.{}", *src),
                        target: id_str.clone(),
                        label: wire_name.clone(),
                        description: wire_descr.clone(),
                    },
                });
            }
        } else if let Some(dsts) = wire_read_by.get(&wire) {
            // wires only read, ie, external inputs
            let mut id_str: String;
            for dst in dsts {
                if let Some(node) = input_nodes.get(&wire) {
                    id_str = node.clone();
                } else {
                    let nd = &nodes[*dst];
                    let parent = nd.data.parent.as_ref().unwrap().clone();

                    // create new "input" node
                    let id = nodes.len();
                    id_str = format!("input.{id}");
                    nodes.push(Node {
                        data: NodeData {
                            id: id_str.clone(),
                            label: descr.describe_wire_id(wire, DescriptionContext::Node),
                            description: descr.describe_input(wire),
                            // TODO: we could set the parent to be the parent of this parent
                            parent: Some(parent),
                        },
                        classes: Some("input".into()),
                    });
                    input_nodes.insert(wire, id_str.clone());
                }
                edges.push(Edge {
                    data: EdgeData {
                        id: format!("wire.{}..{}", wire, *dst),
                        source: id_str.clone(),
                        target: format!("term.{}", *dst),
                        label: wire_name.clone(),
                        description: wire_descr.clone(),
                    },
                });
            }
        }
    }

    let mut elements: Vec<serde_json::Value> = Vec::with_capacity(nodes.len() + edges.len());
    elements.extend(nodes.into_iter().map(|n| serde_json::to_value(n).unwrap()));
    elements.extend(edges.into_iter().map(|e| serde_json::to_value(e).unwrap()));

    Graph { elements }
}

pub fn write_to_html<T, I, D: Descriptor<T, I>>(
    module: &Module<T, I>,
    path: &str,
    descr: Option<&D>,
) -> Result<(), std::io::Error>
where
    Module<T, I>: std::fmt::Display,
    Atom<T, I>: std::fmt::Display,
    Term<T, I>: std::fmt::Display,
    I: std::fmt::Display,
    T: std::fmt::Display,
{
    //let palette = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500"];

    let data = if let Some(descriptor) = descr {
        module_to_graph(module, descriptor)
    } else {
        module_to_graph(module, &DefaultDescriptor {})
    };

    // Serialize into vis-network JSON
    let json = serde_json::to_string_pretty(&data).unwrap();

    let module_name = "<unnamed>";
    //let module_dump = module.to_string();

    let html = format!(
        r#"
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Module {module_name}</title>

  <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/layout-base/layout-base.js"></script>
  <script src="https://unpkg.com/cose-base/cose-base.js"></script>
  <script src="https://unpkg.com/cytoscape-fcose/cytoscape-fcose.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>

  <style>
    body {{
      margin: 0;
      display: flex;
      min-height: 100vh;
      font-family: sans-serif;
    }}
    #cy {{
      flex: 1;
      border-right: 1px solid #ccc;
    }}
    #info {{
      width: 400px;
      padding: 15px;
      box-sizing: border-box;
      background: #f8f8f8;
    }}
    #info h2 {{
      margin-top: 20px;
      font-size: 20px;
      color: #333;
    }}
    #info h3 {{
      margin-top: 20px;
      font-size: 18px;
      color: #333;
    }}
    #info h4 {{
      margin-top: 20px;
      font-size: 16px;
      color: #333;
    }}
    #info strong {{
      font-size: 16px;
      color: #333;
    }}
    #info p {{
      margin: 5px 0;
      color: #555;
    }}
  </style>
</head>
<body>

<div id="cy"></div>
<div id="info">
  <h2>Node Information</h2>
  <p>Click any node or cluster to see details here.</p>
</div>

<script>

const graphData = {json};

// Register the dagre layout
cytoscape.use(cytoscapeDagre);

// --- Cytoscape Setup ---
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: graphData.elements,
  style: [
    {{ selector: 'node', style: {{
        'label': 'data(label)',
        'text-valign': 'center',
        'color': '#222',
        'background-color': '#77b5fe',
        'font-size': 12,
        'width': 'label',
        'height': 'label',
        'shape': 'rectangle',
        'padding': '1em',
    }}}},
    {{ selector: 'node.input', style: {{
        'background-color': '#77feb5',
        'shape': 'ellipse',
    }}}},
    {{ selector: 'node.output', style: {{
        'background-color': '#fe77b5',
        'shape': 'ellipse',
    }}}},
    {{ selector: ':parent', style: {{
        'label': 'data(label)',
        'text-valign': 'top',
        'text-halign': 'center',
        'background-opacity': 0.1,
        'border-width': 2,
        'border-color': '#666',
        'shape': 'round-rectangle',
        'padding': '25px',
        'font-weight': 'bold',
        'font-size': 14
    }}}},
    {{ selector: 'edge', style: {{
        'width': 2,
        'label': 'data(label)',
        'line-color': '#aaa',
        'text-background-color': 'white',
        'text-background-opacity': 0.7,
        'text-background-shape': 'roundrectangle',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'edge-text-rotation': 'autorotate'
    }}}},
    {{ selector: 'edge[id^="order."]', style: {{
        'opacity': 0,
        'width': 0,
        'label': ''
    }}}},
    {{ selector: 'edge:selected', style: {{
        'line-color': 'red',
    }}}},
    {{ selector: 'node:selected', style: {{
        'border-width': 3,
        'border-color': '#ff6600'
    }}}}
  ],
  layout: {{ 
    name: 'dagre',
    rankDir: 'LR',  // Left to Right
    nodeSep: 30,    // Vertical separation between unconnected nodes
    edgeSep: 30,    // Edge separation
    rankSep: 100,   // Horizontal separation between ranks (wire length/distance between terms)
    ranker: 'network-simplex',
    animate: false,
    fit: true,
    padding: 20
  }}
}});

// --- Info Panel Logic ---
const infoDiv = document.getElementById('info');

cy.on('tap', 'node', (evt) => {{
    const node = evt.target;
    const data = node.data();

    // Use the full HTML description provided by the descriptor when
    // available. Fall back to a simple heading if no description is set.
    if (data.description && data.description.length > 0) {{
        infoDiv.innerHTML = data.description;
    }} else {{
        infoDiv.innerHTML = `
            <h2>${{data.label}}</h2>
            <p>No description available.</p>
        `;
    }}
}});

/// We do not have anything interesting to show about wires now
///cy.on('tap', 'edge', (evt) => {{
///    const edge = evt.target;
///    const data = edge.data();
///
///    // Use the full HTML description provided by the descriptor when
///    // available. Fall back to a simple heading if no description is set.
///    if (data.description && data.description.length > 0) {{
///        infoDiv.innerHTML = data.description;
///    }} else {{
///        infoDiv.innerHTML = `
///            <h2>${{data.label}}</h2>
///            <p>No description available.</p>
///        `;
///    }}
///}});



cy.on('tap', (evt) => {{
  // Click on background → reset info panel
  if (evt.target === cy) {{
    infoDiv.innerHTML = `
      <h2>Node Information</h2>
      <p>Click any node or cluster to see details here.</p>
    `;
  }}
}});
</script>

</body>
</html>
"#
    );

    File::create(path)?.write_all(html.as_bytes())?;
    Ok(())
}
