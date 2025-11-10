use std::collections::{HashMap, HashSet};
use std::fmt;

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
}

#[derive(Serialize)]
struct Graph {
    elements: Vec<serde_json::Value>,
}

///
/// By implementing this trait, one can adjust how information
/// about terms, atoms, modules, etc. is dumped to HTML.
pub trait Descriptor<T, I> {
    fn describe_module(&self, module: &Module<T, I>) -> String;
    fn describe_atom(&self, atom: &Atom<T, I>) -> String;
    fn describe_term(&self, term: &Term<T, I>) -> String;

    fn describe_wire_id(&self, id: usize) -> String {
        format!("w{id}")
    }

    // describe node representing output. `id` is the identifier
    // of the input wire
    fn describe_input(&self, _id: usize) -> String {
        "(input node)".into()
    }
    // describe node representing output. `id` is the identifier
    // of the output wire
    fn describe_output(&self, _id: usize) -> String {
        "(output node)".into()
    }
}

///
/// This is the default Descriptor, it simply calls `display()`
/// on items.
struct DefaultDescriptor {}
impl<T, I> Descriptor<T, I> for DefaultDescriptor
where
    Module<T, I>: std::fmt::Display,
    Atom<T, I>: std::fmt::Display,
    Term<T, I>: std::fmt::Display,
{
    fn describe_module(&self, module: &Module<T, I>) -> String {
        module.to_string()
    }

    fn describe_atom(&self, atom: &Atom<T, I>) -> String {
        atom.to_string()
    }

    fn describe_term(&self, term: &Term<T, I>) -> String {
        term.to_string()
    }
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
            description: descr.describe_module(module),
            parent: Some(module_name.to_string()),
        },
        classes: Some("module".into()),
    });

    for (atom_id, atom) in module.atoms().iter().enumerate() {
        // Add a node for the whole atom
        nodes.push(Node {
            data: NodeData {
                id: format!("atom.{atom_id}"),
                label: format!("Atom {atom_id}"),
                description: descr.describe_atom(atom),
                parent: Some(format!("module.{module_name}")),
            },
            classes: Some("atom".into()),
        });

        // Add a node for the atom init terms and update terms
        let atom_id_init = format!("atom.{atom_id}.init");
        nodes.push(Node {
            data: NodeData {
                id: atom_id_init.clone(),
                label: "Init".into(),
                description: "atom init section".into(),
                parent: Some(format!("atom.{atom_id}")),
            },
            classes: Some("atom-init".into()),
        });
        let atom_id_update = format!("atom.{atom_id}.update");
        nodes.push(Node {
            data: NodeData {
                id: atom_id_update.clone(),
                label: "Update".into(),
                description: "atom update section".into(),
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
            },
        });

        // Add the atom init terms
        for term in atom.init() {
            let id = nodes.len();
            nodes.push(Node {
                data: NodeData {
                    id: format!("term.{id}"),
                    label: term.itype().to_string(),
                    description: descr.describe_term(term),
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
                    label: term.itype().to_string(),
                    description: descr.describe_term(term),
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

    let wires: HashSet<usize> = HashSet::from_iter(
        wire_written_by.keys().chain(wire_read_by.keys()).copied(), //.map(|x| *x),
    );

    for wire in wires {
        let wire_name = descr.describe_wire_id(wire);

        if let (Some(srcs), Some(dests)) = (wire_written_by.get(&wire), wire_read_by.get(&wire)) {
            for src in srcs {
                for dst in dests {
                    edges.push(Edge {
                        data: EdgeData {
                            id: format!("wire.{}.{}.{}", wire, *src, *dst),
                            source: format!("term.{}", *src),
                            target: format!("term.{}", *dst),
                            label: wire_name.clone(),
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
                        label: wire_name.clone(),
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
                    },
                });
            }
        } else if let Some(dsts) = wire_read_by.get(&wire) {
            // wires only read, ie, external inputs
            for dst in dsts {
                let nd = &nodes[*dst];
                let parent = nd.data.parent.as_ref().unwrap().clone();

                // create new "input" node
                let id = nodes.len();
                let id_str = format!("input.{id}");
                nodes.push(Node {
                    data: NodeData {
                        id: id_str.clone(),
                        label: wire_name.clone(),
                        description: descr.describe_input(wire),
                        // TODO: we could set the parent to be the parent of this parent
                        parent: Some(parent),
                    },
                    classes: Some("input".into()),
                });
                edges.push(Edge {
                    data: EdgeData {
                        id: format!("wire.{}..{}", wire, *dst),
                        source: id_str.clone(),
                        target: format!("term.{}", *dst),
                        label: wire_name.clone(),
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

    let data = if descr.is_some() {
        module_to_graph(module, descr.unwrap())
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
  <!-- script src="https://unpkg.com/cytoscape@3.30.0/dist/cytoscape.min.js"></script-->
  <!-- script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script-->

  <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/layout-base/layout-base.js"></script>
  <script src="https://unpkg.com/cose-base/cose-base.js"></script>
  <script src="https://unpkg.com/cytoscape-fcose/cytoscape-fcose.js"></script>


  <style>
    body {{
      margin: 0;
      display: flex;
      height: 100vh;
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
      margin-top: 0;
      font-size: 18px;
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

//cytoscape.use(cytoscapeFcose);

const graphData = {json};
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
    }}}},
    {{ selector: 'node.output', style: {{
        'background-color': '#fe77b5',
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
        'target-arrow-shape': 'triangle'
    }}}},
    {{ selector: 'edge:selected', style: {{
        'line-color': 'red',

    }}}},
    {{ selector: 'node:selected', style: {{
        'border-width': 3,
        'border-color': '#ff6600'
    }}}}
  ],
  //layout: {{ name: 'breadthfirst', directed: true,  spacingFactor: 0.5, padding: 3, avoidOverlap: true, nodeDimensionsIncludeLabels: true, animate: false }},
  layout: {{ name: 'fcose', nodeRepulsion: 4500, avoidOverlap: true, nodeDimensionsIncludeLabels: true, animate: false, gravity: 0.5, gravityRange: 5, nodeSeparation: 80, fit: true}}
}});

// --- Info Panel Logic ---
const infoDiv = document.getElementById('info');

cy.on('tap', 'node', (evt) => {{
  const node = evt.target;
  const data = node.data();

  infoDiv.innerHTML = `
    <h2>${{data.label}}</h2>
    <!--p><strong>Type:</strong> ${{data.type || 'Node'}}</p>
    <p><strong>ID:</strong> ${{data.id}}</p-->
    <p>${{data.description || 'No description available.'}}</p>
  `;
}});

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
