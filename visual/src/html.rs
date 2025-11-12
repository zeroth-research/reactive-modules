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

    /// Describe the short label to show on the node itself (used as the
    /// Cytoscape node label). By default this returns the instruction's
    /// display value (same as `term.itype().to_string()`). Implementors
    /// can override this to show a different compact label (for example,
    /// a constant's raw value).
    fn describe_term_label(&self, term: &Term<T, I>) -> String;

    /// Describe a concise single-line "flow" suitable for showing in
    /// the atom Init/Update boxes. Default implementation falls back to
    /// the term's Display.
    fn describe_term_flow(&self, _term: &Term<T, I>) -> String {
        // Default fallback when a descriptor does not implement a
        // specialized flow renderer — return an empty string so callers
        // don't accidentally display raw Debug/Display text.
        String::new()
    }

    /// Context-aware flow renderer which receives the list of all terms
    /// in the atom (init or update). The default implementation falls
    /// back to `describe_term_flow` for backward compatibility.
    fn describe_term_flow_with_context(&self, term: &Term<T, I>, _atom_terms: &[&Term<T, I>]) -> String {
        self.describe_term_flow(term)
    }

    fn describe_wire_id(&self, id: usize) -> String {
        format!("w{id}")
    }

    /// Describe the label to use on wire *edges* in the visualiser.
    ///
    /// By default this returns the internal wire name `w{id}`. Implementors
    /// can override this to provide different edge-only labels while keeping
    /// `describe_wire_id` for node labels.
    fn describe_wire_label_for_edge(&self, id: usize) -> String {
        format!("w{}", id)
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

    fn describe_term_label(&self, term: &Term<T, I>) -> String {
        // Default: use the full term display as the node label.
        term.to_string()
    }
}

fn module_to_graph<T, I, D: Descriptor<T, I>>(module: &Module<T, I>, descr: &D) -> Graph
where
    I: fmt::Display,
    //  T: fmt::Display,
{
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    // what wire is written by what node
    let mut wire_written_by: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut wire_read_by: HashMap<usize, Vec<usize>> = HashMap::new();

    let module_name = module.name().unwrap_or("");

    // Allow the descriptor to precompute any module-local state
    // (for example, SMV's descriptor uses `describe_module` to
    // populate the wire-name mapping xN / xN'). Call it early so
    // subsequent calls to `describe_wire_id` use the correct names.
    let _ = descr.describe_module(module);

    // Precompute the set of "next" wires so we can render only flows
    // that target module-next wires (xN').
    let next_wires: HashSet<usize> = module.wire()[1].iter().map(|p| p.0).collect();

    // Reserve a node for the module; we'll populate its description after
    // we've built the atom summaries so it can include a per-atom view.
    nodes.push(Node {
        data: NodeData {
            id: format!("module.{module_name}"),
            label: format!("Module {module_name}"),
            description: "".into(),
            parent: Some(module_name.to_string()),
        },
        classes: Some("module".into()),
    });
    // We'll collect per-atom descriptions later by scanning the JSON elements.

    for (atom_id, atom) in module.atoms().iter().enumerate() {
        // Add a node for the whole atom. We'll populate its description
        // after we compute the init/update flow summaries so the atom
        // panel can include both as subtitles.
    let atom_node_index = nodes.len();
        nodes.push(Node {
            data: NodeData {
                id: format!("atom.{atom_id}"),
                label: format!("Atom {atom_id}"),
                description: "".into(),
                parent: Some(format!("module.{module_name}")),
            },
            classes: Some("atom".into()),
        });
    // atom node recorded inline; we'll pick up descriptions later.

        // Add a node for the atom init terms and update terms
        let atom_id_init = format!("atom.{atom_id}.init");
        // Build a concise flows summary for init terms (context-aware)
        let init_terms: Vec<&Term<T, I>> = atom.init().iter().collect();
        let init_flows: Vec<String> = atom
            .init()
            .iter()
            .filter(|t| {
                // Only show flows that write to next wires (xN') to avoid
                // listing temporaries that are internal to the lowering.
                t.writes().iter().any(|p| next_wires.contains(&p.0))
            })
            .map(|t| descr.describe_term_flow_with_context(t, &init_terms))
            .filter(|s| !s.is_empty())
            .collect();
        let init_desc = if init_flows.is_empty() {
            "".into()
        } else {
            // Include a clear title for the init cluster
            format!("<h3>Init</h3><div>{}</div>", init_flows.join("<br>"))
        };

        nodes.push(Node {
            data: NodeData {
                id: atom_id_init.clone(),
                label: "Init".into(),
                description: init_desc,
                parent: Some(format!("atom.{atom_id}")),
            },
            classes: Some("atom-init".into()),
        });
        let atom_id_update = format!("atom.{atom_id}.update");
        // Build a concise flows summary for update terms
        let update_terms: Vec<&Term<T, I>> = atom.update().iter().collect();
        let update_flows: Vec<String> = atom
            .update()
            .iter()
            .filter(|t| {
                t.writes().iter().any(|p| next_wires.contains(&p.0))
            })
            .map(|t| descr.describe_term_flow_with_context(t, &update_terms))
            .filter(|s| !s.is_empty())
            .collect();
        let update_desc = if update_flows.is_empty() {
            "".into()
        } else {
            // Include a clear title for the update cluster
            format!("<h3>Update</h3><div>{}</div>", update_flows.join("<br>"))
        };

        // Build compact blocks without the H2 headings for embedding into
        // the atom description (so we avoid duplicated headings).
        let init_block = if init_flows.is_empty() {
            "<p><em>(none)</em></p>".into()
        } else {
            format!("<div>{}</div>", init_flows.join("<br>"))
        };
        let update_block = if update_flows.is_empty() {
            "<p><em>(none)</em></p>".into()
        } else {
            format!("<div>{}</div>", update_flows.join("<br>"))
        };

        // Build a small controls/waits/reads table for the atom using the
        // descriptor's wire naming so the atom panel shows the key ports.
        let ctrl_wires: Vec<String> = atom
            .ctrl()
            .iter()
            .map(|(w, _)| descr.describe_wire_id(w))
            .collect();
        let wait_wires: Vec<String> = atom
            .wait()
            .iter()
            .map(|(w, _)| descr.describe_wire_id(w))
            .collect();
        let read_wires: Vec<String> = atom
            .read()
            .iter()
            .map(|(w, _)| descr.describe_wire_id(w))
            .collect();

        let fmt_list = |v: Vec<String>| {
            if v.is_empty() {
                "<em>(none)</em>".into()
            } else {
                format!("{}", v.join(", "))
            }
        };

        let ctrl_html = fmt_list(ctrl_wires);
        let wait_html = fmt_list(wait_wires);
        let read_html = fmt_list(read_wires);

        let ports_table = format!(
            "<table><tr><td><strong>ctrl</strong></td><td>{}</td></tr><tr><td><strong>wait</strong></td><td>{}</td></tr><tr><td><strong>read</strong></td><td>{}</td></tr></table>",
            ctrl_html, wait_html, read_html
        );

        // Now populate the atom node description to include the Atom title,
        // a small ports table, and subtitles for Init/Update with their
        // respective content.
        let atom_description = format!(
            "<h3>Atom {}</h3>{}<h4>Init</h4>{}<h4>Update</h4>{}",
            atom_id, ports_table, init_block, update_block
        );
        nodes[atom_node_index].data.description = atom_description;

        nodes.push(Node {
            data: NodeData {
                id: atom_id_update.clone(),
                label: "Update".into(),
                description: update_desc,
                parent: Some(format!("atom.{atom_id}")),
            },
            classes: Some("atom-update".into()),
        });
        
        // Add a dummy invisible edge to hint that Init comes before Update
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
                    label: descr.describe_term_label(term),
                    description: descr.describe_term(term),
                    parent: Some(atom_id_init.clone()),
                },
                classes: Some("term-init".into()),
            });

            // gather information for creating edges
            for (wire, _) in term.writes() {
                wire_written_by.entry(wire).or_insert(Vec::new()).push(id);
            }
            for (wire, _) in term.reads() {
                wire_read_by.entry(wire).or_insert(Vec::new()).push(id);
            }
        }

        // Add the atom update terms
        for term in atom.update() {
            let id = nodes.len();
            nodes.push(Node {
                data: NodeData {
                    id: format!("term.{id}"),
                    label: descr.describe_term_label(term),
                    description: descr.describe_term(term),
                    parent: Some(atom_id_update.clone()),
                },
                classes: Some("term-update".into()),
            });

            // gather information for creating edges
            for (wire, _) in term.writes() {
                wire_written_by.entry(wire).or_insert(Vec::new()).push(id);
            }
            for (wire, _) in term.reads() {
                wire_read_by.entry(wire).or_insert(Vec::new()).push(id);
            }
        }
    }

    let wires: HashSet<usize> = HashSet::from_iter(
        wire_written_by
            .keys()
            .chain(wire_read_by.keys())
            .map(|x| *x),
    );

    for wire in wires {
        let wire_name = descr.describe_wire_id(wire);
        let edge_label = descr.describe_wire_label_for_edge(wire);

        if let (Some(srcs), Some(dests)) = (wire_written_by.get(&wire), wire_read_by.get(&wire)) {
            for src in srcs {
                for dst in dests {
                    edges.push(Edge {
                        data: EdgeData {
                            id: format!("wire.{}.{}.{}", wire, *src, *dst),
                            source: format!("term.{}", *src),
                            target: format!("term.{}", *dst),
                            label: edge_label.clone(),
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
                        label: edge_label.clone(),
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
                        label: edge_label.clone(),
                    },
                });
            }
        }
    }

    let mut elements: Vec<serde_json::Value> = Vec::with_capacity(nodes.len() + edges.len());
    elements.extend(nodes.into_iter().map(|n| serde_json::to_value(n).unwrap()));
    elements.extend(edges.into_iter().map(|e| serde_json::to_value(e).unwrap()));

    // Before returning the graph, populate the module node description
    // with a compact wire-layout table and per-atom summaries. We do
    // this here because we needed to compute init/update flows above.
    // Locate the module node by id (it was the first node we pushed).
    // Note: nodes have been moved into `elements` as JSON values; rebuild
    // the description by updating the JSON directly.
    let mut module_json = elements[0].clone();
    // Build the wire-layout table using descriptor-provided wire ids.
    // We assume `descr` implements `describe_wire_id` sensibly.
    let mut wire_table = String::new();
    wire_table.push_str("<table border=1 style=\"border-collapse:collapse;\">\n");
    wire_table.push_str("<tr><th>extl</th><th>intf</th><th>prvt</th></tr>\n");
    // row: obs (colspan 2) | prvt
    let obs_list: Vec<String> = module.extl()[0]
        .iter()
        .map(|(w, _)| descr.describe_wire_id(w))
        .collect();
    let intf_list: Vec<String> = module.intf()[0]
        .iter()
        .map(|(w, _)| descr.describe_wire_id(w))
        .collect();
    let prvt_list: Vec<String> = module.prvt()[0]
        .iter()
        .map(|(w, _)| descr.describe_wire_id(w))
        .collect();

    let obs_html = if obs_list.is_empty() { "<em>(none)</em>".into() } else { obs_list.join(", ") };
    let _intf_html = if intf_list.is_empty() { "<em>(none)</em>".into() } else { intf_list.join(", ") };
    let prvt_html = if prvt_list.is_empty() { "<em>(none)</em>".into() } else { prvt_list.join(", ") };
    wire_table.push_str(&format!("<tr><td colspan=2>{}</td><td>{}</td></tr>\n", obs_html, prvt_html));
    // row: extl | ctrl (colspan 2)
    let extl_list: Vec<String> = module.extl()[0]
        .iter()
        .map(|(w, _)| descr.describe_wire_id(w))
        .collect();
    let ctrl_list: Vec<String> = module.ctrl()[0]
        .iter()
        .map(|(w, _)| descr.describe_wire_id(w))
        .collect();
    let extl_html = if extl_list.is_empty() { "<em>(none)</em>".into() } else { extl_list.join(", ") };
    let ctrl_html = if ctrl_list.is_empty() { "<em>(none)</em>".into() } else { ctrl_list.join(", ") };
    wire_table.push_str(&format!("<tr><td>{}</td><td colspan=2>{}</td></tr>\n", extl_html, ctrl_html));
    // row: full wire list
    let wire_list: Vec<String> = module.wire()[0]
        .iter()
        .map(|(w, _)| descr.describe_wire_id(w))
        .collect();
    let wire_html = if wire_list.is_empty() { "<em>(none)</em>".into() } else { wire_list.join(", ") };
    wire_table.push_str(&format!("<tr><td colspan=3>{}</td></tr>\n", wire_html));
    wire_table.push_str("</table>\n");

    // Append per-atom summaries: iterate over elements to find only
    // the top-level atom nodes (ids like `atom.<n>`). The element
    // list also contains the atom.init / atom.update cluster nodes
    // (which start with `atom.` as well) — we must avoid duplicating
    // the content by filtering those out.
    let mut atoms_html = String::new();
    for el in elements.iter() {
        if let Some(obj) = el.as_object() {
            if let Some(data) = obj.get("data") {
                if let Some(dobj) = data.as_object() {
                    if let Some(idv) = dobj.get("id") {
                        if let Some(s) = idv.as_str() {
                            // match only top-level atom ids like `atom.0` (no
                            // `.init` or `.update` suffix). Skip cluster nodes
                            // such as `atom.0.init` and `atom.0.update`.
                            if s.starts_with("atom.") && !(s.contains(".init") || s.contains(".update")) {
                                if let Some(desc) = dobj.get("description") {
                                    atoms_html.push_str(&format!("<hr><div>{}</div>", desc.as_str().unwrap_or("")));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Build an SVG-based visualization where `ctrl` points to prvt+intf
    // and `obs` points to intf+extl. The SVG shows labeled boxes and
    // arrows; the actual variable lists are rendered below the graphic.
    // This gives a clearer visual overlap than a pure HTML table.
    let prvt_list: Vec<String> = module.prvt()[0].iter().map(|(w, _)| descr.describe_wire_id(w)).collect();
    let intf_list: Vec<String> = module.intf()[0].iter().map(|(w, _)| descr.describe_wire_id(w)).collect();
    let extl_list: Vec<String> = module.extl()[0].iter().map(|(w, _)| descr.describe_wire_id(w)).collect();

    // Build inline, comma-separated variable strings so each box shows
    // its variables on a single horizontal line.
    let prvt_inline = if prvt_list.is_empty() {
        "(none)".into()
    } else {
        prvt_list.join(", ")
    };

    let intf_inline = if intf_list.is_empty() {
        "(none)".into()
    } else {
        intf_list.join(", ")
    };

    let extl_inline = if extl_list.is_empty() {
        "(none)".into()
    } else {
        extl_list.join(", ")
    };

                let svg_block = format!(r##"
        <div style="display:flex;flex-direction:column;align-items:center;margin:0;padding:0;">
        <svg width="100%" height="120" viewBox="0 0 600 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
            <!-- We'll split the vertical area above the wire row into two bands -->
            <!-- ctrl will be centered in the top band, obs centered in the middle band -->

                <!-- ctrl and obs side-by-side: split the horizontal space into two halves -->
                <rect x="40" y="-20" width="220" height="36" rx="8" fill="#fff2cc" stroke="#b08900"/>
                <text x="150" y="4" font-size="20" text-anchor="middle" fill="#111">ctrl</text>

                <rect x="340" y="-20" width="220" height="36" rx="8" fill="#e6ffe6" stroke="#2f8f3a"/>
                <text x="450" y="4" font-size="20" text-anchor="middle" fill="#111">obs</text>

            <!-- bottom boxes: prvt, intf, extl (aligned horizontally on the same baseline) -->
            <!-- prvt: pastel red, intf: pastel yellow, extl: pastel blue -->
            <rect x="60" y="96" width="140" height="50" rx="8" fill="#ffe5e5" stroke="#d16a6a"/>
            <text x="130" y="118" font-size="20" text-anchor="middle" fill="#111">prvt</text>
            <text x="130" y="134" font-size="16" text-anchor="middle" fill="#111">{}</text>
            <rect x="230" y="96" width="140" height="50" rx="8" fill="#fff8d6" stroke="#d1b35a"/>
            <text x="300" y="118" font-size="20" text-anchor="middle" fill="#111">intf</text>
            <text x="300" y="134" font-size="16" text-anchor="middle" fill="#111">{}</text>
            <rect x="400" y="96" width="140" height="50" rx="8" fill="#d0e7ff" stroke="#6b9bd1"/>
            <text x="470" y="118" font-size="20" text-anchor="middle" fill="#111">extl</text>
            <text x="470" y="134" font-size="16" text-anchor="middle" fill="#111">{}</text>
            <!-- arrows: from ctrl center (300,56) to prvt center (130,162) and intf center (300,162) -->
            <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L10,5 L0,10 z" fill="#666"/>
                </marker>
            </defs>
                    <!-- arrows originate at bottom edge of ctrl (y=16) and hit top edge of targets (y=96) -->
                        <!-- orthogonal (axis-aligned) paths from ctrl bottom to targets -->
                        <path d="M150 16 V56 H130 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
                            <!-- route to intf offset slightly left to avoid overlapping the obs->intf path -->
                            <path d="M150 16 V56 H285 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
                <!-- arrows: from obs center (450,38) to intf center (300,162) and extl center (470,162) -->
                    <!-- arrows originate at bottom edge of obs (y=16) and hit top edge of targets (y=96) -->
                        <!-- orthogonal (axis-aligned) paths from obs bottom to targets -->
                            <!-- route to intf offset slightly right to avoid overlapping the ctrl->intf path -->
                            <path d="M450 16 V56 H315 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
                        <path d="M450 16 V56 H470 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
        </svg>
        </div>
    "##, prvt_inline, intf_inline, extl_inline);

                let wire_table = svg_block;


    let module_html = format!("<h2>Module</h2>{}<h2>Atoms</h2>{}", wire_table, atoms_html);
    if let Some(obj) = module_json.as_object_mut() {
        if let Some(data) = obj.get_mut("data") {
            if let Some(dobj) = data.as_object_mut() {
                dobj.insert("description".into(), serde_json::Value::String(module_html));
            }
        }
    }
    elements[0] = module_json;

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
{
    //let palette = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500"];

    let data = if descr.is_some() {
        module_to_graph(module, descr.unwrap())
    } else {
        module_to_graph(module, &DefaultDescriptor {})
    };

    // Serialize into vis-network JSON
    let json = serde_json::to_string_pretty(&data).unwrap();

    let module_name = module.name().unwrap_or("<unnamed>");
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