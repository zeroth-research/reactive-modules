use crate::{dtype::DType, itype::IType, itype::Val, itype::ArithOp, itype::LogicalOp, itype::CmpOp};
use base::{atom::Atom, module::Module, term::Term};
use visual::html::{DescriptionContext, Descriptor};
use std::cell::RefCell;
use std::collections::HashMap;


pub struct Context {
    wire_names: RefCell<HashMap<usize, String>>,
    wire_dtypes: HashMap<usize, DType>,
}

impl Context {
    pub fn new(module: &Module<DType, IType>) -> Self {
        let mut wire_dtypes = HashMap::new();
        
        // Cache all wire dtypes from latched and next
        for (wire_id, dtype) in module.wire()[0].iter() {
            wire_dtypes.insert(wire_id, *dtype);
        }
        for (wire_id, dtype) in module.wire()[1].iter() {
            wire_dtypes.insert(wire_id, *dtype);
        }

        
        Context {
            wire_names: RefCell::new(HashMap::new()),
            wire_dtypes,
        }
    }

    pub fn add_name(&self, id: usize, name: &str) {
        self.wire_names.borrow_mut().insert(id, name.to_string());
    }

    fn trace_wire_expression(&self, wire_id: usize, terms: &[Term<DType, IType>]) -> String {
        let term = terms
            .iter()
            .find(|t| t.writes().iter().any(|(w, _)|w == wire_id));

        match term {
            None => {
                // No term writes to this wire; it must be an input or external wire.
                self.wire_name(wire_id)
            }
            Some(term) => {
                // Wire is written by a term, expand based on operation type.
                match term.itype() {
                    IType::Const(val) => {
                        format!("{}", val)
                    }
                    IType::Arith(op) => {
                        let read_wires: Vec<String> = term
                            .reads()
                            .iter()
                            .map(|(w, _)| self.trace_wire_expression(w, terms))
                            .collect();
                        let op_str = match op {
                            ArithOp::Add => "+",
                            ArithOp::Sub => "-",
                            ArithOp::Mul => "*",
                            ArithOp::Div => "/",
                        };
                        format!("({} {} {})",
                            read_wires[0],
                            op_str,
                            read_wires[1]
                        )
                    }
                    IType::Logical(op) => {
                        let read_wires: Vec<String> = term
                            .reads()
                            .iter()
                            .map(|(w, _)| self.trace_wire_expression(w, terms))
                            .collect();
                        match op {
                            LogicalOp::Not => {
                                format!("!{}", read_wires[0])
                            }
                            LogicalOp::And => {
                                format!("({} ∧ {})", read_wires[0], read_wires[1])
                            }
                            LogicalOp::Or => {
                                format!("({} ∨ {})", read_wires[0], read_wires[1])
                            }
                        }
                    }
                    IType::Cmp(op) => {
                        let read_wires: Vec<String> = term
                            .reads()
                            .iter()
                            .map(|(w, _)| self.trace_wire_expression(w, terms))
                            .collect();
                        let op_str = match op {
                            CmpOp::Eq => "==",
                            CmpOp::Lt => "<",
                            CmpOp::Le => "<=",
                            CmpOp::Gt => ">",
                            CmpOp::Ge => ">=",
                        };
                        format!("({} {} {})",
                            read_wires[0],
                            op_str,
                            read_wires[1]
                        )
                    }
                    IType::Id => {
                        // Identity: just pass through the input
                        let input_id = term.reads().iter().next().unwrap().0;
                        self.trace_wire_expression(input_id, terms)
                    }
                    IType::Cond => {
                        // Ternary: condition ? true_val : false_val
                        let cond_id = term.reads().iter().next().unwrap().0;
                        let true_id = term.reads().iter().nth(1).unwrap().0;
                        let false_id = term.reads().iter().nth(2).unwrap().0;
                        
                        let cond_expr = self.trace_wire_expression(cond_id, terms);
                        let true_expr = self.trace_wire_expression(true_id, terms);
                        let false_expr = self.trace_wire_expression(false_id, terms);
                        
                        format!("{} ? {} : {}", cond_expr, true_expr, false_expr)
                    }
                }
            }
        }
    }

    fn wire_name(&self, id: usize) -> String {
        self.wire_names
            .borrow()
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("w{}", id))
    }

    pub fn populate_default_wire_names(&self, module: &Module<DType, IType>) {
        let pair = module.wire();
        let latched = &pair[0];
        let next = &pair[1];

        let mut map = self.wire_names.borrow_mut();
        map.clear();

        // Name by wire ID: w0→x0, w1→x1, w2→x2, w3→x3, w4→x4, w5→x5 ...
        for (wire_id, _dtype) in latched.iter() {
            map.insert(wire_id, format!("x{}", wire_id));
        }

        // Name by wire ID offset: w6→x0', w7→x1', w8→x2', w9→x3', w10→x4', w11→x5'
        for (wire_id, _dtype) in next.iter() {
            // Assuming next wires are offset by the number of latched wires
            let base_id = wire_id - latched.len();
            map.insert(wire_id, format!("x{}'", base_id));
        }
    }
}

impl Descriptor<DType, IType> for Context {
    fn describe_module(&self, module: &Module<DType, IType>, _how: DescriptionContext) -> String {
        // Describe the module with its variables classified into prvt, intf, extl
        let prvt = module.prvt()[0]
            .iter()
            .map(|(wire_id, _)| self.wire_name(wire_id))
            .collect::<Vec<String>>();
        let intf = module.intf()[0]
            .iter()
            .map(|(wire_id, _)| self.wire_name(wire_id))
            .collect::<Vec<String>>();
        let extl = module.extl()[0]
            .iter()
            .map(|(wire_id, _)| self.wire_name(wire_id))
            .collect::<Vec<String>>();

        // Now describe each atom in the module
        let mut atoms_html = String::new();
        for (atom_id, atom) in module.atoms().iter().enumerate() {
            atoms_html.push_str(&format!(
                "<hr><h3>Atom {atom_id}</h3><div>{}</div>",
                self.describe_atom(atom, DescriptionContext::Inline)
            ));
        }
        
        // Combine module variable diagram and atoms description
        format!(
            "<h2>Module</h2>{}<h2>Atoms</h2>{}",
            module_variables_diagram(&prvt, &intf, &extl),
            atoms_html
        )
    }

    fn describe_atom(&self, atom: &Atom<DType, IType>, how: DescriptionContext) -> String {
        let mut atom_header = String::new();
        let mut atom_body = String::new();
        let mut atom_footer = String::new();

        if matches!(how, DescriptionContext::Standalone) {
            atom_header.push_str("Atom\n");
        }

        let ctrl_wires: Vec<String> = atom
            .ctrl()
            .iter()
            .map(|(w, _)| self.wire_name(w))
            .collect();
        let wait_wires: Vec<String> = atom
            .wait()
            .iter()
            .map(|(w, _)| self.wire_name(w))
            .collect();
        let read_wires: Vec<String> = atom
            .read()
            .iter()
            .map(|(w, _)| self.wire_name(w))
            .collect();

        atom_body.push_str(format!(
            "<table><tr><td><strong>ctrl</strong></td><td><code>{}</code></td></tr><tr><td><strong>wait</strong></td><td><code>{}</code></td></tr><tr><td><strong>read</strong></td><td><code>{}</code></td></tr></table>",
            ctrl_wires.join(", "), wait_wires.join(", "), read_wires.join(", ")
        ).as_str());

        atom_footer.push_str(&self.describe_atom_section(atom, "init", DescriptionContext::Inline));
        atom_footer.push_str(&self.describe_atom_section(atom, "update", DescriptionContext::Inline));
        
        format!(
            "<h2>{}</h2><div>{}</div><div>{}</div>",
            atom_header, atom_body, atom_footer
        )
    }

    fn describe_atom_section(&self, atom: &Atom<DType, IType>, sec: &str, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return match sec {
                "init" => "Init".into(),
                "update" => "Update".into(),
                _ => panic!("Invalid section"),
            };
        }

        let section_header = match sec {
            "init" => "Init",
            "update" => "Update",
            _ => panic!("Invalid section"),
        };

        let terms = match sec {
            "init" => atom.init(),
            "update" => atom.update(),
            _ => panic!("Invalid section"),
        };

        let section_body = atom.ctrl()
            .iter()
            .map(|(output_wire, _)| {
                let expr = self.trace_wire_expression(output_wire, terms);
                let wire_name = self.wire_name(output_wire);
                format!("<code>{} → {}</code>", expr, wire_name)
            })
            .collect::<Vec<String>>()
            .join("<br>\n");

        format!(
            "<h3>{}</h3><p>{}</p>",
            section_header,
            section_body
        )
    }

    fn describe_term(&self, term: &Term<DType, IType>, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return term.itype().to_string();
        }

        let mut term_header;
        let term_body;
        let mut term_footer = String::new();

        for wire in term.writes() {
            let wire_id = wire.0;
            let wire_name = self.wire_name(wire_id);
            let wire_dtype = &wire.1;
            term_footer.push_str(&format!(
                "<strong>writes</strong> <code>→</code> <small>{} (w{} : {})</small></p>",
                wire_name, wire_id, wire_dtype
            ));
        }
        for wire in term.reads() {
            let wire_id = wire.0;
            let wire_name = self.wire_name(wire_id);
            let wire_dtype = &wire.1;
            term_footer.push_str(&format!(
                "<strong>reads</strong> <code>←</code> <small>{} (w{} : {})</small></p>",
                wire_name, wire_id, wire_dtype
            ));
        }

        match term.itype() {
            IType::Const(val) => {
                term_header = "Constant".to_string();
                term_body = format!("<code>{}</code> → <code>{}</code>",
                    val,
                    self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                );
                match val {
                    Val::None => {},
                    Val::Real(_v) => {
                        term_header.push_str(" (Real)");
                    }
                    Val::Int(_v) => {
                        term_header.push_str(" (Int)");
                    }
                    Val::Bool(_v) => {
                        term_header.push_str(" (Bool)");
                    }
                }
            }
            IType::Arith(val) => {
                let op = match val {
                    ArithOp::Add => {
                        term_header = "Addition".to_string();
                        "+"
                    }
                    ArithOp::Sub => {
                        term_header = "Subtraction".to_string();
                        "-"
                    }
                    ArithOp::Mul => {
                        term_header = "Multiplication".to_string();
                        "*"
                    }
                    ArithOp::Div => {
                        term_header = "Division".to_string();
                        "/"
                    }
                };
                term_body = format!("<code>{}</code> {} <code>{}</code> → <code>{}</code>",
                    self.wire_name(term.reads().iter().next().map(|(idx, _)| idx).unwrap()),
                    op,
                    self.wire_name(term.reads().iter().nth(1).map(|(idx, _)| idx).unwrap()),
                    self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                );
            }
            IType::Logical(val) => {
                let op = match val {
                    LogicalOp::Not => {
                        term_header = "Logical Not".to_string();
                        "!"
                    }
                    LogicalOp::And => {
                        term_header = "Logical And".to_string();
                        "∧"
                    }
                    LogicalOp::Or => {
                        term_header = "Logical Or".to_string();
                        "∨"
                    }
                };
                if let LogicalOp::Not = val {
                    term_body = format!("{} <code>{}</code> → <code>{}</code>",
                        op,
                        self.wire_name(term.reads().iter().next().map(|(idx, _)| idx).unwrap()),
                        self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                    );
                } else {
                    term_body = format!("<code>{}</code> {} <code>{}</code> → <code>{}</code>",
                        self.wire_name(term.reads().iter().next().map(|(idx, _)| idx).unwrap()),
                        op,
                        self.wire_name(term.reads().iter().nth(1).map(|(idx, _)| idx).unwrap()),
                        self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                    );
                }
            }
            IType::Cmp(val) => {
                let op = match val {
                    CmpOp::Eq => {
                        term_header = "Equal".to_string();
                        "=="
                    }
                    CmpOp::Lt => {
                        term_header = "Less Than".to_string();
                        "<"
                    }
                    CmpOp::Le => {
                        term_header = "Less Equal".to_string();
                        "<="
                    }
                    CmpOp::Gt => {
                        term_header = "Greater Than".to_string();
                        ">"
                    }
                    CmpOp::Ge => {
                        term_header = "Greater Equal".to_string();
                        ">="
                    }
                };
                term_body = format!("<code>{}</code> {} <code>{}</code> → <code>{}</code>",
                    self.wire_name(term.reads().iter().next().map(|(idx, _)| idx).unwrap()),
                    op,
                    self.wire_name(term.reads().iter().nth(1).map(|(idx, _)| idx).unwrap()),
                    self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                );
            }
            IType::Id => {
                term_header = "Identity".to_string();
                term_body = format!(
                    "<code>{}</code> → <code>{}</code>",
                    self.wire_name(term.reads().iter().next().map(|(idx, _)| idx).unwrap()),
                    self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                );
            }
            IType::Cond => {
                term_header = "Ternary Condition".to_string();
                term_body = format!(
                    "<code>{}</code> ? <code>{}</code> : <code>{}</code> → <code>{}</code>",
                    self.wire_name(term.reads().iter().next().map(|(idx, _)| idx).unwrap()),
                    self.wire_name(term.reads().iter().nth(1).map(|(idx, _)| idx).unwrap()),
                    self.wire_name(term.reads().iter().nth(2).map(|(idx, _)| idx).unwrap()),
                    self.wire_name(term.writes().iter().next().map(|(idx, _)| idx).unwrap())
                );
            }
        }
        format!(
            "<h3>{}</h3><p>{}</p><hr>{}",
            term_header,
            term_body,
            term_footer
        )
    }

    fn describe_input(&self, id: usize) -> String {
        let name = self.wire_name(id);
        let dtype = self.wire_dtypes.get(&id).copied().unwrap_or(DType::Int);
        format!(
            "<h4>Input wire</h4>\n<code>{}</code> <small>(w{} : {})</small>",
            name, id, dtype
        )
    }

    fn describe_output(&self, id: usize) -> String {
        let name = self.wire_name(id);
        let dtype = self.wire_dtypes.get(&id).copied().unwrap_or(DType::Int);
        format!(
            "<h4>Output wire</h4>\n<code>{}</code> <small>(w{} : {})</small>",
            name, id, dtype
        )
    }

    fn describe_wire_id(&self, id: usize, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            self.wire_name(id)
        } else {
            format!("w{}", id)
        }
    }
}


/// Generate an SVG diagram showing the module variable classification
/// into private (prvt), interface (intf), and external (extl) variables.
fn module_variables_diagram(prvt: &[String], intf: &[String], extl: &[String]) -> String {
    format!(
        r##"
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
    "##,
        prvt.join(", "),
        intf.join(", "),
        extl.join(", ")
    )
}