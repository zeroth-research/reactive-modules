use crate::context::Context;
use crate::{ToyAtom, ToyModule, ToyTerm};
use std::collections::HashMap;
use std::fmt::Write;

use crate::dtype::Type;
use crate::instruction::Instruction;

use visual::html::{DescriptionContext, Descriptor};

impl Context {
    fn wire_name(&self, id: usize) -> String {
        if let Some(name) = self.get_name(id) {
            return name.into();
        }
        format!("w{id}")
    }

    fn _dump_module(&self, module: &ToyModule, fmt: &HashMap<&str, &str>) -> String {
        let empty_str = "";
        let fmt_bold = fmt.get("BOLD_START").unwrap_or(&empty_str);
        let fmt_bold_end = fmt.get("BOLD_END").unwrap_or(&empty_str);

        let mut s = String::new();

        writeln!(s, "{fmt_bold}module{fmt_bold_end}").unwrap();

        writeln!(s, " {fmt_bold}external{fmt_bold_end}").unwrap();
        let extl = module.extl();
        for ((ltc, _), (nxt, dtype)) in extl[0].iter().zip(extl[1].iter()) {
            writeln!(
                s,
                "   {}, {}: {dtype}",
                self.wire_name(ltc),
                self.wire_name(nxt)
            )
            .unwrap();
        }

        writeln!(s, " {fmt_bold}interface{fmt_bold_end}").unwrap();
        let intf = module.intf();
        for ((ltc, _), (nxt, dtype)) in intf[0].iter().zip(intf[1].iter()) {
            writeln!(
                s,
                "   {}, {}: {dtype}",
                self.wire_name(ltc),
                self.wire_name(nxt)
            )
            .unwrap();
        }

        writeln!(s, " {fmt_bold}private{fmt_bold_end}").unwrap();
        let prvt = module.prvt();
        for ((ltc, _), (nxt, dtype)) in prvt[0].iter().zip(prvt[1].iter()) {
            writeln!(
                s,
                "   {}, {}: {dtype}",
                self.wire_name(ltc),
                self.wire_name(nxt)
            )
            .unwrap();
        }

        writeln!(s).unwrap();
        for atom in module.atoms() {
            writeln!(s, "{}", self.dump_atom(atom, fmt)).unwrap();
        }

        s
    }

    fn dump_atom(&self, atom: &ToyAtom, fmt: &HashMap<&str, &str>) -> String {
        let empty_str = "";
        let fmt_bold = fmt.get("BOLD_START").unwrap_or(&empty_str);
        let fmt_bold_end = fmt.get("BOLD_END").unwrap_or(&empty_str);

        let mut s = String::new();
        writeln!(s, "{fmt_bold}atom{fmt_bold_end}").unwrap();
        for (i, (wr, _)) in atom.ctrl().iter().enumerate() {
            if i == 0 {
                write!(
                    s,
                    " {fmt_bold}controls{fmt_bold_end} {}",
                    self.wire_name(wr)
                )
                .unwrap();
            } else {
                write!(s, ", {}", self.wire_name(wr)).unwrap();
            }
        }
        writeln!(s).unwrap();
        for (i, (wr, _)) in atom.read().iter().enumerate() {
            if i == 0 {
                write!(s, " {fmt_bold}reads{fmt_bold_end} {}", self.wire_name(wr)).unwrap();
            } else {
                write!(s, ", {}", self.wire_name(wr)).unwrap();
            }
        }
        writeln!(s).unwrap();
        for (i, (wr, _)) in atom.wait().iter().enumerate() {
            if i == 0 {
                write!(s, " {fmt_bold}awaits{fmt_bold_end} {}", self.wire_name(wr)).unwrap();
            } else {
                write!(s, ", {}", self.wire_name(wr)).unwrap();
            }
        }
        writeln!(s).unwrap();
        writeln!(s, "\n{fmt_bold}init{fmt_bold_end}").unwrap();

        for term in atom.init().iter() {
            write!(s, "  ").unwrap();
            writeln!(s, "{}", self.dump_term(term, fmt)).unwrap();
        }
        writeln!(s, "\n{fmt_bold}update{fmt_bold_end}").unwrap();
        for term in atom.update().iter() {
            write!(s, "  ").unwrap();
            writeln!(s, "{}", self.dump_term(term, fmt)).unwrap();
        }

        s
    }

    fn dump_term(&self, term: &ToyTerm, fmt: &HashMap<&str, &str>) -> String {
        let empty_str = "";
        let fmt_emph = fmt.get("EMPH_START").unwrap_or(&empty_str);
        let fmt_emph_end = fmt.get("EMPH_END").unwrap_or(&empty_str);

        let reads = term
            .reads()
            .iter()
            .map(|(id, _)| self.wire_name(id))
            .collect::<Vec<String>>()
            .join(", ");

        let writes = term
            .writes()
            .iter()
            .map(|(id, _)| self.wire_name(id))
            .collect::<Vec<String>>()
            .join(", ");

        format!(
            "{writes} = {fmt_emph}{}{fmt_emph_end}({reads})",
            term.itype()
        )
    }
}

fn module_variables_diagram(prvt: &Vec<String>, intf: &Vec<String>, extl: &Vec<String>) -> String {
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

///
/// HTML descriptor for Context
impl Descriptor<Type, Instruction> for Context {
    fn describe_module(&self, module: &ToyModule, _how: DescriptionContext) -> String {
        //let fmt = HashMap::from([("BOLD_START", "<b>"), ("BOLD_END", "</b>")]);
        //format!("<pre>\n{}</pre>", self.dump_module(module, &fmt))
        let prvt = module.prvt()[0]
            .iter()
            .map(|(wire_id, _)| self.describe_wire_id(wire_id, DescriptionContext::Inline))
            .collect::<Vec<String>>();
        let intf = module.intf()[0]
            .iter()
            .map(|(wire_id, _)| self.describe_wire_id(wire_id, DescriptionContext::Inline))
            .collect::<Vec<String>>();
        let extl = module.extl()[0]
            .iter()
            .map(|(wire_id, _)| self.describe_wire_id(wire_id, DescriptionContext::Inline))
            .collect::<Vec<String>>();

        format!(
            "<h2>Module {}</h2>{}<h2>Atoms</h2>{}",
            module.name().unwrap_or(""),
            module_variables_diagram(&prvt, &intf, &extl),
            // TODO: we should do this more efficient
            module
                .atoms()
                .iter()
                .enumerate()
                .map(|(atom_id, a)| format!(
                    "<h3>Atom {}</h3>{}",
                    atom_id,
                    self.describe_atom(a, DescriptionContext::Inline)
                ))
                .collect::<Vec<String>>()
                .join("\n\n")
        )
    }

    fn describe_atom(&self, atom: &ToyAtom, how: DescriptionContext) -> String {
        match how {
            DescriptionContext::Node => "Atom".into(),
            _ => {
                let fmt = HashMap::from([("BOLD_START", "<b>"), ("BOLD_END", "</b>")]);
                format!("<h2>Atom</h2>\n<pre>{}</pre>", self.dump_atom(atom, &fmt))
            }
        }
    }

    fn describe_atom_section(&self, atom: &ToyAtom, sec: &str, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return match sec {
                "init" => "Init".into(),
                "update" => "Update".into(),
                _ => panic!("Invalid section"),
            };
        }

        let fmt = HashMap::from([("BOLD_START", "<b>"), ("BOLD_END", "</b>")]);

        match sec {
            "init" => {
                let terms = atom
                    .init()
                    .iter()
                    .map(|t| self.dump_term(t, &fmt))
                    .collect::<Vec<String>>()
                    .join("\n");

                format!("<h2>Atom {}</h2>\n<pre>{}</pre>", sec, terms)
            }
            "update" => {
                let terms = atom
                    .update()
                    .iter()
                    .map(|t| self.dump_term(t, &fmt))
                    .collect::<Vec<String>>()
                    .join("\n");

                format!("<h2>Atom {}</h2>\n<pre>{}</pre>", sec, terms)
            }
            _ => panic!("Invalid atom section"),
        }
    }

    fn describe_term(&self, term: &ToyTerm, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return term.itype().to_string();
        }

        let fmt = HashMap::from([("EMPH_START", "<i>"), ("EMPH_END", "</i>")]);
        format!(
            "<pre>\n{}\n\nraw:\n\n{}</pre>",
            self.dump_term(term, &fmt),
            term
        )
    }

    fn describe_wire_id(&self, id: usize, _how: DescriptionContext) -> String {
        self.wire_name(id)
    }
}
