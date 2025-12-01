use crate::context::Context;
use std::collections::HashMap;
use std::fmt::Write;

use crate::{DType, IType, TorchAtom, TorchModule, TorchTerm};

use visual::html::{DescriptionContext, Descriptor};

impl Context {
    fn wire_name(&self, id: usize) -> String {
        if let Some(name) = self.get_name(id) {
            return name.into();
        }
        format!("w{id}")
    }

    fn dump_module(&self, module: &TorchModule, fmt: &HashMap<&str, &str>) -> String {
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

    fn dump_atom(&self, atom: &TorchAtom, fmt: &HashMap<&str, &str>) -> String {
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

    fn dump_atom_section(&self, atom: &TorchAtom, sec: &str, fmt: &HashMap<&str, &str>) -> String {
        let mut s = String::new();

        match sec {
            "init" => {
                for term in atom.init().iter() {
                    write!(s, "  ").unwrap();
                    writeln!(s, "{}", self.dump_term(term, fmt)).unwrap();
                }
            }
            "update" => {
                for term in atom.update().iter() {
                    write!(s, "  ").unwrap();
                    writeln!(s, "{}", self.dump_term(term, fmt)).unwrap();
                }
            }
            _ => panic!("Invalid section"),
        }

        s
    }

    fn dump_term(&self, term: &TorchTerm, fmt: &HashMap<&str, &str>) -> String {
        let empty_str = "";
        let fmt_emph = fmt.get("EMPH_START").unwrap_or(&empty_str);
        let fmt_emph_end = fmt.get("EMPH_END").unwrap_or(&empty_str);
        let color_term_lhs = fmt.get("COLOR_TERM_LHS").unwrap_or(&empty_str);
        let color_term_op = fmt.get("COLOR_TERM_OP").unwrap_or(&empty_str);
        let color_clr = fmt.get("COLOR_CLEAR").unwrap_or(&empty_str);

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
            "{color_term_lhs}{writes}{color_clr} = {color_term_op}{fmt_emph}{}{fmt_emph_end}{color_clr}({reads})",
            term.itype()
        )
    }
}

impl Descriptor<DType, IType> for Context {
    fn describe_module(&self, module: &TorchModule, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return format!("Module {}", module.name().unwrap_or(""));
        }

        let fmt = HashMap::from([
            ("BOLD_START", "<b>"),
            ("BOLD_END", "</b>"),
            ("COLOR_TERM_LHS", "<span style=\"color: #106EE2\">"),
            ("COLOR_TERM_OP", "<span style=\"color: #E88914\">"),
            ("COLOR_CLEAR", "</span>"),
        ]);
        format!("<pre>\n{}</pre>", self.dump_module(module, &fmt))
    }

    fn describe_atom(&self, atom: &TorchAtom, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return "Atom".into();
        }

        let fmt = HashMap::from([
            ("BOLD_START", "<b>"),
            ("BOLD_END", "</b>"),
            ("COLOR_TERM_LHS", "<span style=\"color: #106EE2\">"),
            ("COLOR_TERM_OP", "<span style=\"color: #E88914\">"),
            ("COLOR_CLEAR", "</span>"),
        ]);
        format!("<h2>Atom</h2><pre>\n{}</pre>", self.dump_atom(atom, &fmt))
    }

    fn describe_atom_section(
        &self,
        atom: &TorchAtom,
        sec: &str,
        how: DescriptionContext,
    ) -> String {
        if matches!(how, DescriptionContext::Node) {
            return match sec {
                "init" => "Init".into(),
                "update" => "Update".into(),
                _ => panic!("Invalid section"),
            };
        }

        let fmt = HashMap::from([
            ("BOLD_START", "<b>"),
            ("BOLD_END", "</b>"),
            ("COLOR_TERM_LHS", "<span style=\"color: #106EE2\">"),
            ("COLOR_TERM_OP", "<span style=\"color: #E88914\">"),
            ("COLOR_CLEAR", "</span>"),
        ]);
        format!(
            "<h2>Atom {}</h2><pre>\n{}</pre>",
            sec,
            self.dump_atom_section(atom, sec, &fmt)
        )
    }

    fn describe_term(&self, term: &TorchTerm, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return term.itype().to_string();
        }

        let fmt = HashMap::from([
            //("EMPH_START", "<i>"),
            //("EMPH_END", "</i>"),
            ("COLOR_TERM_LHS", "<span style=\"color: #106EE2\">"),
            ("COLOR_TERM_OP", "<span style=\"color: #E88914\">"),
            ("COLOR_CLEAR", "</span>"),
        ]);
        format!(
            "<pre>\n{}</pre>\n\nraw:\n\n<pre>{}</pre>",
            self.dump_term(term, &fmt),
            term
        )
    }

    fn describe_wire_id(&self, id: usize, _how: DescriptionContext) -> String {
        self.wire_name(id)
    }
}
