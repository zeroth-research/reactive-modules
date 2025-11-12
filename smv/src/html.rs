use base::{module::Module, atom::Atom, term::Term};
use crate::{dtype::DType, itype::IType};
use visual::html::Descriptor;
use std::cell::RefCell;
use std::collections::HashMap;

/// A small, SMV-specific HTML descriptor used by the visualiser.
///
/// This implements the `visual::html::Descriptor` trait for the
/// SMV `DType`/`IType` pair and provides a compact HTML rendering of
/// modules, atoms and terms. It also builds a wire-name mapping so
/// wires are displayed as `x0, x1, ...` and next wires as `x0', x1', ...`.
pub struct SmvDescriptor {
	/// Map from numeric wire index -> display name (e.g. "x0" or "x0'").
	wire_names: RefCell<HashMap<usize, String>>,
}

impl SmvDescriptor {
	pub fn new() -> Self {
		SmvDescriptor {
			wire_names: RefCell::new(HashMap::new()),
		}
	}

	fn describe_term_flow_with_context(&self, term: &Term<DType, IType>, atom_terms: &[&Term<DType, IType>]) -> String {
		use crate::itype::IType;
		use std::collections::HashSet;

		// Build map from wire -> term that writes it (within this atom)
		let mut writer_map: std::collections::HashMap<usize, &Term<DType, IType>> = std::collections::HashMap::new();
		for &t in atom_terms.iter() {
			for (wi, _dt) in t.writes().iter() {
				writer_map.insert(wi, t);
			}
		}

		// Render a wire: if it has a non-user name (xN/xN') return it,
		// otherwise if it's an internal wN and has a writer, expand it.
		fn render_wire(rec: &SmvDescriptor, idx: usize, writer_map: &std::collections::HashMap<usize, &Term<DType, IType>>, visited: &mut HashSet<usize>) -> String {
			let name = rec.describe_wire_id(idx);
			if !name.starts_with('w') {
				return format!("<code>{}</code>", name);
			}
			if visited.contains(&idx) {
				return format!("<code>{}</code>", name);
			}
			if let Some(t) = writer_map.get(&idx) {
				visited.insert(idx);
				let s = render_term_expr(rec, t, writer_map, visited);
				visited.remove(&idx);
				return s;
			}
			format!("<code>{}</code>", name)
		}

		// Render a term as an expression (without its target arrow)
		fn render_term_expr(rec: &SmvDescriptor, term: &Term<DType, IType>, writer_map: &std::collections::HashMap<usize, &Term<DType, IType>>, visited: &mut HashSet<usize>) -> String {
			use crate::itype::IType;
			let it = term.itype();
			match it {
				IType::ConstInt(v) => format!("<code>{}</code>", v),
				IType::ConstBool(b) => format!("<code>{}</code>", b),
				IType::Add | IType::Sub | IType::Mul | IType::Div => {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 2 {
						let op = match it {
							IType::Add => "+",
							IType::Sub => "-",
							IType::Mul => "*",
							IType::Div => "/",
							_ => "?",
						};
						let a = render_wire(rec, rds[0].0, writer_map, visited);
						let b = render_wire(rec, rds[1].0, writer_map, visited);
						return format!("{} {} {}", a, op, b);
					}
					String::new()
				}
				IType::Not => {
					if let Some(r) = term.reads().get_single() {
						let s = render_wire(rec, r.0, writer_map, visited);
						return format!("!{}", s);
					}
					String::new()
				}
				IType::And | IType::Or => {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 2 {
						let op = if let IType::And = it { "∧" } else { "∨" };
						let a = render_wire(rec, rds[0].0, writer_map, visited);
						let b = render_wire(rec, rds[1].0, writer_map, visited);
						return format!("{} {} {}", a, op, b);
					}
					String::new()
				}
				IType::Lt | IType::Le | IType::Gt | IType::Ge | IType::Eq => {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 2 {
						let op = match it {
							IType::Lt => "<",
							IType::Le => "<=",
							IType::Gt => ">",
							IType::Ge => ">=",
							IType::Eq => "==",
							_ => "?",
						};
						let a = render_wire(rec, rds[0].0, writer_map, visited);
						let b = render_wire(rec, rds[1].0, writer_map, visited);
						return format!("{} {} {}", a, op, b);
					}
					String::new()
				}
				IType::Abs => {
					if let Some(r) = term.reads().get_single() {
						let s = render_wire(rec, r.0, writer_map, visited);
						return format!("|{}|", s);
					}
					String::new()
				}
				IType::Cond => {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 3 {
						let c = render_wire(rec, rds[0].0, writer_map, visited);
						let t = render_wire(rec, rds[1].0, writer_map, visited);
						let e = render_wire(rec, rds[2].0, writer_map, visited);
						// Use parentheses so complex sub-expressions are clearly grouped
						return format!("({}) ? ({}) : ({})", c, t, e);
					}
					String::new()
				}
				IType::Next | IType::Assign | IType::Init => {
					// simple pass-through of sources
					let srcs: Vec<String> = term.reads().iter().map(|r| render_wire(rec, r.0, writer_map, visited)).collect();
					return srcs.join(", ");
				}
			}
		}

		// Top-level: render the expression and append the target(s) for this term
		let mut visited = HashSet::new();
		let expr = render_term_expr(self, term, &writer_map, &mut visited);
		if expr.is_empty() {
			return String::new();
		}
 		// Use the first write that exists as the target label
 		if let Some(w) = term.writes().get_single() {
 			let tgt = self.describe_wire_id(w.0);
 			return format!("{} → <code>{}</code>", expr, tgt)
 		}
		expr
	}

	/// Populate the internal wire-name mapping from a module.
	fn populate_wire_names(&self, module: &Module<DType, IType>) {
		let pair = module.wire();
		let latched = &pair[0];
		let next = &pair[1];

		let mut map = self.wire_names.borrow_mut();
		map.clear();

		// Assign names by position in the latched vector: x0, x1, ...
		for (pos, (idx, _dtype)) in latched.iter().enumerate() {
			map.insert(idx, format!("x{}", pos));
		}

		// For the next wires, assign xN' for the corresponding position
		for (pos, (nidx, _dtype)) in next.iter().enumerate() {
			map.insert(nidx, format!("x{}'", pos));
		}
	}
}

impl Descriptor<DType, IType> for SmvDescriptor {
	fn describe_module(&self, module: &Module<DType, IType>) -> String {
		// Build wire name map first so describe_wire_id can use it.
		self.populate_wire_names(module);

		// Create a simple HTML table showing latched vs next names and indices
		let pair = module.wire();
		let latched = &pair[0];
		let next = &pair[1];

		let mut rows = String::new();
		rows.push_str("<tr><th>Name</th><th>Latched idx</th><th>Next idx</th><th>Type</th></tr>");

		// Build indexable vectors with owned values so we can safely index by position
		let latched_v: Vec<(usize, String)> = latched.iter().map(|p| (p.0, format!("{:?}", p.1))).collect();
		let next_v: Vec<usize> = next.iter().map(|p| p.0).collect();
		let max = std::cmp::max(latched_v.len(), next_v.len());
		for i in 0..max {
			let (lat_idx_s, dtype_s) = latched_v.get(i).map(|p| (p.0.to_string(), p.1.clone())).unwrap_or_else(|| ("".into(), "".into()));
			let name = latched_v.get(i).map(|p| self.describe_wire_id(p.0)).unwrap_or_else(|| "".into());
			let prim_idx = next_v.get(i).map(|p| p.to_string()).unwrap_or_else(|| "".into());

			rows.push_str(&format!("<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>", name, lat_idx_s, prim_idx, dtype_s));
		}

		format!("<h3>Wires</h3><table>{}</table><hr><pre>{}</pre>", rows, module)
	}

	fn describe_atom(&self, atom: &Atom<DType, IType>) -> String {
		format!("<pre>{}</pre>", atom)
	}

	fn describe_term_label(&self, term: &Term<DType, IType>) -> String {
		use crate::itype::IType;
		match term.itype() {
			IType::ConstInt(v) => format!("{}", v),
			IType::ConstBool(b) => format!("{}", b),
			other => other.to_string(),
		}
	}

	fn describe_term_flow(&self, term: &Term<DType, IType>) -> String {
		use crate::itype::IType;

		let it = term.itype();
	let writes: Vec<(usize, DType)> = term.writes().iter().map(|(i, d)| (i.clone(), d.clone())).collect();
	let reads: Vec<(usize, DType)> = term.reads().iter().map(|(i, d)| (i.clone(), d.clone())).collect();

		let emit_tgt = |widx: usize| format!("<code>{}</code>", self.describe_wire_id(widx));
		let emit_src = |r: &(usize, DType)| format!("<code>{}</code>", self.describe_wire_id(r.0));

		match it {
			IType::ConstInt(v) => {
				if let Some((widx, _)) = writes.get(0) {
					format!("<code>{}</code> → {}", v, emit_tgt(*widx))
				} else {
					format!("<code>{}</code>", v)
				}
			}
			IType::ConstBool(b) => {
				if let Some((widx, _)) = writes.get(0) {
					format!("<code>{}</code> → {}", b, emit_tgt(*widx))
				} else {
					format!("<code>{}</code>", b)
				}
			}
			IType::Add | IType::Sub | IType::Mul | IType::Div => {
				if !reads.is_empty() && writes.get(0).is_some() {
					if reads.len() >= 2 {
						let op = match it {
							IType::Add => "+",
							IType::Sub => "-",
							IType::Mul => "*",
							IType::Div => "/",
							_ => "?",
						};
						let src1 = emit_src(&reads[0]);
						let src2 = emit_src(&reads[1]);
						let tgt = emit_tgt(writes[0].0);
						return format!("{} {} {} → {}", src1, op, src2, tgt);
					}
				}
				String::new()
			}
			IType::Not => {
				if !reads.is_empty() && writes.get(0).is_some() {
					let src = emit_src(&reads[0]);
					let tgt = emit_tgt(writes[0].0);
					return format!("!{} → {}", src, tgt);
				}
				String::new()
			}
			IType::And | IType::Or => {
				if !reads.is_empty() && writes.get(0).is_some() {
					if reads.len() >= 2 {
						let op = if let IType::And = it { "∧" } else { "∨" };
						let src1 = emit_src(&reads[0]);
						let src2 = emit_src(&reads[1]);
						let tgt = emit_tgt(writes[0].0);
						return format!("{} {} {} → {}", src1, op, src2, tgt);
					}
				}
				String::new()
			}
			IType::Lt | IType::Le | IType::Gt | IType::Ge | IType::Eq => {
				if !reads.is_empty() && writes.get(0).is_some() {
					if reads.len() >= 2 {
						let op = match it {
							IType::Lt => "<",
							IType::Le => "<=",
							IType::Gt => ">",
							IType::Ge => ">=",
							IType::Eq => "==",
							_ => "?",
						};
						let src1 = emit_src(&reads[0]);
						let src2 = emit_src(&reads[1]);
						let tgt = emit_tgt(writes[0].0);
						return format!("{} {} {} → {}", src1, op, src2, tgt);
					}
				}
				String::new()
			}
			IType::Next | IType::Assign => {
				if !reads.is_empty() && writes.get(0).is_some() {
					let srcs: Vec<String> = reads.iter().map(|r| emit_src(r)).collect();
					let src_join = srcs.join(", ");
					let tgt = emit_tgt(writes[0].0);
					return format!("{} → {}", src_join, tgt);
				}
				String::new()
			}
			IType::Init => {
				if writes.get(0).is_some() {
					if !reads.is_empty() {
						let srcs: Vec<String> = reads.iter().map(|r| emit_src(r)).collect();
						let src_join = srcs.join(", ");
						let tgt = emit_tgt(writes[0].0);
						return format!("{} → {}", src_join, tgt);
					} else {
						// e.g., a constant init produced as an init term
						let tgt = emit_tgt(writes[0].0);
						return format!("{}", tgt);
					}
				}
				String::new()
			}
			IType::Abs => {
				if !reads.is_empty() && writes.get(0).is_some() {
					let src = emit_src(&reads[0]);
					let tgt = emit_tgt(writes[0].0);
					return format!("|{}| → {}", src, tgt);
				}
				String::new()
			}
			IType::Cond => {
				if writes.get(0).is_some() {
					if reads.len() >= 3 {
						let c = emit_src(&reads[0]);
						let t = emit_src(&reads[1]);
						let e = emit_src(&reads[2]);
						let tgt = emit_tgt(writes[0].0);
						return format!("({}) ? ({}) : ({}) → {}", c, t, e, tgt);
					}
				}
				String::new()
			}
		}
	}

	// Provide the trait override so callers using the Descriptor trait will
	// invoke our context-aware renderer. The heavy lifting is in the
	// inherent method `describe_term_flow_with_context` above.

	fn describe_term_flow_with_context(&self, term: &Term<DType, IType>, atom_terms: &[&Term<DType, IType>]) -> String {
		SmvDescriptor::describe_term_flow_with_context(self, term, atom_terms)
	}

	fn describe_term(&self, term: &Term<DType, IType>) -> String {
		use crate::itype::IType;

		let it = term.itype();

		// Build a simple reads/writes table to show wire names, ids and dtypes.
		let writes_v: Vec<_> = term.writes().iter().collect();
		let reads_v: Vec<_> = term.reads().iter().collect();

		let mut rows = String::new();
		rows.push_str("<table>");
		if !writes_v.is_empty() {
			for (idx, dtype) in writes_v.iter() {
				let name = self.describe_wire_id(*idx);
				rows.push_str(&format!(
					"<tr><td><strong>writes</strong></td><td><code>{}</code> <small>({} : {})</small></td></tr>",
					name, name, dtype
				));
			}
		}
		if !reads_v.is_empty() {
			for (idx, dtype) in reads_v.iter() {
				let name = self.describe_wire_id(*idx);
				rows.push_str(&format!(
					"<tr><td><strong>reads</strong></td><td><code>{}</code> <small>({} : {})</small></td></tr>",
					name, name, dtype
				));
			}
		}
		rows.push_str("</table>");

		// Try to provide a small, specific summary for certain instruction types
		let mut extra = String::new();
		let mut title_html = format!("<h3>{}</h3>", it);
		match it {
			IType::ConstInt(v) => {
				if let Some(w) = term.writes().get_single() {
					let tgt = self.describe_wire_id(w.0);
					title_html = "<h3>Constant (Int)</h3>".into();
					extra = format!("<p><code>{}</code> → <code>{}</code></p>", v, tgt);
				}
			}
			IType::ConstBool(b) => {
				if let Some(w) = term.writes().get_single() {
					let tgt = self.describe_wire_id(w.0);
					title_html = "<h3>Constant (Bool)</h3>".into();
					extra = format!("<p><code>{}</code> → <code>{}</code></p>", b, tgt);
				}
			}
			IType::Add | IType::Sub | IType::Mul | IType::Div => {
				// Binary arithmetic: expect one write and two reads
				if let Some(w) = term.writes().get_single() {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 2 {
						let op = match it {
							IType::Add => "+",
							IType::Sub => "-",
							IType::Mul => "*",
							IType::Div => "/",
							_ => "?",
						};
						let src1 = self.describe_wire_id(rds[0].0);
						let src2 = self.describe_wire_id(rds[1].0);
						let tgt = self.describe_wire_id(w.0);
                        let title_str = match it {
                            IType::Add => "Addition",
                            IType::Sub => "Subtraction",
                            IType::Mul => "Multiplication",
                            IType::Div => "Division",
                            _ => "Arithmetic",
                        };
                        title_html = format!("<h3>{}</h3>", title_str);
						extra = format!("<p><code>{}</code> {} <code>{}</code> → <code>{}</code></p>", src1, op, src2, tgt);
					}
				}
			}
			IType::Not => {
				if let (Some(w), Some(r)) = (term.writes().get_single(), term.reads().get_single()) {
					let tgt = self.describe_wire_id(w.0);
					let src = self.describe_wire_id(r.0);
					extra = format!("<p><code>!{}</code> → <code>{}</code></p>", src, tgt);
				}
			}
			IType::And | IType::Or => {
				if let Some(w) = term.writes().get_single() {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 2 {
						let op = if let IType::And = it { "∧" } else { "∨" };
						let src1 = self.describe_wire_id(rds[0].0);
						let src2 = self.describe_wire_id(rds[1].0);
						let tgt = self.describe_wire_id(w.0);
						extra = format!("<p><code>{}</code> {} <code>{}</code> → <code>{}</code></p>", src1, op, src2, tgt);
					}
				}
			}
			IType::Lt | IType::Le | IType::Gt | IType::Ge | IType::Eq => {
				if let Some(w) = term.writes().get_single() {
					let rds: Vec<_> = term.reads().iter().collect();
					if rds.len() >= 2 {
						let op = match it {
							IType::Lt => "<",
							IType::Le => "<=",
							IType::Gt => ">",
							IType::Ge => ">=",
							IType::Eq => "==",
							_ => "?",
						};
						let src1 = self.describe_wire_id(rds[0].0);
						let src2 = self.describe_wire_id(rds[1].0);
						let tgt = self.describe_wire_id(w.0);
                        let title_str = match it {
                            IType::Lt => "Less Than",
                            IType::Le => "Less Than or Equal",
                            IType::Gt => "Greater Than",
                            IType::Ge => "Greater Than or Equal",
                            IType::Eq => "Equality",
                            _ => "Comparison",
                        };
                        title_html = format!("<h3>{}</h3>", title_str);
						extra = format!("<p><code>{}</code> {} <code>{}</code> → <code>{}</code></p>", src1, op, src2, tgt);
					}
				}
			}
			IType::Next => {
				if let (Some(w), Some(r)) = (term.writes().get_single(), term.reads().get_single()) {
					let tgt = self.describe_wire_id(w.0);
					let src = self.describe_wire_id(r.0);
					extra = format!("<p><code>{}</code> → <code>{}</code> <small>(next)</small></p>", src, tgt);
				}
			}
			IType::Init => {
				let rds: Vec<_> = term.reads().iter().collect();
				let writes_v: Vec<_> = term.writes().iter().collect();
				if !writes_v.is_empty() {
					let tgt = self.describe_wire_id(writes_v[0].0);
					if !rds.is_empty() {
						let srcs: Vec<String> = rds.iter().map(|r| format!("<code>{}</code>", self.describe_wire_id(r.0))).collect();
						let srcs_joined = srcs.join(", ");
						extra = format!("<p>{} → <code>{}</code> <small>(init)</small></p>", srcs_joined, tgt);
					} else {
						extra = format!("<p><code>{}</code> <small>(init)</small></p>", tgt);
					}
				}
			}
			IType::Assign => {
				if let (Some(w), Some(r)) = (term.writes().get_single(), term.reads().get_single()) {
					let tgt = self.describe_wire_id(w.0);
					let src = self.describe_wire_id(r.0);
					extra = format!("<p><code>{}</code> → <code>{}</code></p>", src, tgt);
				}
			}
			IType::Abs => {
				if let (Some(w), Some(r)) = (term.writes().get_single(), term.reads().get_single()) {
					let tgt = self.describe_wire_id(w.0);
					let src = self.describe_wire_id(r.0);
                    title_html = format!("<h3>Absolute Value</h3>");
					extra = format!("<p>|<code>{}</code>| → <code>{}</code></p>", src, tgt);
				}
			}
			IType::Cond => {
				let rds: Vec<_> = term.reads().iter().collect();
				if let Some(w) = term.writes().get_single() {
					if rds.len() >= 3 {
						let tgt = self.describe_wire_id(w.0);
						let c = self.describe_wire_id(rds[0].0);
						let t = self.describe_wire_id(rds[1].0);
						let e = self.describe_wire_id(rds[2].0);
                        title_html = format!("<h3>Ternary Condition</h3>");
						extra = format!("<p><code>{}</code> ? <code>{}</code> : <code>{}</code> → <code>{}</code></p>", c, t, e, tgt);
					}
				}
			}
		}

		// Order: title, extra summary line, then reads/writes table.
		format!("{}{}<hr>{}", title_html, extra, rows)
	}

	fn describe_wire_id(&self, id: usize) -> String {
		let map = self.wire_names.borrow();
		map.get(&id).cloned().unwrap_or_else(|| format!("w{}", id))
	}

	fn describe_wire_label_for_edge(&self, id: usize) -> String {
		// Explicitly return the internal wire name for edge labels.
		format!("w{}", id)
	}
}