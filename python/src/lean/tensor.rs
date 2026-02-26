use tch::{Kind, Tensor};

/// Translate [tch::Tensor] into Lean

// fn lean_dtype(kind: Kind) -> &'static str {
//     match kind {
//         Kind::Float | Kind::Double => "Float",
//         Kind::Int | Kind::Int64 | Kind::Int8 | Kind::Int16 | Kind::Uint8 => "Int",
//         Kind::Bool => "Bool",
//         k => panic!("unsupported dtype: {k:?}"),
//     }
// }

fn lean_float(v: f64) -> String {
    if v.is_nan() {
        return "Float.nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "Float.inf".into()
        } else {
            "-Float.inf".into()
        };
    }
    format!("{v:.8}")
}

/// Extract all elements from a tensor as Lean literal strings, safely.
///
// XXX: we store them in `Vec` which might be a problem for large tensors.
// However, if the size of the tensor should be problem here, it will not definitely
// work in Lean, so we should be good for now.
fn render_elements(tensor: &Tensor) -> Vec<String> {
    let t = tensor
        .to_device(tch::Device::Cpu)
        .contiguous()
        .flatten(0, -1);
    let n = t.size()[0];
    match t.kind() {
        Kind::Float | Kind::Double => (0..n)
            .map(|i| lean_float(f64::from(t.double_value(&[i]))))
            .collect(),
        Kind::Int | Kind::Int8 | Kind::Int16 | Kind::Uint8 | Kind::Int64 => (0..n)
            .map(|i| format!("({} : Int)", t.int64_value(&[i])))
            .collect(),
        Kind::Bool => (0..n)
            .map(|i| {
                if t.int64_value(&[i]) != 0 {
                    "true".into()
                } else {
                    "false".into()
                }
            })
            .collect(),
        k => panic!("unsupported dtype: {k:?}"),
    }
}

// ── Main codegen ─────────────────────────────────────────────────────────────

/// Emit a Lean `def` for `tensor` named `name`.
///
/// Shape [d0, d1, ..., dn] with dtype α produces:
///   def <name> : Fin d0 → Fin d1 → ... → Fin dn → α :=
///     TensorFn.ofArray [d0, d1, ..., dn] #[v0, v1, ...]
// pub fn tensor_to_lean(name: &str, tensor: &Tensor) -> String {
//     let kind = tensor.kind();
//     let shape = tensor.size(); // Vec<i64>
//     let dtype = lean_dtype(kind);
//     let elems = render_elements(tensor);
//
//     // Fin d0 → Fin d1 → ... → α
//     let sig: String = shape
//         .iter()
//         .map(|d| format!("Fin {d} → "))
//         .collect::<String>()
//         + dtype;
//
//     // [d0, d1, ...]
//     let shape_lit = shape
//         .iter()
//         .map(|d| d.to_string())
//         .collect::<Vec<_>>()
//         .join(", ");
//
//     // #[v0, v1, ...]
//     let array_lit = format!("#[{}]", elems.join(", "));
//
//     format!("(fun : {sig} :=\n  TensorFn.ofArray [{shape_lit}] {array_lit})\n")
// }

pub fn tensor_to_lean(tensor: &Tensor) -> String {
    let shape = tensor.size(); // Vec<i64>
    let elems = render_elements(tensor);

    // #[v0, v1, ...]
    let array_lit = format!("#[{}]", elems.join(", "));

    // fun (i0: Fin ...) (i1: Fin ...) ... => array[i0.val * (d1*d2*...) + i1.val * (d2*...) + ... + in.val]!
    let params: String = shape
        .iter()
        .enumerate()
        .map(|(k, d)| format!("(i{k} : Fin {d})"))
        .collect::<Vec<_>>()
        .join(" ");

    let index_expr: String = shape
        .iter()
        .enumerate()
        .map(|(k, _)| {
            // stride for dimension k = product of all dims after k
            let stride: i64 = shape[k + 1..].iter().product();
            if stride == 1 {
                format!("i{k}.val")
            } else {
                format!("i{k}.val * {stride}")
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");

    // handle 0-dim tensors (scalar): no lambda needed
    if shape.is_empty() {
        elems.into_iter().next().unwrap_or_else(|| "default".into())
    } else {
        format!("fun {params} => {array_lit}[{index_expr}]!")
    }
}
