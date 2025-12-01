// TODO: remove in the future
#![allow(dead_code)]

pub mod atom;
pub mod module;
pub mod term;
pub mod wire;

pub use crate::atom::Atom;
pub use crate::module::Module;
pub use crate::term::Term;
pub use crate::wire::Wire;

/// Computes a topological ordering of a directed graph using **Kahn’s algorithm**.
///
/// The graph is given as adjacency lists: `graph[u]` contains all `v` such that
/// there is an edge `u → v`. Vertices are identified by their indices.
///
/// Returns:
/// * `Some(order)` if the graph is acyclic,
/// * `None` if a cycle is detected.
///
/// The algorithm:
/// * Computes in-degrees,
/// * Repeatedly removes vertices with in-degree zero,
/// * Emits them in order and updates neighbours.
///
/// Runs in **O(V + E)** time.
///
/// ```
pub(crate) fn kahn(graph: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = graph.len();
    let mut indeg: Vec<usize> = vec![0; n];

    for adj in graph.iter() {
        for &v in adj.iter() {
            indeg[v] += 1;
        }
    }

    let mut q: std::collections::VecDeque<usize> = (0..n).filter(|&u| indeg[u] == 0).collect();

    let mut out = Vec::with_capacity(n);

    while let Some(u) = q.pop_front() {
        out.push(u);
        for &v in graph[u].iter() {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }

    if out.len() == n { Some(out) } else { None }
}
