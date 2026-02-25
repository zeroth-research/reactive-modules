// TODO: remove in the future
#![allow(dead_code)]

pub mod atom;
pub mod module;
pub mod term;
pub mod wire;

pub use crate::atom::Atom;
pub use crate::module::Module;
pub use crate::term::Block;
pub use crate::term::Term;
pub use crate::wire::Interface;
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
pub(crate) fn topological_order(graph: &[Vec<usize>]) -> Option<Vec<usize>> {
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

#[cfg(test)]
mod tests {
    use super::topological_order;

    #[test]
    fn empty_graph() {
        let graph: Vec<Vec<usize>> = vec![];
        assert_eq!(topological_order(&graph), Some(vec![]));
    }

    #[test]
    fn single_node() {
        let graph = vec![vec![]];
        assert_eq!(topological_order(&graph), Some(vec![0]));
    }

    #[test]
    fn simple_chain() {
        // 0 -> 1 -> 2
        let graph = vec![vec![1], vec![2], vec![]];
        assert_eq!(topological_order(&graph), Some(vec![0, 1, 2]));
    }

    #[test]
    fn branching_dag() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let graph = vec![vec![1, 2], vec![3], vec![3], vec![]];

        let order = topological_order(&graph).unwrap();

        // Check ordering constraints rather than exact order
        let pos = |n| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn disconnected_graph() {
        // 0 -> 1, 2 -> 3
        let graph = vec![vec![1], vec![], vec![3], vec![]];

        let order = topological_order(&graph).unwrap();

        let pos = |n| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn cycle_returns_none() {
        // 0 -> 1 -> 2 -> 0
        let graph = vec![vec![1], vec![2], vec![0]];
        assert_eq!(topological_order(&graph), None);
    }

    #[test]
    fn self_loop_returns_none() {
        // 0 -> 0
        let graph = vec![vec![0]];
        assert_eq!(topological_order(&graph), None);
    }
}
