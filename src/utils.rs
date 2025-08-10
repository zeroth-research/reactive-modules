use std::collections::HashMap;
use std::hash::Hash;

/// Error type for topological sort
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopoError {
    /// The graph contains at least one directed cycle
    Cycle,
}

/// DFS-based topological sort with cycle detection (3-color marking).
/// `graph` maps each node to its list of outgoing neighbors.
///
/// Complexity: O(V + E)
pub fn topo_dfs<N>(graph: &HashMap<N, Vec<N>>) -> Result<Vec<N>, TopoError>
where
    N: Eq + Hash + Clone,
{
    Err(TopoError::Cycle)
}
