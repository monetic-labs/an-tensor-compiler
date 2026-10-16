//! # Scheduling primitives for DAG-structured workloads.
//!
//! Critical path, topological levels, level-based makespan, and ready-set
//! computation. Uses a pure index-based representation (`edges: &[Vec<usize>]`,
//! `costs: &[u64]`) so any consumer with a DAG can reuse these without depending
//! on `dsl::TensorGraph`'s string-ID node model.
//!
//! # Edge convention
//!
//! `edges[i]` is the list of **predecessor** indices of node `i` — i.e. the
//! nodes that must complete before `i` can start. This matches
//! `dsl::TensorNode::dependencies` (after ID→index resolution) and makes
//! [`ready_set`] a direct predecessor-completion check.
//!
//! # Composition with `dsl::TensorGraph`
//!
//! `TensorGraph::ready_nodes` delegates to [`ready_set`] for its dependency-
//! met check (converting its string-ID deps to an index edge list first), then
//! applies its own parallelizable-node filter on top. The canonical logic
//! lives here; `TensorGraph` only adds its struct-specific filtering. Do not
//! re-implement predecessor-completion checks elsewhere — call [`ready_set`].

use crate::{Result, TensorCoreError};

/// Compute the ready set: indices of nodes not yet completed whose every
/// predecessor is completed.
///
/// `edges[i]` = predecessors of `i`. `completed[j]` = whether node `j` is done.
/// A node with no predecessors is ready iff it is not itself completed.
///
/// Does not require acyclicity (it is a local, one-step check), so it is safe
/// to call on graphs that have not been validated as DAGs. Out-of-range
/// predecessor indices are treated as uncompleted (never block readiness
/// spuriously, but also never satisfy it).
pub fn ready_set(edges: &[Vec<usize>], completed: &[bool]) -> Vec<usize> {
    let mut ready = Vec::new();
    for (i, deps) in edges.iter().enumerate() {
        if completed.get(i).copied().unwrap_or(false) {
            continue;
        }
        let deps_met = deps
            .iter()
            .all(|&p| completed.get(p).copied().unwrap_or(false));
        if deps_met {
            ready.push(i);
        }
    }
    ready
}

/// Assign each node its topological level: the length of the longest path from
/// any source (node with no predecessors) to that node. Sources have level 0.
///
/// Returns a `Vec<u32>` of length `n`. Returns an error if the graph contains
/// a cycle (levels are undefined on cyclic graphs).
pub fn topological_levels(edges: &[Vec<usize>]) -> Result<Vec<u32>> {
    let n = edges.len();
    // Build successor list: succ[u] = nodes v such that u is a predecessor of v.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0u32; n];
    for v in 0..n {
        for &u in &edges[v] {
            if u < n {
                succ[u].push(v);
                in_degree[v] += 1;
            }
        }
    }
    let mut level = vec![0u32; n];
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut visited = 0usize;
    while let Some(u) = queue.pop() {
        visited += 1;
        for &v in &succ[u] {
            level[v] = level[v].max(level[u] + 1);
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push(v);
            }
        }
    }
    if visited != n {
        return Err(TensorCoreError::Validation(format!(
            "scheduling::topological_levels: cycle detected (visited {visited} of {n} nodes)"
        )));
    }
    Ok(level)
}

/// Critical path length: the longest weighted path through the DAG, where the
/// length of a path is the sum of node costs along it. Equivalently, the max
/// over all nodes of the earliest finish time. Returns 0 for an empty graph.
///
/// `costs[i]` = cost (e.g., duration) of node `i`. Returns an error on a cycle.
pub fn critical_path(edges: &[Vec<usize>], costs: &[u64]) -> Result<u64> {
    let n = edges.len();
    if n == 0 {
        return Ok(0);
    }
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0u32; n];
    for v in 0..n {
        for &u in &edges[v] {
            if u < n {
                succ[u].push(v);
                in_degree[v] += 1;
            }
        }
    }
    let mut eft = vec![0u64; n]; // earliest finish time
    let mut queue: Vec<usize> = Vec::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            eft[i] = costs.get(i).copied().unwrap_or(0);
            queue.push(i);
        }
    }
    let mut visited = 0usize;
    while let Some(u) = queue.pop() {
        visited += 1;
        for &v in &succ[u] {
            // eft[v] = cost[v] + max over predecessors of eft[u].
            let cv = costs.get(v).copied().unwrap_or(0);
            eft[v] = eft[v].max(eft[u].saturating_add(cv));
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push(v);
            }
        }
    }
    if visited != n {
        return Err(TensorCoreError::Validation(format!(
            "scheduling::critical_path: cycle detected (visited {visited} of {n} nodes)"
        )));
    }
    Ok(*eft.iter().max().unwrap_or(&0))
}

/// Level-based makespan estimate with bounded concurrency.
///
/// For each topological level `L`: `wall-time = ceil(|L| / concurrency) * max(cost in L)`.
/// Total makespan = sum over levels. This is the standard "level scheduling"
/// bound: within a level all nodes are independent, so they spread across
/// `concurrency` workers; the level's wall-time is bounded by the slowest batch.
///
/// `concurrency` is clamped to >= 1. Returns an error on a cycle.
pub fn makespan_with_concurrency(
    edges: &[Vec<usize>],
    costs: &[u64],
    concurrency: u32,
) -> Result<u64> {
    let n = edges.len();
    if n == 0 {
        return Ok(0);
    }
    let c = (concurrency.max(1)) as usize;
    let levels = topological_levels(edges)?;
    let max_level = *levels.iter().max().unwrap_or(&0) as usize;
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (i, &l) in levels.iter().enumerate() {
        by_level[l as usize].push(i);
    }
    let mut total: u64 = 0;
    for level_nodes in &by_level {
        if level_nodes.is_empty() {
            continue;
        }
        let count = level_nodes.len();
        let max_cost = level_nodes
            .iter()
            .map(|&i| costs.get(i).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let batches = count.div_ceil(c); // ceil(count / c)
        total += batches as u64 * max_cost;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        assert_eq!(ready_set(&[], &[]), Vec::<usize>::new());
        assert_eq!(topological_levels(&[]).unwrap(), Vec::<u32>::new());
        assert_eq!(critical_path(&[], &[]).unwrap(), 0);
        assert_eq!(makespan_with_concurrency(&[], &[], 4).unwrap(), 0);
    }

    #[test]
    fn single_node() {
        let edges: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(ready_set(&edges, &[false]), vec![0]);
        assert_eq!(ready_set(&edges, &[true]), Vec::<usize>::new());
        assert_eq!(topological_levels(&edges).unwrap(), vec![0]);
        assert_eq!(critical_path(&edges, &[7]).unwrap(), 7);
        assert_eq!(makespan_with_concurrency(&edges, &[7], 2).unwrap(), 7);
    }

    #[test]
    fn linear_chain() {
        // 0 -> 1 -> 2
        let edges: Vec<Vec<usize>> = vec![vec![], vec![0], vec![1]];
        assert_eq!(ready_set(&edges, &[false, false, false]), vec![0]);
        assert_eq!(ready_set(&edges, &[true, false, false]), vec![1]);
        assert_eq!(ready_set(&edges, &[true, true, false]), vec![2]);
        assert_eq!(topological_levels(&edges).unwrap(), vec![0, 1, 2]);
        assert_eq!(critical_path(&edges, &[1, 2, 3]).unwrap(), 6);
        // Each level has one node; concurrency doesn't help.
        assert_eq!(makespan_with_concurrency(&edges, &[1, 2, 3], 4).unwrap(), 6);
    }

    #[test]
    fn diamond() {
        // 0 -> {1, 2} -> 3
        let edges: Vec<Vec<usize>> = vec![vec![], vec![0], vec![0], vec![1, 2]];
        assert_eq!(ready_set(&edges, &[false, false, false, false]), vec![0]);
        assert_eq!(ready_set(&edges, &[true, false, false, false]), vec![1, 2]);
        assert_eq!(ready_set(&edges, &[true, true, true, false]), vec![3]);
        assert_eq!(topological_levels(&edges).unwrap(), vec![0, 1, 1, 2]);
        // eft: 0=1, 1=1+5=6, 2=1+3=4, 3=max(6,4)+2=8
        assert_eq!(critical_path(&edges, &[1, 5, 3, 2]).unwrap(), 8);
        // c=2: L0={0}→1, L1={1,2}→ceil(2/2)*5=5, L2={3}→2 => 8
        assert_eq!(
            makespan_with_concurrency(&edges, &[1, 5, 3, 2], 2).unwrap(),
            8
        );
        // c=1: L1={1,2}→ceil(2/1)*5=10 => 1+10+2=13
        assert_eq!(
            makespan_with_concurrency(&edges, &[1, 5, 3, 2], 1).unwrap(),
            13
        );
    }

    #[test]
    fn cycle_detected() {
        // 0 <-> 1
        let edges: Vec<Vec<usize>> = vec![vec![1], vec![0]];
        assert!(topological_levels(&edges).is_err());
        assert!(critical_path(&edges, &[1, 1]).is_err());
        assert!(makespan_with_concurrency(&edges, &[1, 1], 1).is_err());
        // ready_set is local and must NOT error on a cycle.
        assert_eq!(ready_set(&edges, &[false, false]), Vec::<usize>::new());
    }

    #[test]
    fn self_loop() {
        let edges: Vec<Vec<usize>> = vec![vec![0]];
        assert!(topological_levels(&edges).is_err());
        // Node 0's only predecessor is itself (uncompleted) -> not ready.
        assert_eq!(ready_set(&edges, &[false]), Vec::<usize>::new());
    }

    #[test]
    fn out_of_range_predecessor_is_uncompleted() {
        // Node 0 depends on index 5 (out of range for a 1-node graph).
        let edges: Vec<Vec<usize>> = vec![vec![5]];
        // Predecessor 5 is treated as uncompleted -> 0 is not ready.
        assert_eq!(ready_set(&edges, &[false]), Vec::<usize>::new());
        // topological_levels ignores out-of-range predecessors (treats node as a source).
        assert_eq!(topological_levels(&edges).unwrap(), vec![0]);
    }

    #[test]
    fn mirrors_tensor_graph_three_node_dag() {
        // build_frontend (0), build_backend (1) -> deploy (2)
        let edges: Vec<Vec<usize>> = vec![vec![], vec![], vec![0, 1]];
        assert_eq!(ready_set(&edges, &[false, false, false]).len(), 2);
        let ready = ready_set(&edges, &[true, true, false]);
        assert_eq!(ready, vec![2]);
    }
}
