use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

use visual::server::VisualServer;
use visual::html::module::module_to_live_json;

use crate::module::Module;
use crate::types::{DType, IType};

static SERVER: OnceLock<VisualServer> = OnceLock::new();

/// Start the live visualisation server on `port` (default 7777) and open the browser.
/// Calling this more than once is a no-op — the server starts at most once.
#[pyfunction]
#[pyo3(signature = (port = 7777))]
pub fn _visual_start(port: u16) -> PyResult<()> {
    SERVER.get_or_init(|| VisualServer::start(port));
    Ok(())
}

/// Push the reactive module graph to all connected browser clients.
/// Does nothing if the server has not been started yet.
#[pyfunction]
pub fn _visual_push_module(module: PyRef<'_, Module>) -> PyResult<()> {
    if let Some(server) = SERVER.get() {
        let json = module_to_live_json::<DType, IType>(&module.base);
        server.push(json);
    }
    Ok(())
}

/// Push a wire-id → value mapping to all connected browser clients so that the
/// graph edges can show current values as overlays.
/// `values` is a dict mapping integer wire IDs to their string representation.
#[pyfunction]
pub fn _visual_push_values(values: HashMap<usize, String>) -> PyResult<()> {
    if let Some(server) = SERVER.get() {
        let json = serde_json::json!({
            "type": "values",
            "values": values
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<HashMap<_, _>>()
        })
        .to_string();
        server.push(json);
    }
    Ok(())
}
