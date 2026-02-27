use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::Html,
    routing::get,
};
use std::sync::Arc;
use tokio::sync::broadcast;

const INDEX_HTML: &str = include_str!("index.html");

type Tx = Arc<broadcast::Sender<String>>;

/// A live HTTP + WebSocket server that streams graph/value updates to the browser.
pub struct VisualServer {
    /// Keep the runtime alive for the lifetime of the server.
    _runtime: tokio::runtime::Runtime,
    tx: Tx,
    pub port: u16,
}

impl VisualServer {
    /// Spawn the server on `port` in a background tokio runtime and open the browser.
    pub fn start(port: u16) -> Self {
        let (tx_inner, _) = broadcast::channel::<String>(32);
        let tx: Tx = Arc::new(tx_inner);

        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let tx_spawn = Arc::clone(&tx);

        runtime.spawn(async move {
            let app = Router::new()
                .route("/", get(serve_index))
                .route("/ws", get(ws_handler))
                .with_state(tx_spawn);

            let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
                .await
                .unwrap_or_else(|e| panic!("visual server: bind {port}: {e}"));

            axum::serve(listener, app).await.expect("visual server");
        });

        // Give the server a moment to bind before opening the browser.
        std::thread::sleep(std::time::Duration::from_millis(150));
        open_browser(&format!("http://localhost:{port}"));

        VisualServer { _runtime: runtime, tx, port }
    }

    /// Broadcast a JSON message to all connected browser clients.
    pub fn push(&self, json: String) {
        // Ignore the error when there are no subscribers yet.
        let _ = self.tx.send(json);
    }
}

async fn serve_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(tx): State<Tx>,
) -> impl axum::response::IntoResponse {
    let rx = tx.subscribe();
    ws.on_upgrade(move |socket| handle_ws(socket, rx))
}

async fn handle_ws(mut socket: WebSocket, mut rx: broadcast::Receiver<String>) {
    loop {
        match rx.recv().await {
            Ok(msg) => {
                if socket.send(Message::Text(msg)).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd").args(["/c", "start", url]).spawn();
}
