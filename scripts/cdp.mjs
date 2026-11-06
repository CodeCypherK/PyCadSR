// Dev-only: drive the running app's WebView2 over the Chrome DevTools Protocol.
// Start the app with WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9223, then:
//   node scripts/cdp.mjs 9223 eval "document.title"
//   node scripts/cdp.mjs 9223 shot out.png
import fs from "node:fs";

const [port, action, arg] = process.argv.slice(2);
const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = targets.find((t) => t.type === "page");
if (!page) throw new Error("no page target");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let seq = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
};
await new Promise((r) => (ws.onopen = r));
const call = (method, params = {}) =>
  new Promise((r) => { const i = ++seq; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });

if (action === "evalfile" || action === "eval") {
  const expr = action === "evalfile" ? fs.readFileSync(arg, "utf8") : arg;
  const r = await call("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  const res = r.result?.result;
  console.log(res?.value !== undefined ? JSON.stringify(res.value, null, 1) : JSON.stringify(r.result ?? r.error));
} else if (action === "shot") {
  const r = await call("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(arg, Buffer.from(r.result.data, "base64"));
  console.log("saved", arg);
}
ws.close();
