import "katex/dist/katex.min.css";
import "./styles.css";
import { App } from "./app";

const app = new App();
document.getElementById("btn-log")!.addEventListener("click", () => app.flushLog());
app.start();
