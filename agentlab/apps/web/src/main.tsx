import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("Run the public demo flow and show the evidence chain.");
  const [out, setOut] = useState<any>(null);

  async function run() {
    const response = await fetch(API + "/api/v1/experiments/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: "demo" }),
    });
    setOut(await response.json());
  }

  return (
    <div className="app">
      <aside>
        <h2>焊接知识智能体实验台</h2>
        <p>Welding Knowledge AgentLab</p>
        <nav>Demo Flow<br />WSR Gate<br />Evidence Chain<br />Public API</nav>
      </aside>
      <main>
        <h1>Paper-linked Agent Workbench</h1>
        <div className="hero">
          <h2>Teacher Candidate → WSR Gate → Structural Negatives → Student Runner</h2>
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button onClick={run}>Run Demo Agent Flow</button>
        </div>
        <section>
          <h3>Public Release Boundary</h3>
          <div className="cards">
            <b>Synthetic demo records only</b>
            <b>Private data excluded</b>
            <b>Paper results excluded</b>
            <b>Checkpoints excluded</b>
          </div>
        </section>
        <pre>{out ? JSON.stringify(out, null, 2) : "Run the demo to inspect API events."}</pre>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);