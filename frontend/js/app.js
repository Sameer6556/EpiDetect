/**
 * Frontend logic for the dashboard.
 * This handles all the UI interactions, fetching predictions from the Python API,
 * and drawing the charts using Chart.js.
 */

// Dynamically set API URL. If on Vercel, replace with your Render URL.
// Otherwise, it automatically points to the same server it's hosted on.
const API = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:5000"
    : window.location.origin;



// Grab all the HTML elements we need to interact with
const apiStatusDot = document.querySelector(".status-dot");
const apiStatusText = document.querySelector(".status-text");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const chartWrapper = document.getElementById("chartWrapper");
const btnAnalyze = document.getElementById("btnAnalyze");
const resultDisplay = document.getElementById("resultDisplay");
const confSection = document.getElementById("confidenceSection");
const confValue = document.getElementById("confValue");
const quickStats = document.getElementById("quickStats");
const shapImg = document.getElementById("shapImg");
const shapPlaceholder = document.getElementById("shapPlaceholder");
const gradcamImg = document.getElementById("gradcamImg");
const gradcamPlaceholder = document.getElementById("gradcamPlaceholder");
const waterfallImg = document.getElementById("waterfallImg");
const waterfallPlaceholder = document.getElementById("waterfallPlaceholder");
const permImg = document.getElementById("permImg");
const permPlaceholder = document.getElementById("permPlaceholder");
const shapBarImg = document.getElementById("shapBarImg");
const shapBarPlaceholder = document.getElementById("shapBarPlaceholder");

// Global variables to hold the current app state
let eegChart = null;
let metricsChart = null;
let confChart = null;
let currentEEG = null;    // raw EEG array for analysis
let currentMetrics = {};      // metrics from /metrics endpoint

// Function to check if the Python backend is running
async function checkApiHealth() {
    try {
        const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
        const data = await res.json();
        if (data.status === "ok") {
            apiStatusDot.className = "status-dot online";
            apiStatusText.textContent = data.model_ready ? "API Ready · Model Loaded" : "API Ready · No Model";
        }
    } catch {
        apiStatusDot.className = "status-dot offline";
        apiStatusText.textContent = "API Offline";
    }
}

// Draw the main EEG waveform chart
function renderEEGChart(values, label = "EEG Signal", isSeizure = null) {
    chartWrapper.style.display = "block";
    const canvas = document.getElementById("eegChart");
    const ctx = canvas.getContext("2d");

    if (eegChart) eegChart.destroy();

    const color = isSeizure === true ? "#ef4444"
        : isSeizure === false ? "#10b981"
            : "#7c3aed";

    const n = values.length;
    const labels = Array.from({ length: n }, (_, i) => i);

    eegChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label,
                data: values,
                borderColor: color,
                backgroundColor: color + "15",
                borderWidth: 1.2,
                pointRadius: 0,
                tension: 0.2,
                fill: true,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600, easing: "easeOutQuart" },
            interaction: { mode: "index", intersect: false },
            scales: {
                x: {
                    display: true,
                    ticks: { color: "#475569", maxTicksLimit: 8, font: { size: 10 } },
                    grid: { color: "rgba(255,255,255,0.04)" },
                    title: { display: true, text: "Sample", color: "#475569", font: { size: 10 } },
                },
                y: {
                    display: true,
                    ticks: { color: "#475569", font: { size: 10 } },
                    grid: { color: "rgba(255,255,255,0.04)" },
                    title: { display: true, text: "Amplitude (norm.)", color: "#475569", font: { size: 10 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0f0f1a",
                    borderColor: color,
                    borderWidth: 1,
                    callbacks: {
                        label: ctx => ` ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
        },
    });

    btnAnalyze.disabled = false;

    // Update chart tag
    const tag = document.getElementById("chartTag");
    if (tag) {
        tag.textContent = `${values.length} samples · 173.61 Hz · Bonn`;
    }
}

// Draw the circular confidence score using HTML Canvas
function renderConfidenceRing(pct, isSeizure) {
    const canvas = document.getElementById("confidenceRing");
    const ctx = canvas.getContext("2d");
    const cx = 80, cy = 80, r = 60, lw = 10;

    ctx.clearRect(0, 0, 160, 160);

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = lw;
    ctx.stroke();

    // Filled arc
    const color = isSeizure ? "#ef4444" : "#10b981";
    const end = -Math.PI / 2 + (pct / 100) * 2 * Math.PI;
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, end);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    ctx.stroke();

    // Centre text
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "bold 22px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${pct.toFixed(0)}%`, cx, cy);
}

// Update the UI with the prediction from the backend
function renderResult(data) {
    const isSeizure = data.prediction === "Seizure";

    // Banner
    resultDisplay.innerHTML = `
    <div class="result-banner ${isSeizure ? "seizure" : "normal"}">
      <span class="result-emoji">${isSeizure ? "⚡" : "✅"}</span>
      <div class="result-label">${isSeizure ? "Seizure Detected!" : "Normal EEG"}</div>
      <div class="result-sub">
        ${isSeizure
            ? "Epileptic activity detected — immediate attention recommended"
            : "No seizure activity detected in this segment"}
      </div>
    </div>
  `;

    // Update EEG chart colour
    if (currentEEG && eegChart) {
        renderEEGChart(currentEEG, "EEG Signal", isSeizure);
    }

    // Confidence ring
    confSection.style.display = "flex";
    confValue.textContent = `${data.confidence}%`;
    renderConfidenceRing(data.confidence, isSeizure);

    // Quick stats
    quickStats.style.display = "grid";
    document.getElementById("qPred").textContent = data.prediction;
    document.getElementById("qPred").style.color = isSeizure ? "var(--danger)" : "var(--success)";
    document.getElementById("qProb").textContent = `${(data.probability * 100).toFixed(1)}%`;
    document.getElementById("qConf").textContent = `${data.confidence}%`;

    // XAI images — fetch global charts from backend static endpoints
    loadXAIImages();

    // Per-input SHAP waterfall (from /predict response)
    if (data.shap_waterfall_img) {
        waterfallImg.src = `data:image/png;base64,${data.shap_waterfall_img}`;
        waterfallImg.style.display = "block";
        waterfallPlaceholder.style.display = "none";
    }

    // Hero stat pills
    const metricsResult = currentMetrics.bonn || Object.values(currentMetrics)[0] || {};
    document.getElementById("statAcc").querySelector("span").textContent =
        metricsResult.accuracy
            ? `${(metricsResult.accuracy * 100).toFixed(1)}%`
            : "—";
}

// Fetch the generated plots for the Explain section
async function loadXAIImages() {
    // SHAP summary beeswarm
    try {
        const shapRes = await fetch(`${API}/xai/shap?t=${Date.now()}`);
        if (shapRes.ok) {
            const blob = await shapRes.blob();
            shapImg.src = URL.createObjectURL(blob);
            shapImg.style.display = "block";
            shapPlaceholder.style.display = "none";
        }
    } catch { /* keep placeholder */ }

    // Feature importance bar chart
    try {
        const impRes = await fetch(`${API}/xai/importance?t=${Date.now()}`);
        if (impRes.ok) {
            const blob = await impRes.blob();
            gradcamImg.src = URL.createObjectURL(blob);
            gradcamImg.style.display = "block";
            gradcamPlaceholder.style.display = "none";
        }
    } catch { /* keep placeholder */ }

    // Permutation importance
    try {
        const permRes = await fetch(`${API}/xai/permutation?t=${Date.now()}`);
        if (permRes.ok) {
            const blob = await permRes.blob();
            permImg.src = URL.createObjectURL(blob);
            permImg.style.display = "block";
            permPlaceholder.style.display = "none";
        }
    } catch { /* keep placeholder */ }

    // SHAP bar chart
    try {
        const shapBarRes = await fetch(`${API}/xai/shap-bar?t=${Date.now()}`);
        if (shapBarRes.ok) {
            const blob = await shapBarRes.blob();
            shapBarImg.src = URL.createObjectURL(blob);
            shapBarImg.style.display = "block";
            shapBarPlaceholder.style.display = "none";
        }
    } catch { /* keep placeholder */ }
}

// Send the loaded EEG signal to the API for classification
async function analyzeSignal() {
    if (!currentEEG) {
        alert("Please load or upload an EEG signal first.");
        return;
    }
    showOverlay("Analyzing EEG signal…");

    try {
        const res = await fetch(`${API}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ eeg: Array.from(currentEEG), dataset: "bonn" }),
        });
        const data = await res.json();

        if (data.error) throw new Error(data.error);
        renderResult(data);
    } catch (err) {
        resultDisplay.innerHTML = `
      <div style="color:var(--danger); text-align:center; padding:20px;">
        ⚠️ ${err.message || "Analysis failed. Is the Flask API running?"}
      </div>`;
    } finally {
        hideOverlay();
    }
}

// Load a sample signal when the user clicks the demo buttons
async function loadDemo(isSeizure) {
    showOverlay(isSeizure ? "Loading seizure sample…" : "Loading normal sample…");
    try {
        const res = await fetch(`${API}/demo?seizure=${isSeizure}&dataset=bonn&seed=${Math.floor(Math.random() * 50)}`);
        const data = await res.json();

        if (data.eeg_values && data.eeg_values.length > 0) {
            currentEEG = data.eeg_values;
            // Only render the waveform — do NOT show prediction yet
            renderEEGChart(currentEEG, `EEG Signal (${isSeizure ? "Seizure" : "Normal"} Sample)`);
            clearResult();
            btnAnalyze.disabled = false;
        } else {
            throw new Error("No EEG data returned from API");
        }
    } catch (err) {
        resultDisplay.innerHTML = `
      <div style="color:var(--danger); text-align:center; padding:20px;">
        ⚠️ ${err.message || "Could not load demo. Is the Flask API running?"}
      </div>`;
    } finally {
        hideOverlay();
    }
}

/**
 * Reset the result panel to its initial "waiting" state.
 */
function clearResult() {
    resultDisplay.innerHTML = `
      <div style="height:100px;display:flex;align-items:center;color:var(--text-faint)">
        Signal loaded. Click <strong style="margin:0 0.35em">Process Signal Sequence</strong> to run diagnosis.
      </div>`;
    confSection.style.display = "none";
    // Reset XAI placeholders
    shapImg.style.display = "none";
    shapPlaceholder.style.display = "block";
    gradcamImg.style.display = "none";
    gradcamPlaceholder.style.display = "block";
    waterfallImg.style.display = "none";
    waterfallPlaceholder.style.display = "block";
    if (permImg) { permImg.style.display = "none"; permPlaceholder.style.display = "block"; }
    if (shapBarImg) { shapBarImg.style.display = "none"; shapBarPlaceholder.style.display = "block"; }
}

// Offline synthetic EEG (no API required for basic demo)
function generateOfflineEEG(isSeizure) {
    const n = 256;
    const fs = 173.61;
    const channels = 1;
    const dt = 1 / fs;

    function genChannel(isSz, phaseOffset = 0) {
        const arr = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const t = i * dt;
            if (isSz) {
                const amp = 0.4 + (i / n) * 1.8;
                arr[i] = amp * Math.sin(2 * Math.PI * 4 * t + phaseOffset)
                    + 0.4 * amp * Math.sin(2 * Math.PI * 8 * t + phaseOffset)
                    + (Math.random() - 0.5) * 0.25;
            } else {
                arr[i] = 0.35 * Math.sin(2 * Math.PI * 10 * t + 0.5 + phaseOffset)
                    + 0.25 * Math.sin(2 * Math.PI * 20 * t + 1.0 + phaseOffset)
                    + 0.15 * Math.sin(2 * Math.PI * 6 * t + 2.0 + phaseOffset)
                    + (Math.random() - 0.5) * 0.15;
            }
        }
        return Array.from(arr);
    }

    if (channels > 1) {
        // Return multi-channel as array of arrays
        const result = [];
        for (let ch = 0; ch < channels; ch++) {
            result.push(genChannel(isSeizure, ch * 0.3));
        }
        return result;
    }
    return genChannel(isSeizure);
}

// ═══════════════════════════════════════════════════════════
//  File Upload
// ═══════════════════════════════════════════════════════════
async function handleFile(file) {
    if (!file) return;
    if (!file.name.endsWith(".csv") && !file.name.endsWith(".txt")) {
        alert("Please upload a .csv or .txt file");
        return;
    }

    showOverlay(`Loading ${file.name}…`);

    try {
        const text = await file.text();
        const nums = parseCSV(text);
        if (nums.length > 0) {
            currentEEG = nums;
            renderEEGChart(currentEEG, file.name);
            clearResult();
            btnAnalyze.disabled = false;
        } else {
            throw new Error("Could not parse numeric values from file");
        }
    } catch (err) {
        alert(`File load failed: ${err.message}`);
    } finally {
        hideOverlay();
    }
}

function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const nums = [];
    for (const line of lines) {
        // Handle both comma-separated and single-value-per-line (Bonn .txt format)
        for (const cell of line.split(/[,\t\s]+/)) {
            const v = parseFloat(cell.trim());
            if (!isNaN(v)) nums.push(v);
        }
    }
    return nums;
}

// ═══════════════════════════════════════════════════════════
//  Hero animated wave
// ═══════════════════════════════════════════════════════════
function animateHeroWave() {
    const canvas = document.getElementById("heroWave");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.parentElement.offsetWidth;
    const H = 80;
    canvas.width = W; canvas.height = H;
    let t = 0;

    function draw() {
        ctx.clearRect(0, 0, W, H);
        const grad = ctx.createLinearGradient(0, 0, W, 0);
        grad.addColorStop(0, "rgba(124,58,237,0.8)");
        grad.addColorStop(0.5, "rgba(6,182,212,0.8)");
        grad.addColorStop(1, "rgba(124,58,237,0.8)");

        ctx.beginPath();
        ctx.moveTo(0, H / 2);
        for (let x = 0; x < W; x++) {
            const tx = x / W;
            const y = H / 2
                + Math.sin((tx * 8 + t) * Math.PI * 2) * 12
                + Math.sin((tx * 3 + t * 0.7) * Math.PI * 2) * 8
                + Math.sin((tx * 15 + t * 1.3) * Math.PI * 2) * 4;
            ctx.lineTo(x, y);
        }
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2;
        ctx.stroke();

        t += 0.008;
        requestAnimationFrame(draw);
    }
    draw();
}

// ═══════════════════════════════════════════════════════════
//  Metrics
// ═══════════════════════════════════════════════════════════
async function loadMetrics() {
    try {
        const res = await fetch(`${API}/metrics`);
        const data = await res.json();

        // Ensure there are actual metrics before rendering
        if (data && data.results) {
            currentMetrics = {
                bonn: {
                    accuracy: (data.results.Accuracy_mean || 0) / 100,
                    sensitivity: (data.results.Recall_mean || 0) / 100,
                    specificity: (data.results.Specificity_mean || 0) / 100,
                    precision: (data.results.Precision_mean || 0) / 100,
                    f1: (data.results.F1_Score_mean || 0) / 100,
                    roc_auc: (data.results.AUC_ROC_mean || 0) / 100,
                    kappa: (data.results.Kappa_mean || 0) / 100
                }
            };
            renderMetrics();
        } else if (data && Object.keys(data).length > 0) {
            currentMetrics = data;
            renderMetrics();
        } else {
            throw new Error("No metrics. Train model to see results.");
        }
    } catch {
        currentMetrics = {};
        const grid = document.getElementById("metricsGrid");
        if (grid) grid.innerHTML = "<p style='color:var(--text-faint)'>No metrics found in backend. Please train the model to generate evaluation metrics.</p>";
    }
}

function renderMetrics() {
    const m = currentMetrics.bonn || Object.values(currentMetrics)[0];
    const grid = document.getElementById("metricsGrid");
    if (!m) { grid.innerHTML = "<p style='color:var(--text-faint)'>No metrics available yet.</p>"; return; }

    const items = [
        { key: "accuracy", label: "Accuracy" },
        { key: "sensitivity", label: "Sensitivity" },
        { key: "specificity", label: "Specificity" },
        { key: "precision", label: "Precision" },
        { key: "f1", label: "F1-Score" },
        { key: "roc_auc", label: "ROC-AUC" },
        { key: "kappa", label: "Cohen's Kappa" },
    ];

    grid.innerHTML = items
        .filter(it => m[it.key] !== undefined)
        .map(({ key, label }) => {
            const v = m[key];
            const pct = (v * 100).toFixed(2);
            return `
        <div class="metric-card">
          <div class="m-label">${label}</div>
          <div class="m-value">${pct}%</div>
          <div class="m-bar"><div class="m-fill" style="width:${pct}%"></div></div>
        </div>`;
        }).join("");

    // Update hero stats
    if (m.accuracy) document.getElementById("statAcc").querySelector("span").textContent = `${(m.accuracy * 100).toFixed(1)}%`;
    if (m.sensitivity) document.getElementById("statSens").querySelector("span").textContent = `${(m.sensitivity * 100).toFixed(1)}%`;
    if (m.roc_auc) document.getElementById("statAUC").querySelector("span").textContent = `${(m.roc_auc * 100).toFixed(1)}%`;

    renderMetricsBar(m);
}

function renderMetricsBar(m) {
    const canvas = document.getElementById("metricsBarChart");
    if (!canvas) return;
    if (metricsChart) metricsChart.destroy();

    const labels = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "Kappa"];
    const values = [m.accuracy, m.sensitivity, m.specificity, m.precision, m.f1, m.roc_auc, m.kappa]
        .map(v => v !== undefined ? +(v * 100).toFixed(2) : 0);

    metricsChart = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Score (%)",
                data: values,
                backgroundColor: [
                    "rgba(124,58,237,0.7)", "rgba(6,182,212,0.7)", "rgba(16,185,129,0.7)",
                    "rgba(245,158,11,0.7)", "rgba(239,68,68,0.7)", "rgba(139,92,246,0.7)",
                    "rgba(56,189,248,0.7)",
                ],
                borderRadius: 6,
                borderSkipped: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: "easeOutQuart" },
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#94a3b8", font: { size: 11 } } },
                y: { min: 80, max: 100, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#94a3b8", callback: v => `${v}%` } },
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => ` ${c.parsed.y.toFixed(2)}%` } },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════
//  Overlay helpers
// ═══════════════════════════════════════════════════════════
function showOverlay(msg = "Processing…") {
    loadingText.textContent = msg;
    loadingOverlay.style.display = "flex";
}
function hideOverlay() { loadingOverlay.style.display = "none"; }

// ═══════════════════════════════════════════════════════════
//  Event Listeners
// ═══════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    // API check
    checkApiHealth();
    setInterval(checkApiHealth, 30000);

    // Fade-in animations on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("active");
                observer.unobserve(entry.target); // only animate once
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll(".reveal").forEach(el => observer.observe(el));

    // Start hero wave
    animateHeroWave();

    // Load metrics
    loadMetrics();

    // Demo buttons
    document.getElementById("btnDemoSeizure").addEventListener("click", () => loadDemo(true));
    document.getElementById("btnDemoNormal").addEventListener("click", () => loadDemo(false));

    // Analyze button
    btnAnalyze.addEventListener("click", analyzeSignal);

    // Remove datasetSelector block

    // File input
    document.getElementById("fileInput").addEventListener("change", e => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    // Drag-and-drop
    const uploadZone = document.getElementById("uploadZone");
    uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
    uploadZone.addEventListener("drop", e => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });



    // Metrics dataset tabs
    document.querySelectorAll(".mtab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".mtab-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeDataset = btn.dataset.mtab;
            renderMetrics(activeDataset);
        });
    });

    // Smooth nav scroll
    document.querySelectorAll(".nav-link").forEach(link => {
        link.addEventListener("click", e => {
            const href = link.getAttribute("href");
            if (href && href.startsWith("#")) {
                e.preventDefault();
                document.querySelector(href)?.scrollIntoView({ behavior: "smooth" });
                document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
                link.classList.add("active");
            }
        });
    });
});
