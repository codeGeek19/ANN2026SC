"""
EnergyLVQ — Flask Web Application
Learning Vector Quantization Classifier
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# ── Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lvq_energy_model.pkl")
with open(MODEL_PATH, "rb") as f:
    _model = pickle.load(f)

PROTOTYPES   = _model["prototypes"]
PROTO_LABELS = _model["proto_labels"]
SCALER       = _model["scaler"]


def lvq_predict(raw_18):
    scaled = SCALER.transform([raw_18])[0]
    x = scaled[:10]
    dists = [float(np.linalg.norm(x - p)) for p in PROTOTYPES]
    best_idx = int(np.argmin(dists))
    label_int = int(PROTO_LABELS[best_idx])
    label_str = ["LOW", "MEDIUM", "HIGH"][label_int]
    confidence = round(1.0 / (1.0 + dists[best_idx]), 4)
    return label_int, label_str, confidence, dists


app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EnergyLVQ</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #0d0d0d;
  --panel:    #141414;
  --border:   #2a2a2a;
  --lime:     #c8f135;
  --text:     #e0e0e0;
  --muted:    #555;
  --input-bg: #1a1a1a;
  --font:     'IBM Plex Mono', monospace;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  min-height: 100vh;
  font-size: 13px;
}

/* ── Header ── */
header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 32px;
  border-bottom: 1px solid var(--border);
}

.logo-icon {
  width: 32px; height: 32px;
  background: var(--lime);
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  flex-shrink: 0;
}

.logo-text-wrap { display: flex; flex-direction: column; gap: 2px; }

.logo-name {
  font-size: 15px;
  font-weight: 600;
  color: var(--lime);
  letter-spacing: 0.03em;
}

.logo-sub {
  font-size: 9px;
  letter-spacing: 0.22em;
  color: var(--muted);
  text-transform: uppercase;
}

/* ── Main ── */
main { padding: 32px; max-width: 1200px; }

.section-heading {
  font-size: 9px;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 20px;
}

/* ── Zone block ── */
.zone-block { margin-bottom: 28px; }

.zone-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text);
  margin-bottom: 10px;
}

.zone-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--lime);
  flex-shrink: 0;
}

.zone-rule { flex: 1; height: 1px; background: var(--border); }

/* ── 4-col fields row ── */
.fields-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.field { display: flex; flex-direction: column; gap: 6px; }

.field-label {
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 0.04em;
}

/* ── Number input box ── */
.num-wrap {
  position: relative;
  display: flex;
  align-items: center;
}

.num-wrap input[type="number"] {
  width: 100%;
  background: var(--input-bg);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  padding: 10px 36px 10px 12px;
  outline: none;
  -moz-appearance: textfield;
  appearance: textfield;
  transition: border-color 0.15s;
}

.num-wrap input[type="number"]::-webkit-inner-spin-button,
.num-wrap input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; }

.num-wrap input[type="number"]:focus { border-color: var(--lime); }

.num-unit {
  position: absolute;
  right: 10px;
  font-size: 10px;
  color: var(--muted);
  pointer-events: none;
}

/* ── Slider input ── */
.slider-wrap {
  background: var(--input-bg);
  border: 1px solid var(--border);
  padding: 8px 12px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.slider-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.slider-range-label { font-size: 9px; color: var(--muted); }
.slider-val { font-size: 14px; color: var(--text); }

input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 2px;
  background: var(--border);
  outline: none;
  cursor: pointer;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: var(--lime);
  cursor: pointer;
  border: none;
}

input[type="range"]::-moz-range-thumb {
  width: 14px; height: 14px;
  border-radius: 50%;
  background: var(--lime);
  cursor: pointer;
  border: none;
}

/* ── Toggle switch ── */
.toggle-wrap {
  background: var(--input-bg);
  border: 1px solid var(--border);
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
  height: 44px;
}

.toggle-track {
  width: 34px; height: 18px;
  border-radius: 9px;
  background: var(--border);
  position: relative;
  flex-shrink: 0;
  transition: background 0.2s;
}

.toggle-track.on { background: var(--lime); }

.toggle-thumb {
  position: absolute;
  top: 3px; left: 3px;
  width: 12px; height: 12px;
  border-radius: 50%;
  background: #111;
  transition: transform 0.2s;
}

.toggle-track.on .toggle-thumb { transform: translateX(16px); }

.toggle-text { font-size: 13px; color: var(--muted); }
.toggle-text.on { color: var(--text); }

/* ── Building & Schedule 6-col ── */
.sched-row {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
}

/* ── Run button ── */
.run-section {
  margin-top: 32px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.run-btn {
  background: var(--lime);
  color: #0d0d0d;
  border: none;
  padding: 14px 28px;
  font-family: var(--font);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.05em;
  cursor: pointer;
  transition: opacity 0.15s, transform 0.1s;
}

.run-btn:hover  { opacity: 0.85; }
.run-btn:active { transform: scale(0.98); }

.run-hint {
  font-size: 11px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 6px;
}

.run-hint kbd {
  border: 1px solid var(--border);
  padding: 2px 7px;
  font-family: var(--font);
  font-size: 11px;
  color: var(--text);
  border-radius: 2px;
}

/* ── Result banner ── */
.result-banner {
  display: none;
  margin-top: 28px;
  border: 1px solid var(--border);
  background: var(--input-bg);
  padding: 20px 24px;
  gap: 32px;
  align-items: center;
}

.result-banner.show { display: flex; }

.result-class {
  font-size: 28px;
  font-weight: 600;
  letter-spacing: 0.08em;
}

.result-class.low  { color: #3af0a0; }
.result-class.med  { color: var(--lime); }
.result-class.high { color: #f05050; }

.result-meta { display: flex; flex-direction: column; gap: 4px; }

.result-meta-item { font-size: 10px; color: var(--muted); letter-spacing: 0.08em; }
.result-meta-item strong { color: var(--text); font-weight: 500; }

.result-dist-section {
  margin-left: auto;
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 220px;
}

.dist-mini-row { display: flex; align-items: center; gap: 6px; }

.dist-mini-label { font-size: 9px; width: 18px; flex-shrink: 0; }
.dist-mini-label.c0 { color: #3af0a0; }
.dist-mini-label.c1 { color: var(--lime); }
.dist-mini-label.c2 { color: #f05050; }

.dist-mini-track { flex: 1; height: 3px; background: var(--border); }

.dist-mini-fill { height: 100%; transition: width 0.35s ease; }
.dist-mini-fill.c0 { background: #3af0a0; }
.dist-mini-fill.c1 { background: var(--lime); }
.dist-mini-fill.c2 { background: #f05050; }

.dist-mini-num { font-size: 9px; color: var(--muted); width: 28px; text-align: right; }
</style>
</head>
<body>

<header>
  <div class="logo-icon"></div>
  <div class="logo-text-wrap">
    <span class="logo-name">EnergyLVQ</span>
    <span class="logo-sub">Learning Vector Quantization Classifier</span>
  </div>
</header>

<main>
  <div class="section-heading">Input Features</div>

  <!-- Zone 1 -->
  <div class="zone-block">
    <div class="zone-label">
      <span class="zone-dot"></span>Zone 1 – HVAC &amp; Electrical<span class="zone-rule"></span>
    </div>
    <div class="fields-row">
      <div class="field">
        <div class="field-label">Indoor Temp Zone 1</div>
        <div class="num-wrap">
          <input type="number" id="f0" value="24" min="22.9" max="35.0" step="0.1">
          <span class="num-unit">°C</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Current Draw Zone 1</div>
        <div class="num-wrap">
          <input type="number" id="f1" value="0.017" min="0.012" max="0.023" step="0.001">
          <span class="num-unit">A</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Power Factor Zone 1</div>
        <div class="num-wrap">
          <input type="number" id="f2" value="0.2" min="0.0" max="0.52" step="0.01">
        </div>
      </div>
      <div class="field">
        <div class="field-label">Power Load Zone 1</div>
        <div class="num-wrap">
          <input type="number" id="f3" value="10" min="0.009" max="39.2" step="0.1">
          <span class="num-unit">kW</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Zone 2 -->
  <div class="zone-block">
    <div class="zone-label">
      <span class="zone-dot"></span>Zone 2 – HVAC &amp; Electrical<span class="zone-rule"></span>
    </div>
    <div class="fields-row">
      <div class="field">
        <div class="field-label">Indoor Temp Zone 2</div>
        <div class="num-wrap">
          <input type="number" id="f4" value="24" min="19.8" max="39.1" step="0.1">
          <span class="num-unit">°C</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Current Draw Zone 2</div>
        <div class="num-wrap">
          <input type="number" id="f5" value="0.016" min="0.010" max="0.022" step="0.001">
          <span class="num-unit">A</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Power Factor Zone 2</div>
        <div class="num-wrap">
          <input type="number" id="f6" value="0.18" min="0.0" max="0.48" step="0.01">
        </div>
      </div>
      <div class="field">
        <div class="field-label">Power Load Zone 2</div>
        <div class="num-wrap">
          <input type="number" id="f7" value="8" min="0.060" max="24.5" step="0.1">
          <span class="num-unit">kW</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Zone 3 -->
  <div class="zone-block">
    <div class="zone-label">
      <span class="zone-dot"></span>Zone 3 – HVAC &amp; Electrical<span class="zone-rule"></span>
    </div>
    <div class="fields-row">
      <div class="field">
        <div class="field-label">Indoor Temp Zone 3</div>
        <div class="num-wrap">
          <input type="number" id="f8" value="23" min="19.9" max="34.2" step="0.1">
          <span class="num-unit">°C</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Current Draw Zone 3</div>
        <div class="num-wrap">
          <input type="number" id="f9" value="0.015" min="0.010" max="0.021" step="0.001">
          <span class="num-unit">A</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Power Factor Zone 3</div>
        <div class="num-wrap">
          <input type="number" id="f10" value="0.17" min="0.0" max="0.48" step="0.01">
        </div>
      </div>
      <div class="field">
        <div class="field-label">Power Load Zone 3</div>
        <div class="num-wrap">
          <input type="number" id="f11" value="4" min="0.015" max="10.3" step="0.1">
          <span class="num-unit">kW</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Building & Schedule -->
  <div class="zone-block">
    <div class="zone-label">
      <span class="zone-dot"></span>Building &amp; Schedule<span class="zone-rule"></span>
    </div>
    <div class="sched-row">

      <div class="field">
        <div class="field-label">Occupancy Count (ppl)</div>
        <div class="slider-wrap">
          <div class="slider-top">
            <span class="slider-range-label">0–22</span>
            <span class="slider-val" id="v12">5</span>
          </div>
          <input type="range" id="f12" min="0" max="22" step="1" value="5"
            oninput="document.getElementById('v12').textContent=this.value">
        </div>
      </div>

      <div class="field">
        <div class="field-label">HVAC Cooling Active</div>
        <div class="toggle-wrap" onclick="toggleSwitch('f13','t13','tl13')">
          <div class="toggle-track" id="t13"><div class="toggle-thumb"></div></div>
          <span class="toggle-text" id="tl13">Off</span>
          <input type="hidden" id="f13" value="0">
        </div>
      </div>

      <div class="field">
        <div class="field-label">HVAC Heating Active</div>
        <div class="toggle-wrap" onclick="toggleSwitch('f14','t14','tl14')">
          <div class="toggle-track" id="t14"><div class="toggle-thumb"></div></div>
          <span class="toggle-text" id="tl14">Off</span>
          <input type="hidden" id="f14" value="0">
        </div>
      </div>

      <div class="field">
        <div class="field-label">Hour of Day (h)</div>
        <div class="slider-wrap">
          <div class="slider-top">
            <span class="slider-range-label">0–23</span>
            <span class="slider-val" id="v15">12</span>
          </div>
          <input type="range" id="f15" min="0" max="23" step="1" value="12"
            oninput="document.getElementById('v15').textContent=this.value">
        </div>
      </div>

      <div class="field">
        <div class="field-label">Day of Week</div>
        <div class="slider-wrap">
          <div class="slider-top">
            <span class="slider-range-label">0–6</span>
            <span class="slider-val" id="v16">2</span>
          </div>
          <input type="range" id="f16" min="0" max="6" step="1" value="2"
            oninput="document.getElementById('v16').textContent=this.value">
        </div>
      </div>

      <div class="field">
        <div class="field-label">Month</div>
        <div class="slider-wrap">
          <div class="slider-top">
            <span class="slider-range-label">1–12</span>
            <span class="slider-val" id="v17">6</span>
          </div>
          <input type="range" id="f17" min="1" max="12" step="1" value="6"
            oninput="document.getElementById('v17').textContent=this.value">
        </div>
      </div>

    </div>
  </div>

  <!-- Run -->
  <div class="run-section">
    <button class="run-btn" onclick="runPredict()">Run Prediction</button>
    <span class="run-hint">or press <kbd>Enter</kbd></span>
  </div>

  <!-- Result -->
  <div class="result-banner" id="result-banner">
    <div>
      <div style="font-size:9px;letter-spacing:0.2em;color:var(--muted);margin-bottom:6px;text-transform:uppercase">Energy Class</div>
      <div class="result-class" id="result-class">—</div>
    </div>
    <div class="result-meta">
      <div class="result-meta-item">Confidence &nbsp;<strong id="result-conf">—</strong></div>
      <div class="result-meta-item">Nearest dist &nbsp;<strong id="result-dist">—</strong></div>
      <div class="result-meta-item">Model &nbsp;<strong>15 prototypes · 3 classes</strong></div>
    </div>
    <div class="result-dist-section" id="dist-section"></div>
  </div>

</main>

<script>
  function toggleSwitch(inputId, trackId, labelId) {
    const input = document.getElementById(inputId);
    const track = document.getElementById(trackId);
    const label = document.getElementById(labelId);
    const isOn  = input.value === '1';
    input.value = isOn ? '0' : '1';
    track.classList.toggle('on', !isOn);
    label.textContent = isOn ? 'Off' : 'On';
    label.classList.toggle('on', !isOn);
  }

  function getFeatures() {
    return Array.from({length: 18}, (_, i) =>
      parseFloat(document.getElementById('f' + i).value) || 0
    );
  }

  async function runPredict() {
    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({features: getFeatures()})
      });
      const data = await resp.json();
      if (data.error) { alert(data.error); return; }

      const clsMap = {0: 'low', 1: 'med', 2: 'high'};
      const cls    = clsMap[data.label_int];

      document.getElementById('result-class').textContent = data.label_str;
      document.getElementById('result-class').className   = 'result-class ' + cls;
      document.getElementById('result-conf').textContent  = (data.confidence * 100).toFixed(1) + '%';
      document.getElementById('result-dist').textContent  = data.nearest_dist.toFixed(3);
      document.getElementById('result-banner').className  = 'result-banner show';

      const maxD     = Math.max(...data.distances);
      const clsNames = ['c0','c1','c2'];
      const clsLbls  = ['L','M','H'];
      let html = '';
      data.distances.forEach((d, i) => {
        const pct = maxD > 0 ? (d / maxD * 100).toFixed(1) : 0;
        const c   = clsNames[data.proto_labels[i]];
        html += `<div class="dist-mini-row">
          <span class="dist-mini-label ${c}">${clsLbls[data.proto_labels[i]]}${i%5+1}</span>
          <div class="dist-mini-track"><div class="dist-mini-fill ${c}" style="width:${pct}%"></div></div>
          <span class="dist-mini-num">${d.toFixed(2)}</span>
        </div>`;
      });
      document.getElementById('dist-section').innerHTML = html;

    } catch(e) { alert('Error: ' + e.message); }
  }

  document.addEventListener('keydown', e => { if (e.key === 'Enter') runPredict(); });
  window.addEventListener('DOMContentLoaded', runPredict);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        raw = [float(x) for x in data["features"]]
        if len(raw) != 18:
            return jsonify({"error": f"Expected 18 features, got {len(raw)}"}), 400
        label_int, label_str, confidence, dists = lvq_predict(raw)
        return jsonify({
            "label_int":    label_int,
            "label_str":    label_str,
            "confidence":   confidence,
            "nearest_dist": min(dists),
            "distances":    dists,
            "proto_labels": [int(x) for x in PROTO_LABELS],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  EnergyLVQ  →  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
