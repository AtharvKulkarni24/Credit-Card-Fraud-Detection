document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let sampleTransactions = [];
  let currentFeatures = {};
  let benchmarkMetrics = null;

  // DOM Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const presetChipsContainer = document.getElementById('presetChips');
  const pcaGrid = document.getElementById('pcaGrid');
  const togglePcaBtn = document.getElementById('togglePcaBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');

  // Key Slider Inputs
  const amountInput = document.getElementById('amountInput');
  const amountSlider = document.getElementById('amountSlider');
  const v14Slider = document.getElementById('v14Slider');
  const v12Slider = document.getElementById('v12Slider');
  const v10Slider = document.getElementById('v10Slider');
  const v17Slider = document.getElementById('v17Slider');

  // Gauge & Results Elements
  const riskGaugeFill = document.getElementById('riskGaugeFill');
  const riskNum = document.getElementById('riskNum');
  const riskLabel = document.getElementById('riskLabel');
  const statusBanner = document.getElementById('statusBanner');
  const statusBadge = document.getElementById('statusBadge');
  const statusDesc = document.getElementById('statusDesc');
  const anomalyScoreVal = document.getElementById('anomalyScoreVal');
  const supervisedProbVal = document.getElementById('supervisedProbVal');
  const driversList = document.getElementById('driversList');

  // Batch Elements
  const dropzone = document.getElementById('dropzone');
  const batchFileInput = document.getElementById('batchFileInput');
  const batchResultsSection = document.getElementById('batchResultsSection');
  const exportBatchBtn = document.getElementById('exportBatchBtn');

  // ==========================================
  // 1. Tab Navigation
  // ==========================================
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');

      if (tabId === 'benchmarksTab' && !benchmarkMetrics) {
        fetchMetrics();
      }
    });
  });

  // Toggle PCA grid
  if (togglePcaBtn && pcaGrid) {
    togglePcaBtn.addEventListener('click', () => {
      pcaGrid.classList.toggle('open');
      togglePcaBtn.innerHTML = pcaGrid.classList.contains('open')
        ? 'Hide Advanced PCA Features (V1 - V28) ▲'
        : 'Show All 28 PCA Features (V1 - V28) ▼';
    });
  }

  // ==========================================
  // 2. Initialize PCA Inputs & Features
  // ==========================================
  function initPcaGrid() {
    pcaGrid.innerHTML = '';
    for (let i = 1; i <= 28; i++) {
      const col = `V${i}`;
      currentFeatures[col] = currentFeatures[col] || 0.0;
      
      const div = document.createElement('div');
      div.innerHTML = `
        <label style="font-size:0.72rem; color:var(--text-dim); display:block; margin-bottom:2px;">${col}</label>
        <input type="number" step="0.01" class="pca-mini-input" id="input_${col}" value="${currentFeatures[col].toFixed(2)}">
      `;
      pcaGrid.appendChild(div);

      div.querySelector('input').addEventListener('input', (e) => {
        currentFeatures[col] = parseFloat(e.target.value) || 0.0;
        syncKeySlidersFromFeatures();
      });
    }
  }

  // ==========================================
  // 3. Load Samples from API
  // ==========================================
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) throw new Error('Failed to fetch samples');
      sampleTransactions = await res.json();
      renderPresetChips();

      // Automatically load the first sample (Normal)
      if (sampleTransactions.length > 0) {
        applySample(sampleTransactions[0], 0);
      }
    } catch (err) {
      console.error('Error loading samples:', err);
      presetChipsContainer.innerHTML = `<div style="color:var(--text-dim); font-size:0.85rem;">Failed to load sample transactions.</div>`;
    }
  }

  function renderPresetChips() {
    presetChipsContainer.innerHTML = '';
    sampleTransactions.forEach((sample, idx) => {
      const chip = document.createElement('div');
      chip.className = `preset-chip ${idx === 0 ? 'active' : ''}`;
      chip.id = `chip-${idx}`;
      const isFraud = sample.is_fraud === 1;
      
      chip.innerHTML = `
        <div class="chip-title">${sample.title}</div>
        <div class="chip-sub">
          <span>$${sample.amount.toFixed(2)}</span>
          <span class="${isFraud ? 'chip-tag-fraud' : 'chip-tag-norm'}">
            ${isFraud ? '⚠️ FRAUD' : '✓ NORMAL'}
          </span>
        </div>
      `;

      chip.addEventListener('click', () => applySample(sample, idx));
      presetChipsContainer.appendChild(chip);
    });
  }

  function applySample(sample, activeIdx) {
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    const activeChip = document.getElementById(`chip-${activeIdx}`);
    if (activeChip) activeChip.classList.add('active');

    // Populate current features
    currentFeatures = { ...sample.features };

    // Update controls
    const amt = sample.amount || currentFeatures['Amount'] || 0;
    amountInput.value = amt.toFixed(2);
    amountSlider.value = Math.min(2500, Math.max(0, amt));
    document.getElementById('amountSliderVal').textContent = `$${amt.toFixed(2)}`;

    syncKeySlidersFromFeatures();
    updatePcaInputs();

    // Trigger analysis
    analyzeCurrentTransaction();
  }

  function syncKeySlidersFromFeatures() {
    const v14 = currentFeatures['V14'] || 0;
    v14Slider.value = v14;
    document.getElementById('v14Val').textContent = v14.toFixed(2);

    const v12 = currentFeatures['V12'] || 0;
    v12Slider.value = v12;
    document.getElementById('v12Val').textContent = v12.toFixed(2);

    const v10 = currentFeatures['V10'] || 0;
    v10Slider.value = v10;
    document.getElementById('v10Val').textContent = v10.toFixed(2);

    const v17 = currentFeatures['V17'] || 0;
    v17Slider.value = v17;
    document.getElementById('v17Val').textContent = v17.toFixed(2);
  }

  function updatePcaInputs() {
    for (let i = 1; i <= 28; i++) {
      const input = document.getElementById(`input_V${i}`);
      if (input && currentFeatures[`V${i}`] !== undefined) {
        input.value = currentFeatures[`V${i}`].toFixed(2);
      }
    }
  }

  // ==========================================
  // 4. Interactive Sliders Binding
  // ==========================================
  amountSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    amountInput.value = val.toFixed(2);
    currentFeatures['Amount'] = val;
    document.getElementById('amountSliderVal').textContent = `$${val.toFixed(2)}`;
  });

  amountInput.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value) || 0;
    amountSlider.value = Math.min(2500, Math.max(0, val));
    currentFeatures['Amount'] = val;
    document.getElementById('amountSliderVal').textContent = `$${val.toFixed(2)}`;
  });

  function bindSlider(slider, feature, labelId) {
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      currentFeatures[feature] = val;
      document.getElementById(labelId).textContent = val.toFixed(2);
      const miniInput = document.getElementById(`input_${feature}`);
      if (miniInput) miniInput.value = val.toFixed(2);
    });
  }

  bindSlider(v14Slider, 'V14', 'v14Val');
  bindSlider(v12Slider, 'V12', 'v12Val');
  bindSlider(v10Slider, 'V10', 'v10Val');
  bindSlider(v17Slider, 'V17', 'v17Val');

  analyzeBtn.addEventListener('click', analyzeCurrentTransaction);

  // ==========================================
  // 5. Run Prediction & Update UI
  // ==========================================
  async function analyzeCurrentTransaction() {
    try {
      analyzeBtn.innerHTML = `<span>Analyzing...</span>`;
      analyzeBtn.disabled = true;

      const payload = {
        features: {
          Time: currentFeatures['Time'] || 1000.0,
          Amount: parseFloat(amountInput.value) || 0.0,
          ...currentFeatures
        }
      };

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      renderAnalysisResult(data);
    } catch (err) {
      console.error('Error analyzing transaction:', err);
    } finally {
      analyzeBtn.innerHTML = `<span>⚡ Analyze Transaction In Real-Time</span>`;
      analyzeBtn.disabled = false;
    }
  }

  function renderAnalysisResult(data) {
    const risk = data.composite_risk_score;
    riskNum.textContent = `${risk.toFixed(1)}%`;
    riskLabel.textContent = `${data.risk_level} RISK`;
    riskLabel.style.color = data.badge_color;

    // SVG Gauge calculation: stroke-dasharray = 565
    // offset = 565 - (565 * risk / 100)
    const dashoffset = 565 - (565 * (risk / 100));
    riskGaugeFill.style.strokeDashoffset = dashoffset;
    riskGaugeFill.style.stroke = data.badge_color;

    // Status Banner
    statusBanner.style.borderColor = data.badge_color;
    statusBanner.style.background = `${data.badge_color}18`;
    statusBadge.textContent = data.status;
    statusBadge.style.background = data.badge_color;
    statusDesc.textContent = data.action;

    // Dual Engine Values
    anomalyScoreVal.textContent = `${data.anomaly_score.toFixed(1)}%`;
    supervisedProbVal.textContent = `${data.supervised_probability.toFixed(1)}%`;

    // Explainability Drivers
    driversList.innerHTML = '';
    if (data.top_drivers && data.top_drivers.length > 0) {
      data.top_drivers.forEach(d => {
        const isSpike = d.z_score > 0;
        const item = document.createElement('div');
        item.className = 'driver-item';
        item.innerHTML = `
          <div class="driver-row">
            <span class="driver-code">${d.feature}: ${d.value}</span>
            <span class="driver-z" style="background:${Math.abs(d.z_score) > 2 ? 'rgba(239,68,68,0.18)' : 'rgba(245,158,11,0.18)'}; color:${Math.abs(d.z_score) > 2 ? 'var(--status-danger)' : 'var(--status-warn)'};">
              Z = ${d.z_score > 0 ? '+' : ''}${d.z_score} (${d.direction})
            </span>
          </div>
          <div class="driver-desc">${d.description} (Normal μ: ${d.normal_mean})</div>
        `;
        driversList.appendChild(item);
      });
    }
  }

  // ==========================================
  // 6. Batch CSV Audit Handler
  // ==========================================
  if (dropzone && batchFileInput) {
    dropzone.addEventListener('click', () => batchFileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'var(--accent-cyan)';
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.style.borderColor = 'rgba(6, 182, 212, 0.4)';
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'rgba(6, 182, 212, 0.4)';
      if (e.dataTransfer.files.length > 0) {
        uploadBatchFile(e.dataTransfer.files[0]);
      }
    });

    batchFileInput.addEventListener('change', () => {
      if (batchFileInput.files.length > 0) {
        uploadBatchFile(batchFileInput.files[0]);
      }
    });
  }

  async function uploadBatchFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      dropzone.innerHTML = `<p style="color:var(--accent-cyan); font-weight:600;">Auditing batch transactions with Isolation Forest...</p>`;
      const res = await fetch('/api/predict/batch', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Failed batch upload');
      }

      const data = await res.json();
      renderBatchResults(data);
    } catch (err) {
      alert(`Batch Error: ${err.message}`);
    } finally {
      dropzone.innerHTML = `
        <div style="font-size:2rem; margin-bottom:0.5rem;">📂</div>
        <div style="font-weight:700; font-size:1.1rem; color:#fff;">Drag & Drop CSV File to Audit</div>
        <div style="color:var(--text-dim); font-size:0.85rem; margin-top:0.25rem;">Supports standard creditcard.csv or custom batch transaction files</div>
      `;
    }
  }

  function renderBatchResults(data) {
    batchResultsSection.style.display = 'block';
    const sum = data.summary;

    document.getElementById('bTotal').textContent = sum.total_transactions.toLocaleString();
    document.getElementById('bFraud').textContent = sum.fraud_count.toLocaleString();
    document.getElementById('bSuspicious').textContent = sum.suspicious_count.toLocaleString();
    document.getElementById('bSaved').textContent = `$${sum.fraud_amount_prevented.toLocaleString()}`;

    // Render Table Preview
    const tbody = document.getElementById('batchPreviewTbody');
    tbody.innerHTML = '';
    data.flagged_preview.forEach((row, i) => {
      const tr = document.createElement('tr');
      const isFraud = row.Prediction === 'FRAUD';
      tr.innerHTML = `
        <td>#${i + 1}</td>
        <td>$${parseFloat(row.Amount || 0).toFixed(2)}</td>
        <td>${row.Anomaly_Score}%</td>
        <td>${row.Supervised_Prob}%</td>
        <td>${row.Risk_Score}%</td>
        <td>
          <span class="badge-pill" style="background:${isFraud ? 'var(--status-danger-bg)' : 'var(--status-warn-bg)'}; color:${isFraud ? 'var(--status-danger)' : 'var(--status-warn)'};">
            ${row.Prediction}
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });

    if (exportBatchBtn) {
      exportBatchBtn.style.display = 'inline-flex';
      exportBatchBtn.onclick = () => window.open('/api/export/batch', '_blank');
    }
  }

  // ==========================================
  // 7. Benchmarks & Model Metrics
  // ==========================================
  async function fetchMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (!res.ok) throw new Error('Failed to load metrics');
      benchmarkMetrics = await res.json();
      renderMetrics(benchmarkMetrics);
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  }

  function renderMetrics(metrics) {
    const ifM = metrics.isolation_forest;
    const rfM = metrics.random_forest;

    if (ifM) {
      document.getElementById('ifRocAuc').textContent = ifM.roc_auc;
      document.getElementById('ifPrAuc').textContent = ifM.pr_auc;
      document.getElementById('ifRecall').textContent = `${(ifM.recall * 100).toFixed(1)}%`;
      document.getElementById('ifLatency').textContent = `${ifM.latency_ms} ms`;

      // Fill confusion matrix
      const cm = ifM.confusion_matrix;
      document.getElementById('ifTN').textContent = cm.true_negatives.toLocaleString();
      document.getElementById('ifFP').textContent = cm.false_positives.toLocaleString();
      document.getElementById('ifFN').textContent = cm.false_negatives.toLocaleString();
      document.getElementById('ifTP').textContent = cm.true_positives.toLocaleString();
    }

    if (rfM) {
      document.getElementById('rfRocAuc').textContent = rfM.roc_auc;
      document.getElementById('rfPrAuc').textContent = rfM.pr_auc;
      document.getElementById('rfRecall').textContent = `${(rfM.recall * 100).toFixed(1)}%`;
      document.getElementById('rfLatency').textContent = `${rfM.latency_ms} ms`;

      const cm = rfM.confusion_matrix;
      document.getElementById('rfTN').textContent = cm.true_negatives.toLocaleString();
      document.getElementById('rfFP').textContent = cm.false_positives.toLocaleString();
      document.getElementById('rfFN').textContent = cm.false_negatives.toLocaleString();
      document.getElementById('rfTP').textContent = cm.true_positives.toLocaleString();
    }
  }

  // Initialize
  initPcaGrid();
  loadSamples();
});
