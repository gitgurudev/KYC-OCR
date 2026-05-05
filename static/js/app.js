'use strict';

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const browseBtn     = document.getElementById('browse-btn');
const dropIdle      = document.getElementById('drop-idle');
const dropPreview   = document.getElementById('drop-preview');
const previewImg    = document.getElementById('preview-img');
const removeBtn     = document.getElementById('remove-btn');
const scanBtn       = document.getElementById('scan-btn');

const processing    = document.getElementById('processing');
const steps         = [1, 2, 3, 4].map(n => document.getElementById(`step-${n}`));

const resultInfo    = document.getElementById('result-info');
const docName       = document.getElementById('doc-name');
const docMeta       = document.getElementById('doc-meta');
const docIcon       = document.getElementById('doc-icon');
const confidencePct = document.getElementById('confidence-pct');
const confidenceBar = document.getElementById('confidence-bar');
const detectionReason = document.getElementById('detection-reason');
const copyBtn       = document.getElementById('copy-btn');
const resetBtn      = document.getElementById('reset-btn');

const formEmpty     = document.getElementById('form-empty');
const formGrid      = document.getElementById('form-grid');
const jsonDebugWrap = document.getElementById('json-debug-wrap');
const rawJson       = document.getElementById('raw-json');

const errorBanner   = document.getElementById('error-banner');
const errorMsg      = document.getElementById('error-msg');
const errorClose    = document.getElementById('error-close');

// ─── State ────────────────────────────────────────────────────────────────────
let selectedFile = null;
let lastResult   = null;

// ─── Document type icons ──────────────────────────────────────────────────────
const DOC_ICONS = {
  aadhaar:         '🪪',
  pan:             '💳',
  passport:        '📘',
  voter_id:        '🗳️',
  driving_license: '🚗',
  ration_card:     '🏠',
  utility_bill:    '⚡',
  bank_passbook:   '🏦',
  unknown:         '📄',
};

// ─── File selection ───────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target === dropZone || dropIdle.contains(e.target)) fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

// Drag-and-drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
});

// Remove file
removeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

function setFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  dropIdle.classList.add('hidden');
  dropPreview.classList.remove('hidden');
  scanBtn.disabled = false;
  hideError();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  dropIdle.classList.remove('hidden');
  dropPreview.classList.add('hidden');
  scanBtn.disabled = true;
}

// ─── Scan ─────────────────────────────────────────────────────────────────────
scanBtn.addEventListener('click', runScan);

async function runScan() {
  if (!selectedFile) return;

  hideError();
  showProcessing();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    await animateStep(0, 600);
    await animateStep(1, 1000);

    const response = await fetch('/api/ocr', { method: 'POST', body: formData });

    await animateStep(2, 400);
    await animateStep(3, 200);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error. Please try again.');
    }

    lastResult = await response.json();
    renderResult(lastResult);

  } catch (err) {
    showError(err.message || 'Something went wrong. Is the server running?');
  } finally {
    hideProcessing();
  }
}

// ─── Processing animation ─────────────────────────────────────────────────────
function showProcessing() {
  steps.forEach(s => { s.classList.remove('active', 'done'); });
  processing.classList.remove('hidden');
}

function hideProcessing() {
  processing.classList.add('hidden');
}

function animateStep(index, delay) {
  return new Promise(resolve => {
    if (index > 0) {
      steps[index - 1].classList.remove('active');
      steps[index - 1].classList.add('done');
    }
    steps[index].classList.add('active');
    setTimeout(resolve, delay);
  });
}

// ─── Render result ────────────────────────────────────────────────────────────
function renderResult(data) {
  // Doc info card
  docIcon.textContent = DOC_ICONS[data.document_type] || '📄';
  docName.textContent = data.display_name || 'Unknown Document';
  docMeta.textContent = `RAG Match: ${(data.rag_similarity_score * 100).toFixed(0)}% · Category: ${data.schema?.category || '—'}`;

  const pct = Math.round((data.confidence || 0) * 100);
  confidencePct.textContent = `${pct}%`;
  confidenceBar.style.width = `${pct}%`;
  confidenceBar.style.background = pct >= 80 ? 'linear-gradient(90deg,#22c55e,#16a34a)'
                                  : pct >= 50 ? 'linear-gradient(90deg,#f59e0b,#d97706)'
                                              : 'linear-gradient(90deg,#ef4444,#b91c1c)';
  detectionReason.textContent = data.detection_reason || '—';
  resultInfo.classList.remove('hidden');

  // Build form fields
  formGrid.innerHTML = '';
  const fields    = data.schema?.fields || {};
  const extracted = data.extracted_fields || {};

  Object.entries(fields).forEach(([key, info]) => {
    const value   = extracted[key] ?? '';
    const isEmpty = !value;

    const group = document.createElement('div');
    group.className = 'field-group' + (info.type === 'textarea' ? ' full-width' : '');

    const label = document.createElement('label');
    label.className = 'field-label';
    label.textContent = info.label || key;
    label.htmlFor = `field-${key}`;
    group.appendChild(label);

    if (info.type === 'textarea') {
      const ta = document.createElement('textarea');
      ta.className = 'field-textarea' + (isEmpty ? ' empty' : '');
      ta.id = `field-${key}`;
      ta.rows = 3;
      ta.value = value || 'Not detected';
      group.appendChild(ta);
    } else {
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'field-input' + (isEmpty ? ' empty' : '');
      input.id = `field-${key}`;
      input.value = value || 'Not detected';
      group.appendChild(input);
    }

    if (info.hint) {
      const hint = document.createElement('p');
      hint.className = 'field-hint';
      hint.textContent = `Format: ${info.hint}`;
      group.appendChild(hint);
    }

    formGrid.appendChild(group);
  });

  // Show form, hide empty state
  formEmpty.classList.add('hidden');
  formGrid.classList.remove('hidden');

  // Show raw JSON
  rawJson.textContent = JSON.stringify(data, null, 2);
  jsonDebugWrap.classList.remove('hidden');

  // Scroll to form section
  document.getElementById('form-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Copy JSON ────────────────────────────────────────────────────────────────
copyBtn.addEventListener('click', async () => {
  if (!lastResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastResult.extracted_fields, null, 2));
    copyBtn.textContent = '✓ Copied!';
    setTimeout(() => {
      copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="1"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
      </svg> Copy JSON`;
    }, 2000);
  } catch {
    copyBtn.textContent = 'Copy failed';
  }
});

// ─── Reset ────────────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  clearFile();
  lastResult = null;

  // Hide result info
  resultInfo.classList.add('hidden');

  // Reset form section to empty state
  formGrid.innerHTML = '';
  formGrid.classList.add('hidden');
  formEmpty.classList.remove('hidden');
  jsonDebugWrap.classList.add('hidden');

  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ─── Error helpers ────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.remove('hidden');
}

function hideError() {
  errorBanner.classList.add('hidden');
}

errorClose.addEventListener('click', hideError);
