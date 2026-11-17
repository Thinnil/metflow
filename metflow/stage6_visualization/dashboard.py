"""Art Nouveau HTML dashboard generation and --only_dashboard fast path."""

import os
from typing import Optional
import pandas as pd

DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MET Flow - Art Nouveau Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-color: #f4f1ea; --text-color: #1b3a30; --gold: #d4af37; --dark-green: #0f251e; --panel-bg: #ffffff; }
        body { font-family: 'Lato', sans-serif; background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
        header { background-color: var(--dark-green); color: var(--gold); padding: 20px; text-align: center; font-family: 'Playfair Display', serif; border-bottom: 4px solid var(--gold); box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
        h1 { margin: 0; font-size: 2.5em; font-weight: 700; letter-spacing: 2px; }
        .container { display: flex; flex: 1; padding: 20px; gap: 20px; overflow: hidden; }
        .sidebar { width: 300px; background: var(--panel-bg); border: 2px solid var(--gold); border-radius: 15px 50px 15px 50px; padding: 20px; box-shadow: 5px 5px 15px rgba(0,0,0,0.1); display: flex; flex-direction: column; gap: 15px; overflow-y: auto; }
        .sidebar h2 { font-family: 'Playfair Display', serif; color: var(--dark-green); margin-top: 0; border-bottom: 1px solid var(--gold); padding-bottom: 10px;}
        .main-content { flex: 1; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; padding-right: 10px;}
        .plot-container { background: var(--panel-bg); border: 2px solid var(--gold); border-radius: 5px; padding: 10px; box-shadow: 3px 3px 10px rgba(0,0,0,0.1); position: relative; }
        select, input[type="file"], .export-btn { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid var(--gold); border-radius: 5px; background-color: var(--bg-color); font-family: 'Lato', sans-serif; }
        .export-btn { background: var(--gold); color: var(--dark-green); font-weight: bold; cursor: pointer; transition: 0.2s; border: none;}
        .export-btn:hover { filter: brightness(1.1); }
        .file-drop { border: 2px dashed var(--gold); padding: 15px; text-align: center; cursor: pointer; background: var(--bg-color); border-radius: 10px; transition: 0.3s; font-size: 0.9em; }
        .file-drop:hover { background: #e8e4d8; }
        #pymol-modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--dark-green); color: var(--gold); padding: 30px; border: 3px solid var(--gold); border-radius: 10px; z-index: 1000; box-shadow: 0 10px 30px rgba(0,0,0,0.5); width: 80%; max-width: 600px; }
        #pymol-modal h3 { font-family: 'Playfair Display', serif; margin-top:0; }
        #pymol-code { background: #000; color: #0f0; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; }
        .close-btn { background: var(--gold); color: var(--dark-green); border: none; padding: 10px 20px; cursor: pointer; font-weight: bold; margin-top: 15px; border-radius: 5px;}
        .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.55); z-index: 999; backdrop-filter: blur(2px); }
        
        /* Interactive Image Customization & Export Modal */
        #export-modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #ffffff; color: var(--text-color); padding: 25px 30px; border: 3px solid var(--gold); border-radius: 16px; z-index: 1001; box-shadow: 0 15px 45px rgba(0,0,0,0.45); width: 92%; max-width: 820px; max-height: 90vh; overflow-y: auto; font-family: 'Lato', sans-serif; }
        #export-modal h2 { font-family: 'Playfair Display', serif; color: var(--dark-green); margin: 0 0 15px 0; border-bottom: 2px solid var(--gold); padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .modal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px 20px; }
        .modal-section { background: var(--bg-color); border: 1px solid rgba(212, 175, 55, 0.45); border-radius: 10px; padding: 12px 16px; }
        .modal-section h4 { margin: 0 0 12px 0; color: var(--dark-green); font-family: 'Playfair Display', serif; font-size: 1.05em; border-bottom: 1px solid rgba(212, 175, 55, 0.35); padding-bottom: 4px; display: flex; align-items: center; gap: 6px; }
        .field-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 9px; font-size: 0.88em; }
        .field-row label { font-weight: 600; color: #1b3a30; }
        .field-row input[type="text"], .field-row input[type="number"], .field-row select { width: 145px; padding: 5px 8px; border: 1px solid var(--gold); border-radius: 4px; background: #fff; font-size: 0.9em; font-family: inherit; }
        .field-row input[type="color"] { width: 45px; height: 28px; border: 1px solid var(--gold); border-radius: 4px; cursor: pointer; padding: 0; background: none; }
        .slider-row { margin-bottom: 10px; font-size: 0.88em; }
        .slider-header { display: flex; justify-content: space-between; margin-bottom: 3px; font-weight: 600; color: #1b3a30; }
        .slider-row input[type="range"] { width: 100%; cursor: pointer; accent-color: var(--dark-green); }
        .modal-actions { display: flex; justify-content: flex-end; gap: 12px; margin-top: 18px; padding-top: 14px; border-top: 2px solid var(--gold); }
        .btn-apply { background: #1b3a30; color: #d4af37; border: 1px solid #d4af37; padding: 9px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 0.92em; }
        .btn-apply:hover { background: #234c3f; }
        .btn-download { background: var(--gold); color: var(--dark-green); border: none; padding: 9px 24px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 0.95em; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
        .btn-download:hover { filter: brightness(1.1); }
        .btn-cancel { background: #e2e8f0; color: #4a5568; border: none; padding: 9px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.92em; }
        .btn-cancel:hover { background: #cbd5e1; }
    </style>
</head>
<body>
    <header>
        <h1>MET Flow</h1>
        <div style="font-style: italic;">Thermodynamic & Conformational Analytics</div>
    </header>
    <div class="container">
        <div class="sidebar">
            <h2>Data Ingestion</h2>
            <div class="file-drop" id="drop-zone">
                Drop <b>global_cluster_metrics.csv</b>, <b>energy_checkpoint.csv</b>, or <b>pca_loadings.csv</b> here
                <input type="file" id="file-input" style="display: none;" accept=".csv" multiple>
            </div>
            <button class="export-btn" onclick="exportCSV()">💾 Export Merged CSV</button>
            <button class="export-btn" onclick="openExportModal()" style="margin-top: 10px; background: #1b3a30; color: #d4af37; border: 1px solid #d4af37;">🎨 Customize & Export Image</button>
            <h2>Plot Controls</h2>
            <label>Filter Dataset: <select id="sel-filter-dataset" onchange="updatePlots()">
                <option value="all" selected>All Datasets (Combined)</option>
            </select></label>
            <label>X-Axis: <select id="sel-x"></select></label>
            <label>Y-Axis: <select id="sel-y"></select></label>
            <label>Z-Axis (3D): <select id="sel-z"><option value="none">None (2D Plot)</option></select></label>
            <label>Color By: <select id="sel-color"></select></label>
            <div id="color-mode-container" style="margin-top:8px;">
                <label>Coloring Mode: <select id="sel-color-mode" onchange="updatePlots()">
                    <option value="binned" selected>Interval Ranges (Box-Plot with Outliers)</option>
                    <option value="continuous">Continuous Gradient</option>
                </select></label>
            </div>
            <div id="colorscale-container" style="display:none; margin-top:8px;">
                <label>Colorscale: <select id="sel-colorscale" onchange="updatePlots()">
                    <option value="Viridis">Viridis</option>
                    <option value="Plasma">Plasma</option>
                    <option value="Inferno">Inferno</option>
                    <option value="Jet">Jet</option>
                    <option value="RdBu">Red-Blue (RdBu)</option>
                </select></label>
            </div>
            <div style="margin-top: 15px;">
                <label style="display:flex; align-items:center; gap: 8px;">
                    <input type="checkbox" id="filter-energy" onchange="updatePlots()"> Only show frames with Energy data
                </label>
            </div>
        </div>
        <div class="main-content">
            <div class="plot-container" id="main-plot" style="height: 85vh; min-height: 700px; position: relative;">
                <div id="main-plot-placeholder" style="position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #888; pointer-events: none; z-index: 1;">
                    <div style="font-size: 1.3em; font-family: 'Playfair Display', serif; color: #1b3a30; margin-bottom: 8px;">Awaiting Trajectory Data</div>
                    <div>Drag & drop <b>global_cluster_metrics.csv</b> or wait for sample metrics to render.</div>
                </div>
            </div>
            <div class="plot-container" id="loadings-plot" style="display: none; height: 400px;"></div>
            <div style="display: flex; gap: 20px;">
                <div class="plot-container" id="timeseries-plot" style="flex: 1; height: 400px; overflow:hidden;">
                    <div style="text-align:center; padding: 100px; color: #888;">No Energy Data Found.<br><br>Drag & drop <b>energy_checkpoint.csv</b> to dynamically merge and visualize energy!</div>
                </div>
                <div class="plot-container" id="hist-plot" style="flex: 1; height: 400px;"></div>
            </div>
        </div>
    </div>
    <div class="overlay" id="overlay" onclick="closeModal()"></div>
    <div id="pymol-modal">
        <h3>Frame Selected!</h3>
        <p>Frame ID: <span id="modal-frame"></span> | Cluster/Color: <span id="modal-cluster"></span> | Energy: <span id="modal-energy"></span></p>
        <p>To extract this frame in PyMOL on your server, copy and paste this command:</p>
        <div id="pymol-code"></div>
        <button class="close-btn" onclick="closeModal()">Close</button>
    </div>

    <!-- Export & Styling Customization Modal -->
    <div id="export-modal">
        <h2>
            <span>🎨 Customize Image & Export</span>
            <span style="cursor: pointer; font-size: 0.9em;" onclick="closeModal()">✖</span>
        </h2>
        <div class="modal-grid">
            <!-- Section 1: Typography & Labels -->
            <div class="modal-section">
                <h4>🔤 Typography & Text</h4>
                <div class="field-row">
                    <label>Plot Title:</label>
                    <input type="text" id="exp-title" value="Structural Landscape" oninput="liveUpdateTitle()">
                </div>
                <div class="field-row">
                    <label>Font Family:</label>
                    <select id="exp-font-family">
                        <option value="Playfair Display, Georgia, serif" selected>Playfair Display (Serif)</option>
                        <option value="Lato, sans-serif">Lato (Clean Sans)</option>
                        <option value="Arial, Helvetica, sans-serif">Arial / Helvetica</option>
                        <option value="Times New Roman, Times, serif">Times New Roman (Academic)</option>
                        <option value="Courier New, monospace">Courier New (Monospace)</option>
                    </select>
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Title Size:</span><span id="lbl-title-size">24 px</span></div>
                    <input type="range" id="exp-title-size" min="14" max="44" value="24" oninput="document.getElementById('lbl-title-size').innerText = this.value + ' px'">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Axis Label Size:</span><span id="lbl-axis-size">14 px</span></div>
                    <input type="range" id="exp-axis-size" min="10" max="30" value="14" oninput="document.getElementById('lbl-axis-size').innerText = this.value + ' px'">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Tick Numbers Size:</span><span id="lbl-tick-size">11 px</span></div>
                    <input type="range" id="exp-tick-size" min="8" max="22" value="11" oninput="document.getElementById('lbl-tick-size').innerText = this.value + ' px'">
                </div>
                <div class="field-row">
                    <label>Text & Label Color:</label>
                    <input type="color" id="exp-font-color" value="#1b3a30">
                </div>
            </div>

            <!-- Section 2: Data Points / Dots -->
            <div class="modal-section">
                <h4>🔴 Data Points & Dots</h4>
                <div class="slider-row">
                    <div class="slider-header"><span>Marker / Dot Size:</span><span id="lbl-marker-size">5 px</span></div>
                    <input type="range" id="exp-marker-size" min="1" max="22" value="5" oninput="document.getElementById('lbl-marker-size').innerText = this.value + ' px'">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Marker Opacity:</span><span id="lbl-marker-opacity">0.85</span></div>
                    <input type="range" id="exp-marker-opacity" min="0.1" max="1.0" step="0.05" value="0.85" oninput="document.getElementById('lbl-marker-opacity').innerText = this.value">
                </div>
                <div class="slider-row">
                    <div class="slider-header"><span>Dot Border Width:</span><span id="lbl-border-width">0 px</span></div>
                    <input type="range" id="exp-border-width" min="0" max="3" step="0.5" value="0" oninput="document.getElementById('lbl-border-width').innerText = this.value + ' px'">
                </div>
                <div class="field-row">
                    <label>Dot Border Color:</label>
                    <input type="color" id="exp-border-color" value="#ffffff">
                </div>
                <div class="field-row">
                    <label>Legend Position:</label>
                    <select id="exp-legend-pos">
                        <option value="right" selected>Right (Vertical)</option>
                        <option value="top">Top (Horizontal)</option>
                        <option value="bottom">Bottom (Horizontal)</option>
                        <option value="none">Hidden</option>
                    </select>
                </div>
            </div>

            <!-- Section 3: Colors & Background -->
            <div class="modal-section">
                <h4>🎨 Colors & Background</h4>
                <div class="field-row">
                    <label>Canvas Background:</label>
                    <select id="exp-bg-mode" onchange="toggleCustomBgInput(this.value)">
                        <option value="white" selected>Pure White (#ffffff)</option>
                        <option value="transparent">Transparent</option>
                        <option value="cream">Theme Cream (#f4f1ea)</option>
                        <option value="dark">Dark Forest (#0f251e)</option>
                        <option value="custom">Custom Color</option>
                    </select>
                </div>
                <div class="field-row" id="row-custom-bg" style="display: none;">
                    <label>Custom BG Color:</label>
                    <input type="color" id="exp-custom-bg" value="#ffffff">
                </div>
                <div class="field-row">
                    <label>Show Gridlines:</label>
                    <input type="checkbox" id="exp-show-grid" checked>
                </div>
                <div class="field-row">
                    <label>Gridline Color:</label>
                    <input type="color" id="exp-grid-color" value="#e2e8f0">
                </div>
                <div class="field-row">
                    <label>Show Zero Line:</label>
                    <input type="checkbox" id="exp-show-zeroline" checked>
                </div>
            </div>

            <!-- Section 4: Resolution & File Format -->
            <div class="modal-section">
                <h4>📐 Dimensions & Resolution</h4>
                <div class="field-row">
                    <label>Dimension Preset:</label>
                    <select id="exp-preset" onchange="applyPresetDimensions(this.value)">
                        <option value="1200x800">Standard Web (1200 x 800)</option>
                        <option value="1920x1080" selected>Full HD 1080p (1920 x 1080)</option>
                        <option value="2560x1440">2K QHD (2560 x 1440)</option>
                        <option value="3840x2160">4K Ultra HD (3840 x 2160)</option>
                        <option value="1050x800">Journal Single-Col (1050 x 800)</option>
                        <option value="2100x1400">Journal Double-Col (2100 x 1400)</option>
                        <option value="custom">Custom Width / Height</option>
                    </select>
                </div>
                <div class="field-row">
                    <label>Width (pixels):</label>
                    <input type="number" id="exp-width" value="1920" min="400" max="7680">
                </div>
                <div class="field-row">
                    <label>Height (pixels):</label>
                    <input type="number" id="exp-height" value="1080" min="300" max="4320">
                </div>
                <div class="field-row">
                    <label>Resolution / DPI Scale:</label>
                    <select id="exp-scale">
                        <option value="1">1x (Standard ~96 DPI)</option>
                        <option value="2" selected>2x (HiDPI / Retina ~192 DPI)</option>
                        <option value="3">3x (Publication Print ~300 DPI)</option>
                        <option value="4">4x (Ultra Publication ~400 DPI)</option>
                    </select>
                </div>
                <div class="field-row">
                    <label>Export Format:</label>
                    <select id="exp-format">
                        <option value="png" selected>PNG (.png - Lossless)</option>
                        <option value="svg">SVG (.svg - Vector Graphics)</option>
                        <option value="jpeg">JPEG (.jpg - Web)</option>
                        <option value="webp">WebP (.webp)</option>
                    </select>
                </div>
            </div>
        </div>
        <div class="modal-actions">
            <button class="btn-cancel" onclick="closeModal()">Cancel</button>
            <button class="btn-apply" onclick="applySettingsToLiveView()">👁️ Apply to Dashboard View</button>
            <button class="btn-download" onclick="downloadCustomImage()">💾 Download High-Res Image</button>
        </div>
    </div>
    <script>
        const EMBEDDED_METRICS_JSON = /*EMBEDDED_DATA_PLACEHOLDER*/[];
        let globalData = [];
        let globalLoadings = [];
        let cols = [];
        let pendingFiles = [];
        let mergeTimeout = null;
        
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.background = '#e8e4d8'; });
        dropZone.addEventListener('dragleave', () => dropZone.style.background = '');
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault(); dropZone.style.background = '';
            Array.from(e.dataTransfer.files).forEach(f => processFile(f));
        });
        fileInput.addEventListener('change', (e) => { 
            Array.from(e.target.files).forEach(f => processFile(f)); 
        });

        window.addEventListener('DOMContentLoaded', () => {
            if (Array.isArray(EMBEDDED_METRICS_JSON) && EMBEDDED_METRICS_JSON.length > 0) {
                const ph = document.getElementById('main-plot-placeholder');
                if (ph) { ph.style.display = 'none'; ph.remove(); }
                globalData = EMBEDDED_METRICS_JSON;
                cols = Object.keys(globalData[0]);
                populateDropdowns();
                updatePlots();
                return;
            }
            fetch('global_cluster_metrics.csv')
                .then(r => { if (r.ok) return r.text(); throw new Error('Not found'); })
                .then(text => {
                    Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: function(res) {
                        if (res.meta && res.meta.fields) {
                            const ph = document.getElementById('main-plot-placeholder');
                            if (ph) { ph.style.display = 'none'; ph.remove(); }
                            globalData = res.data;
                            cols = res.meta.fields;
                            populateDropdowns();
                            updatePlots();
                        }
                    }});
                }).catch(() => {});
        });
        
        function processFile(file) {
            const ph = document.getElementById('main-plot-placeholder');
            if (ph) { ph.style.display = 'none'; ph.remove(); }
            Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true,
                complete: function(results) {
                    if (results.meta.fields.includes('Residue') && results.meta.fields.includes('PC1_Contribution')) {
                        globalLoadings = results.data;
                        alert("✅ Loaded PCA Loadings Data!");
                        updatePlots();
                        return;
                    }
                    
                    pendingFiles.push(results);
                    
                    clearTimeout(mergeTimeout);
                    mergeTimeout = setTimeout(() => {
                        if (pendingFiles.length > 0) mergePendingFiles();
                    }, 100);
                }
            });
        }
        
        function mergePendingFiles() {
            pendingFiles.sort((a, b) => b.data.length - a.data.length);
            
            pendingFiles.forEach(results => {
                let rCols = results.meta.fields;
                
                if(rCols.includes('Energy_kcal_mol') && !rCols.includes('OpenMM_Energy_kcal_mol')) {
                    results.data.forEach(d => d['OpenMM_Energy_kcal_mol'] = d['Energy_kcal_mol']);
                    rCols.push('OpenMM_Energy_kcal_mol');
                }
                if(rCols.includes('Energy') && !rCols.includes('OpenMM_Energy_kcal_mol')) {
                    results.data.forEach(d => d['OpenMM_Energy_kcal_mol'] = d['Energy']);
                    rCols.push('OpenMM_Energy_kcal_mol');
                }
                
                if (globalData.length === 0) {
                    globalData = results.data; 
                    cols = rCols;
                } else {
                    let newCols = rCols.filter(c => !cols.includes(c));
                    if (newCols.length > 0) {
                        let dataMap = new Map();
                        results.data.forEach(d => { 
                            let df = (d['Frame'] !== undefined) ? d['Frame'] : d['Frame_Index'];
                            if (df !== undefined) dataMap.set(df, d); 
                        });
                        
                        globalData.forEach(g => {
                            let f = (g['Frame'] !== undefined) ? g['Frame'] : g['Frame_Index'];
                            if (dataMap.has(f)) {
                                let nd = dataMap.get(f);
                                newCols.forEach(c => g[c] = nd[c]);
                            }
                        });
                        cols = [...cols, ...newCols];
                        alert("✅ Successfully merged new dataset into existing plots!");
                    }
                }
            });
            
            pendingFiles = [];
            populateDropdowns(); 
            updatePlots();
        }
        
        function exportCSV() {
            if(globalData.length === 0) return;
            const csv = Papa.unparse(globalData);
            const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "global_cluster_metrics_MERGED.csv");
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        function populateDropdowns() {
            const sels = ['sel-x', 'sel-y', 'sel-z', 'sel-color'];
            const prevVals = {};
            sels.forEach(id => {
                prevVals[id] = document.getElementById(id).value;
            });

            sels.forEach(id => {
                const el = document.getElementById(id);
                if(id !== 'sel-z') el.innerHTML = '';
                else el.innerHTML = '<option value="none">None (2D Plot)</option>';
                cols.forEach(c => { el.innerHTML += `<option value="${c}">${c}</option>`; });
            });

            const pc1_col = cols.find(c => c.startsWith('PCA_1'));
            const pc2_col = cols.find(c => c.startsWith('PCA_2'));

            const selX = document.getElementById('sel-x');
            if (prevVals['sel-x'] && cols.includes(prevVals['sel-x'])) selX.value = prevVals['sel-x'];
            else if (pc1_col) selX.value = pc1_col;

            const selY = document.getElementById('sel-y');
            if (prevVals['sel-y'] && cols.includes(prevVals['sel-y'])) selY.value = prevVals['sel-y'];
            else if (pc2_col) selY.value = pc2_col;

            const selZ = document.getElementById('sel-z');
            if (prevVals['sel-z'] && (prevVals['sel-z'] === 'none' || cols.includes(prevVals['sel-z']))) selZ.value = prevVals['sel-z'];
            else if (cols.includes('OpenMM_Energy_kcal_mol')) selZ.value = 'OpenMM_Energy_kcal_mol';
            else selZ.value = 'none';

            const dsSelect = document.getElementById('sel-filter-dataset');
            if (dsSelect && cols.includes('Dataset')) {
                const prevDs = dsSelect.value;
                const uniqueDs = Array.from(new Set(globalData.map(d => d['Dataset']).filter(Boolean)));
                dsSelect.innerHTML = '<option value="all">All Datasets (Combined)</option>';
                uniqueDs.forEach(ds => {
                    const opt = document.createElement('option');
                    opt.value = ds;
                    opt.innerText = ds;
                    dsSelect.appendChild(opt);
                });
                if (uniqueDs.includes(prevDs)) dsSelect.value = prevDs;
            }

            const selColor = document.getElementById('sel-color');
            if (prevVals['sel-color'] && cols.includes(prevVals['sel-color'])) {
                selColor.value = prevVals['sel-color'];
            } else if (cols.includes('Dataset')) {
                selColor.value = 'Dataset';
            } else if (cols.includes('Daura_Cluster')) {
                selColor.value = 'Daura_Cluster';
            } else if (cols.includes('PCA_KMeans_Cluster')) {
                selColor.value = 'PCA_KMeans_Cluster';
            }
        }
        
        ['sel-x', 'sel-y', 'sel-z', 'sel-color', 'sel-filter-dataset'].forEach(id => { 
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', updatePlots); 
        });
        
        let customSettings = {
            title: 'Structural Landscape',
            fontFamily: 'Playfair Display, Georgia, serif',
            fontColor: '#1b3a30',
            titleSize: 24,
            axisLabelSize: 14,
            tickSize: 11,
            markerSize: 5,
            markerOpacity: 0.85,
            borderWidth: 0,
            borderColor: '#ffffff',
            bgMode: 'white',
            customBg: '#ffffff',
            showGrid: true,
            showZeroLine: true,
            gridColor: '#e2e8f0',
            legendPos: 'right',
            width: 1920,
            height: 1080,
            scale: 2,
            format: 'png'
        };

        function updatePlots() {
            if(globalLoadings.length > 0) {
                document.getElementById('loadings-plot').style.display = 'block';
                Plotly.react('loadings-plot', [
                    { x: globalLoadings.map(d=>d.Residue), y: globalLoadings.map(d=>d.PC1_Contribution), name: 'PC1', type: 'scatter', mode: 'lines+markers', line: {color: '#1b3a30'} },
                    { x: globalLoadings.map(d=>d.Residue), y: globalLoadings.map(d=>d.PC2_Contribution), name: 'PC2', type: 'scatter', mode: 'lines+markers', line: {color: '#d4af37'} }
                ], { title: {text: 'PCA Loadings (Residue Contribution)', font: {family: customSettings.fontFamily, color: customSettings.fontColor}}, xaxis: {title: 'Residue Number'}, yaxis: {title: 'Loading Magnitude'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
            }
            
            if(!globalData.length) return;
            const ph = document.getElementById('main-plot-placeholder');
            if (ph) { ph.style.display = 'none'; ph.remove(); }
            
            let plotData = globalData;
            const selDataset = document.getElementById('sel-filter-dataset') ? document.getElementById('sel-filter-dataset').value : 'all';
            if (selDataset && selDataset !== 'all' && cols.includes('Dataset')) {
                plotData = plotData.filter(d => d['Dataset'] === selDataset);
            }
            if(document.getElementById('filter-energy') && document.getElementById('filter-energy').checked && cols.includes('OpenMM_Energy_kcal_mol')) {
                plotData = plotData.filter(d => d['OpenMM_Energy_kcal_mol'] !== undefined && d['OpenMM_Energy_kcal_mol'] !== null && d['OpenMM_Energy_kcal_mol'] !== '');
            }
            if(!plotData.length) plotData = globalData;
            
            const cx = document.getElementById('sel-x').value;
            const cy = document.getElementById('sel-y').value;
            const cz = document.getElementById('sel-z').value;
            const cColor = document.getElementById('sel-color').value;
            const cScale = document.getElementById('sel-colorscale') ? document.getElementById('sel-colorscale').value : 'Viridis';
            
            const x = plotData.map(d => d[cx]);
            const y = plotData.map(d => d[cy]);
            const color = plotData.map(d => d[cColor]);
            const frames = plotData.map(d => (d['Frame'] !== undefined) ? d['Frame'] : d['Frame_Index']);
            const hovertext = plotData.map(d => `Frame: ${(d['Frame'] !== undefined) ? d['Frame'] : d['Frame_Index']}<br>${cColor}: ${d[cColor]}${d['Dataset'] ? '<br>Dataset: ' + d['Dataset'] : ''}`);
            
            const colorMode = document.getElementById('sel-color-mode') ? document.getElementById('sel-color-mode').value : 'binned';
            const csContainer = document.getElementById('colorscale-container');
            const cmContainer = document.getElementById('color-mode-container');

            function parseVal(v) {
                if (v === null || v === undefined) return null;
                if (typeof v === 'number') return isNaN(v) ? null : v;
                let s = String(v).trim();
                if (s === '' || s.toLowerCase() === 'null' || s.toLowerCase() === 'nan' || s.toLowerCase() === 'n/a' || s.toLowerCase() === 'undefined') return null;
                let n = parseFloat(s);
                return isNaN(n) ? null : n;
            }

            let numericVals = color.map(v => parseVal(v));
            let validNums = numericVals.filter(v => v !== null);
            let validCount = validNums.length;
            
            const isEnergy = cColor.toLowerCase().includes('energy');
            const isExplicitCategory = ['dataset', 'cluster', 'label', 'method'].some(k => cColor.toLowerCase().includes(k));
            const uniqueValCount = new Set(color.map(v => (v === null || v === undefined) ? '' : String(v).trim())).size;
            const isCategory = isExplicitCategory || (!isEnergy && !['pca', 'rmsd', 'tsne', 'loading'].some(k => cColor.toLowerCase().includes(k)) && uniqueValCount <= 40);
            
            if (isCategory) {
                if (cmContainer) cmContainer.style.display = 'none';
                if (csContainer) csContainer.style.display = 'none';
            } else if (isEnergy) {
                if (cmContainer) cmContainer.style.display = 'block';
                if (csContainer) csContainer.style.display = (colorMode === 'continuous') ? 'block' : 'none';
            } else {
                if (cmContainer) cmContainer.style.display = 'none';
                if (csContainer) csContainer.style.display = 'block';
            }

            let traces = [];
            
            if (isCategory && color.length > 0) {
                const catPalette = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#3366cc', '#dc3912', '#ff9900', '#109618', '#990099',
                    '#0099c6', '#dd4477', '#66aa00', '#b82e2e', '#316395'
                ];
                let groupMap = new Map();
                for (let i = 0; i < color.length; i++) {
                    let val = (color[i] === null || color[i] === undefined || String(color[i]).trim() === '') ? 'Unknown' : String(color[i]);
                    if (!groupMap.has(val)) groupMap.set(val, []);
                    groupMap.get(val).push(i);
                }
                let colorIdx = 0;
                groupMap.forEach((indices, val) => {
                    const c = catPalette[colorIdx % catPalette.length];
                    colorIdx++;
                    let trace = {
                        name: `${cColor}: ${val} (n=${indices.length})`,
                        x: indices.map(i => x[i]),
                        y: indices.map(i => y[i]),
                        mode: 'markers',
                        marker: {
                            size: customSettings.markerSize,
                            color: c,
                            opacity: customSettings.markerOpacity,
                            line: { width: customSettings.borderWidth, color: customSettings.borderColor }
                        },
                        text: indices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: indices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = indices.map(i => plotData[i][cz]); } 
                    else { trace.type = 'scatter'; }
                    traces.push(trace);
                });
            } else if (isEnergy && colorMode === 'binned') {
                if (validNums.length === 0) {
                    let trace = {
                        name: `${cColor}: No Valid Data / NaN (n=${plotData.length})`,
                        x: x,
                        y: y,
                        mode: 'markers',
                        marker: { size: 4, color: '#6b7280', opacity: 0.5 },
                        text: hovertext,
                        hoverinfo: 'text+x+y+z',
                        customdata: plotData
                    };
                    if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = plotData.map(d => d[cz]); } 
                    else { trace.type = 'scatter'; }
                    traces.push(trace);
                } else {
                    const sorted = [...validNums].sort((a, b) => a - b);
                    const nValid = sorted.length;
                    
                    const q1 = sorted[Math.floor(nValid * 0.25)];
                    const med = sorted[Math.floor(nValid * 0.50)];
                    const q3 = sorted[Math.floor(nValid * 0.75)];
                    const iqr = q3 - q1;
                    
                    // For energy or skewed metrics, calculate robust cutoff
                    const p90 = sorted[Math.floor(nValid * 0.90)];
                    let highCutoff = (iqr > 0) ? (q3 + 1.5 * iqr) : p90;
                    // If extreme clashes pull highCutoff above 2500 kcal/mol, clamp it to p90 or 2000
                    if (cColor.toLowerCase().includes('energy')) {
                        if (p90 > q3 && p90 < highCutoff) highCutoff = p90;
                        if (highCutoff > 2000 && q3 < 2000) highCutoff = Math.max(q3, 1500);
                    }
                    let lowCutoff = (iqr > 0) ? (q1 - 1.5 * iqr) : sorted[0];
                    
                    const repVals = sorted.filter(v => v >= lowCutoff && v <= highCutoff);
                    const repMin = repVals.length > 0 ? repVals[0] : sorted[0];
                    const repMax = repVals.length > 0 ? repVals[repVals.length - 1] : highCutoff;
                    const nRep = repVals.length;
                    
                    const b1 = repVals[Math.floor(nRep * 0.20)] || repMin;
                    const b2 = repVals[Math.floor(nRep * 0.40)] || b1;
                    const b3 = repVals[Math.floor(nRep * 0.60)] || b2;
                    const b4 = repVals[Math.floor(nRep * 0.80)] || b3;
                    
                    const colors = ['#059669', '#0d9488', '#0284c7', '#d97706', '#dc2626'];
                    
                    let hasLowOutliers = (lowCutoff > sorted[0]);
                    let buckets = [];
                    if (hasLowOutliers) {
                        buckets.push({ name: `🟣 Low Outliers (< ${lowCutoff.toFixed(1)})`, color: '#4338ca', indices: [] });
                    }
                    buckets.push(
                        { name: `[${repMin.toFixed(1)}, ${b1.toFixed(1)})`, color: colors[0], indices: [] },
                        { name: `[${b1.toFixed(1)}, ${b2.toFixed(1)})`, color: colors[1], indices: [] },
                        { name: `[${b2.toFixed(1)}, ${b3.toFixed(1)})`, color: colors[2], indices: [] },
                        { name: `[${b3.toFixed(1)}, ${b4.toFixed(1)})`, color: colors[3], indices: [] },
                        { name: `[${b4.toFixed(1)}, ${repMax.toFixed(1)}]`, color: colors[4], indices: [] },
                        { name: `⚫ High Outliers / Clashes (> ${highCutoff.toFixed(1)})`, color: '#111111', indices: [] },
                        { name: `❌ Clashes / NaN`, color: '#4b5563', indices: [] }
                    );
                    
                    const offset = hasLowOutliers ? 1 : 0;
                    for (let i = 0; i < plotData.length; i++) {
                        const val = numericVals[i];
                        if (val === null || isNaN(val)) {
                            buckets[buckets.length - 1].indices.push(i);
                        } else if (val > highCutoff) {
                            buckets[buckets.length - 2].indices.push(i);
                        } else if (hasLowOutliers && val < lowCutoff) {
                            buckets[0].indices.push(i);
                        } else if (val < b1) {
                            buckets[offset].indices.push(i);
                        } else if (val < b2) {
                            buckets[offset + 1].indices.push(i);
                        } else if (val < b3) {
                            buckets[offset + 2].indices.push(i);
                        } else if (val < b4) {
                            buckets[offset + 3].indices.push(i);
                        } else {
                            buckets[offset + 4].indices.push(i);
                        }
                    }
                    
                    buckets.forEach(b => {
                        if (b.indices.length > 0) {
                            let isHighOutlier = b.name.includes('High Outliers');
                            let isNaNGroup = b.name.includes('NaN');
                            let trace = {
                                name: `${b.name} (n=${b.indices.length})`,
                                x: b.indices.map(i => x[i]),
                                y: b.indices.map(i => y[i]),
                                mode: 'markers',
                                marker: {
                                    size: isHighOutlier ? Math.max(3, customSettings.markerSize - 1) : (isNaNGroup ? Math.max(2.5, customSettings.markerSize - 1.5) : customSettings.markerSize),
                                    color: b.color,
                                    opacity: isNaNGroup ? 0.35 : customSettings.markerOpacity,
                                    line: { width: customSettings.borderWidth, color: customSettings.borderColor }
                                },
                                text: b.indices.map(i => hovertext[i]),
                                hoverinfo: 'text+x+y+z',
                                customdata: b.indices.map(i => plotData[i])
                            };
                            if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = b.indices.map(i => plotData[i][cz]); } 
                            else { trace.type = 'scatter'; }
                            traces.push(trace);
                        }
                    });
                }
            } else {
                let cmin = undefined;
                let cmax = undefined;
                const sorted = [...validNums].sort((a, b) => a - b);
                if (sorted.length > 20) {
                    cmin = sorted[Math.floor(sorted.length * 0.05)];
                    cmax = sorted[Math.floor(sorted.length * 0.95)];
                }

                let validIndices = [];
                let nanIndices = [];
                for (let i = 0; i < plotData.length; i++) {
                    if (numericVals[i] !== null && !isNaN(numericVals[i])) validIndices.push(i);
                    else nanIndices.push(i);
                }

                if (validIndices.length > 0) {
                    let trace = {
                        name: `${cColor} (Continuous)`,
                        x: validIndices.map(i => x[i]),
                        y: validIndices.map(i => y[i]),
                        mode: 'markers',
                        marker: {
                            size: customSettings.markerSize,
                            color: validIndices.map(i => numericVals[i]),
                            colorscale: cScale,
                            cmin: cmin,
                            cmax: cmax,
                            opacity: customSettings.markerOpacity,
                            line: { width: customSettings.borderWidth, color: customSettings.borderColor },
                            showscale: true,
                            colorbar: {
                                title: { text: cColor, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                                tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor }
                            }
                        },
                        text: validIndices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: validIndices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { trace.type = 'scatter3d'; trace.z = validIndices.map(i => plotData[i][cz]); } 
                    else { trace.type = 'scatter'; }
                    traces.push(trace);
                }

                if (nanIndices.length > 0) {
                    let nanTrace = {
                        name: `${cColor}: No Data / NaN (n=${nanIndices.length})`,
                        x: nanIndices.map(i => x[i]),
                        y: nanIndices.map(i => y[i]),
                        mode: 'markers',
                        marker: {
                            size: Math.max(2.5, customSettings.markerSize - 1.5),
                            color: '#9ca3af',
                            opacity: 0.4,
                            line: { width: customSettings.borderWidth, color: customSettings.borderColor }
                        },
                        text: nanIndices.map(i => hovertext[i]),
                        hoverinfo: 'text+x+y+z',
                        customdata: nanIndices.map(i => plotData[i])
                    };
                    if (cz !== 'none') { nanTrace.type = 'scatter3d'; nanTrace.z = nanIndices.map(i => plotData[i][cz]); } 
                    else { nanTrace.type = 'scatter'; }
                    traces.push(nanTrace);
                }
            }
            
            let liveBg = 'rgba(0,0,0,0)';
            if (customSettings.bgMode === 'white') liveBg = '#ffffff';
            else if (customSettings.bgMode === 'cream') liveBg = '#f4f1ea';
            else if (customSettings.bgMode === 'dark') liveBg = '#0f251e';
            else if (customSettings.bgMode === 'custom') liveBg = customSettings.customBg;

            let legendConfig = {
                title: { text: cColor, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                orientation: 'v',
                x: 1.02,
                y: 1,
                font: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor }
            };
            if (customSettings.legendPos === 'none') {
                legendConfig = false;
            } else if (customSettings.legendPos === 'top') {
                legendConfig = { title: { text: cColor }, orientation: 'h', y: 1.12, x: 0, font: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor } };
            } else if (customSettings.legendPos === 'bottom') {
                legendConfig = { title: { text: cColor }, orientation: 'h', y: -0.22, x: 0, font: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor } };
            }

            let layout = {
                title: { text: customSettings.title, font: {family: customSettings.fontFamily, size: customSettings.titleSize, color: customSettings.fontColor} },
                paper_bgcolor: liveBg,
                plot_bgcolor: liveBg,
                margin: {l: 60, r: 60, b: 65, t: 80},
                font: {family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor},
                showlegend: customSettings.legendPos !== 'none',
                legend: (customSettings.legendPos !== 'none') ? legendConfig : undefined,
                xaxis: {
                    title: { text: cx, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                },
                yaxis: {
                    title: { text: cy, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                }
            };

            const plotConfig = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: [{
                    name: '🎨 Customize & Export High-Res Image',
                    icon: Plotly.Icons.camera,
                    click: function() {
                        openExportModal();
                    }
                }],
                modeBarButtonsToRemove: ['toImage']
            };
            
            Plotly.react('main-plot', traces, layout, plotConfig);
            
            const mainPlotEl = document.getElementById('main-plot');
            if (!mainPlotEl._hasClickBound) {
                mainPlotEl._hasClickBound = true;
                mainPlotEl.on('plotly_click', function(data){
                    const pt = data.points[0]; const custom = pt.customdata;
                    const frame = (custom['Frame'] !== undefined) ? custom['Frame'] : custom['Frame_Index'];
                    const cluster = custom[cColor];
                    const ds_tag = custom['Dataset'] ? ` (${custom['Dataset']})` : '';
                    const energy = (custom['OpenMM_Energy_kcal_mol'] !== undefined && custom['OpenMM_Energy_kcal_mol'] !== null && !isNaN(custom['OpenMM_Energy_kcal_mol'])) ? custom['OpenMM_Energy_kcal_mol'] : 'N/A (Steric Clash / Uncalculated)';
                    
                    document.getElementById('modal-frame').innerText = frame;
                    document.getElementById('modal-cluster').innerText = (cluster !== undefined && cluster !== null) ? (cluster + ds_tag) : ('N/A' + ds_tag);
                    document.getElementById('modal-energy').innerText = energy;
                    
                    const code = `# Load unified trajectory in PyMOL:\nload merged_reference.pdb\nload_traj merged_trajectory.xtc, merged_reference\ncmd.frame(${frame + 1})\nsave frame_${frame}_extracted.pdb, merged_reference, state=${frame + 1}\nprint("Viewing Frame ${frame} (State ${frame + 1})")`;
                    document.getElementById('pymol-code').innerText = code;
                    document.getElementById('overlay').style.display = 'block';
                    document.getElementById('pymol-modal').style.display = 'block';
                });
            }
            
            if(cols.includes('OpenMM_Energy_kcal_mol')) {
                const e_vals = plotData.map(d => Number(d['OpenMM_Energy_kcal_mol'])).filter(v => v !== null && v !== undefined && !isNaN(v));
                if (e_vals.length > 0) {
                    document.getElementById('timeseries-plot').innerHTML = '';
                    document.getElementById('hist-plot').innerHTML = '';
                    const sorted_e = [...e_vals].sort((a,b) => a - b);
                    const q1 = sorted_e[Math.floor(sorted_e.length * 0.25)];
                    const q3 = sorted_e[Math.floor(sorted_e.length * 0.75)];
                    const iqr = q3 - q1;
                    const e_filt = (iqr > 0) ? e_vals.filter(v => v >= q1 - 2*iqr && v <= q3 + 2*iqr) : e_vals;
                    Plotly.react('timeseries-plot', [{ x: frames, y: plotData.map(d => d['OpenMM_Energy_kcal_mol']), type: 'scatter', mode: 'lines', line: {color: '#d4af37'} }], { title: {text: 'Energy Progression', font: {family: customSettings.fontFamily, color: customSettings.fontColor}}, xaxis: {title: 'Frame Index'}, yaxis: {title: 'Energy (kcal/mol)'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
                    Plotly.react('hist-plot', [{ x: e_filt, type: 'histogram', marker: {color: '#0f251e'} }], { title: {text: 'Energy Distribution (Representative Subset)', font: {family: customSettings.fontFamily, color: customSettings.fontColor}}, xaxis: {title: 'Energy (kcal/mol)'}, yaxis: {title: 'Frequency'}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }, {responsive: true});
                }
            }
        }

        function openExportModal() {
            document.getElementById('exp-title').value = customSettings.title;
            document.getElementById('exp-font-family').value = customSettings.fontFamily;
            document.getElementById('exp-title-size').value = customSettings.titleSize;
            document.getElementById('lbl-title-size').innerText = customSettings.titleSize + ' px';
            document.getElementById('exp-axis-size').value = customSettings.axisLabelSize;
            document.getElementById('lbl-axis-size').innerText = customSettings.axisLabelSize + ' px';
            document.getElementById('exp-tick-size').value = customSettings.tickSize;
            document.getElementById('lbl-tick-size').innerText = customSettings.tickSize + ' px';
            document.getElementById('exp-font-color').value = customSettings.fontColor;
            
            document.getElementById('exp-marker-size').value = customSettings.markerSize;
            document.getElementById('lbl-marker-size').innerText = customSettings.markerSize + ' px';
            document.getElementById('exp-marker-opacity').value = customSettings.markerOpacity;
            document.getElementById('lbl-marker-opacity').innerText = customSettings.markerOpacity;
            document.getElementById('exp-border-width').value = customSettings.borderWidth;
            document.getElementById('lbl-border-width').innerText = customSettings.borderWidth + ' px';
            document.getElementById('exp-border-color').value = customSettings.borderColor;
            document.getElementById('exp-legend-pos').value = customSettings.legendPos;
            
            document.getElementById('exp-bg-mode').value = customSettings.bgMode;
            toggleCustomBgInput(customSettings.bgMode);
            document.getElementById('exp-custom-bg').value = customSettings.customBg;
            document.getElementById('exp-show-grid').checked = customSettings.showGrid;
            document.getElementById('exp-grid-color').value = customSettings.gridColor;
            document.getElementById('exp-show-zeroline').checked = customSettings.showZeroLine;
            
            document.getElementById('exp-width').value = customSettings.width;
            document.getElementById('exp-height').value = customSettings.height;
            document.getElementById('exp-scale').value = customSettings.scale;
            document.getElementById('exp-format').value = customSettings.format;
            
            document.getElementById('overlay').style.display = 'block';
            document.getElementById('export-modal').style.display = 'block';
        }

        function toggleCustomBgInput(mode) {
            const row = document.getElementById('row-custom-bg');
            if (row) row.style.display = (mode === 'custom') ? 'flex' : 'none';
        }

        function applyPresetDimensions(preset) {
            if (preset === '1200x800') { document.getElementById('exp-width').value = 1200; document.getElementById('exp-height').value = 800; }
            else if (preset === '1920x1080') { document.getElementById('exp-width').value = 1920; document.getElementById('exp-height').value = 1080; }
            else if (preset === '2560x1440') { document.getElementById('exp-width').value = 2560; document.getElementById('exp-height').value = 1440; }
            else if (preset === '3840x2160') { document.getElementById('exp-width').value = 3840; document.getElementById('exp-height').value = 2160; }
            else if (preset === '1050x800') { document.getElementById('exp-width').value = 1050; document.getElementById('exp-height').value = 800; }
            else if (preset === '2100x1400') { document.getElementById('exp-width').value = 2100; document.getElementById('exp-height').value = 1400; }
        }

        function liveUpdateTitle() {
            const val = document.getElementById('exp-title').value;
            customSettings.title = val || 'Structural Landscape';
        }

        function updateSettingsFromModal() {
            customSettings.title = document.getElementById('exp-title').value || 'Structural Landscape';
            customSettings.fontFamily = document.getElementById('exp-font-family').value;
            customSettings.titleSize = parseInt(document.getElementById('exp-title-size').value) || 24;
            customSettings.axisLabelSize = parseInt(document.getElementById('exp-axis-size').value) || 14;
            customSettings.tickSize = parseInt(document.getElementById('exp-tick-size').value) || 11;
            customSettings.fontColor = document.getElementById('exp-font-color').value;
            
            customSettings.markerSize = parseFloat(document.getElementById('exp-marker-size').value) || 5;
            customSettings.markerOpacity = parseFloat(document.getElementById('exp-marker-opacity').value) || 0.85;
            customSettings.borderWidth = parseFloat(document.getElementById('exp-border-width').value) || 0;
            customSettings.borderColor = document.getElementById('exp-border-color').value;
            customSettings.legendPos = document.getElementById('exp-legend-pos').value;
            
            customSettings.bgMode = document.getElementById('exp-bg-mode').value;
            customSettings.customBg = document.getElementById('exp-custom-bg').value;
            customSettings.showGrid = document.getElementById('exp-show-grid').checked;
            customSettings.gridColor = document.getElementById('exp-grid-color').value;
            customSettings.showZeroLine = document.getElementById('exp-show-zeroline').checked;
            
            customSettings.width = parseInt(document.getElementById('exp-width').value) || 1920;
            customSettings.height = parseInt(document.getElementById('exp-height').value) || 1080;
            customSettings.scale = parseFloat(document.getElementById('exp-scale').value) || 2;
            customSettings.format = document.getElementById('exp-format').value || 'png';
        }

        function applySettingsToLiveView() {
            updateSettingsFromModal();
            updatePlots();
            closeModal();
        }

        function downloadCustomImage() {
            updateSettingsFromModal();
            
            let exportBg = '#ffffff';
            if (customSettings.bgMode === 'transparent') exportBg = 'rgba(0,0,0,0)';
            else if (customSettings.bgMode === 'cream') exportBg = '#f4f1ea';
            else if (customSettings.bgMode === 'dark') exportBg = '#0f251e';
            else if (customSettings.bgMode === 'custom') exportBg = customSettings.customBg;

            const mainEl = document.getElementById('main-plot');
            const cx = document.getElementById('sel-x') ? document.getElementById('sel-x').value : 'X';
            const cy = document.getElementById('sel-y') ? document.getElementById('sel-y').value : 'Y';
            const cleanTitle = (customSettings.title || 'plot').replace(/[^a-zA-Z0-9_-]/g, '_');
            const filename = `metflow_${cleanTitle}_${cx}_vs_${cy}`;

            Plotly.relayout(mainEl, {
                title: { text: customSettings.title, font: {family: customSettings.fontFamily, size: customSettings.titleSize, color: customSettings.fontColor} },
                paper_bgcolor: exportBg,
                plot_bgcolor: exportBg,
                font: {family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor},
                xaxis: {
                    title: { text: cx, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                },
                yaxis: {
                    title: { text: cy, font: { family: customSettings.fontFamily, size: customSettings.axisLabelSize, color: customSettings.fontColor } },
                    tickfont: { family: customSettings.fontFamily, size: customSettings.tickSize, color: customSettings.fontColor },
                    showgrid: customSettings.showGrid,
                    gridcolor: customSettings.gridColor,
                    zeroline: customSettings.showZeroLine
                }
            }).then(() => {
                return Plotly.downloadImage(mainEl, {
                    format: customSettings.format,
                    width: customSettings.width,
                    height: customSettings.height,
                    scale: customSettings.scale,
                    filename: filename
                });
            }).then(() => {
                closeModal();
            }).catch(err => {
                console.error("Download failed:", err);
                closeModal();
            });
        }

        function closeModal() {
            document.getElementById('overlay').style.display = 'none';
            document.getElementById('pymol-modal').style.display = 'none';
            const expM = document.getElementById('export-modal');
            if (expM) expM.style.display = 'none';
        }
    </script>
</body>
</html>"""



def generate_dashboard(out_dir: str, df: Optional[pd.DataFrame] = None) -> str:
    """Generate metflow_dashboard.html with embedded JSON metrics."""
    os.makedirs(out_dir, exist_ok=True)
    html = DASHBOARD_HTML_TEMPLATE
    if df is not None:
        df_json = df.to_json(orient="records")
        html = html.replace("/*EMBEDDED_DATA_PLACEHOLDER*/[]", df_json)
    else:
        csv_file = os.path.join(out_dir, "global_cluster_metrics.csv")
        if os.path.exists(csv_file):
            try:
                df_dash = pd.read_csv(csv_file)
                df_json = df_dash.to_json(orient="records")
                html = html.replace("/*EMBEDDED_DATA_PLACEHOLDER*/[]", df_json)
            except Exception:
                pass

    dash_file = os.path.join(out_dir, "metflow_dashboard.html")
    with open(dash_file, "w", encoding="utf-8") as f:
        f.write(html)
    return dash_file
