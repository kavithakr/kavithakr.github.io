//----------line number generation code
const codeGrid = document.getElementById('code-grid');
const codeBlock = document.getElementById('code-block');

const rawCode = codeBlock.textContent;
const allLines = rawCode.split('\n');

// Filter out the leading/trailing blank lines caused by <pre> formatting
// but keep blank lines in the middle of the code
const firstNonEmpty = allLines.findIndex(l => l.trim() !== '');
const lastNonEmpty = [...allLines].reverse().findIndex(l => l.trim() !== '');
const lines = allLines.slice(firstNonEmpty, allLines.length - lastNonEmpty);

let gridHTML = '';

for (let i = 0; i < lines.length; i++) {
    const safeLine = lines[i]
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    gridHTML += `<span class="line-number" id="gutter-${i}">${i + 1}</span>`;
    gridHTML += `<span class="code-line" id="line-${i}">${safeLine}</span>`;
}

codeGrid.innerHTML = gridHTML;

//-----------------------------------------------------

let iter = 0;
let plot_iterations = [0, 1, 10000, 25000, 40000, 50000, 60000, 80000, 99999];

/**
 * Highlights a group of lines (current step in the traversal)
 * @param {Array} lineNums - Array of 1-based line numbers
 */
function highlightGroup(lineNums) {
    if (!lineNums || lineNums.length === 0) return;

    // 1. Remove all existing highlights first
    document.querySelectorAll('.code-line').forEach(el => el.classList.remove('highlighted-line'));
    document.querySelectorAll('.line-number').forEach(el => el.classList.remove('highlighted-gutter'));

    // 2. Loop through the group and apply highlights
    lineNums.forEach(lineNum => {
        const index = lineNum - 1; // Convert to 0-based
        const targetLine = document.getElementById(`line-${index}`);
        const targetGutter = document.getElementById(`gutter-${index}`);

        if (targetLine && targetGutter) {
            targetLine.classList.add('highlighted-line');
            targetGutter.classList.add('highlighted-gutter');
        }
    });

    // 3. Scroll to the FIRST line of the group to keep it in view
    const firstLine = document.getElementById(`line-${lineNums[0] - 1}`);
    if (firstLine) {
        firstLine.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
    }
}

/**
 * Renders the intermediate computation card for the current step's
 * weight/bias update, mirroring the logistic-regression version but
 * without the sigmoid/threshold terms.
 */
function renderComputationStep(weight, bias, dw, db, learningRate, newWeight, newBias) {
    const stepEl = document.getElementById('step');
    stepEl.innerHTML = `
        <div class="weight-line">weight ( weight = weight&ndash; learning_rate x dw )</div>
        <div class="weight-line">weight = ${weight} &ndash; ${learningRate} x ${dw}</div>
        <div class="weight-line indent">= ${newWeight}</div>

        <div class="bias-line">bias (bias = bias &ndash; learning_rate x db)</div>
        <div class="bias-line">bias = ${bias} &ndash; ${learningRate} x ${db}</div>
        <div class="bias-line indent">= ${newBias}</div>
    `;
}

/**
 * Updates the read-only weight/bias display fields on the
 * Real-Time Visualization panel.
 */
function updateDisplays(weight, bias) {
    document.getElementById('weightDisplay').value = weight;
    document.getElementById('biasDisplay').value = bias;
}

// ---------------------------------------------------------------
// Training + step-through logic
// ---------------------------------------------------------------

let trainingSteps = [];   // array of snapshot objects, one per checkpoint (+ initial state)
let currentStepIndex = 0; // index into trainingSteps
let weight_hist = [];     // full per-iteration weight history (mirrors Python's weight_hist)
let bias_hist = [];       // full per-iteration bias history (mirrors Python's bias_hist)
let checkpointSnapshots = []; // {iter, weight, bias} for each plot_iterations checkpoint reached, in order

/**
 * Runs full gradient descent in JS. Records a flat list of
 * micro-steps: one initial-state step, then — for every checkpoint
 * in plot_iterations — one step PER LINE of the loop body
 * (22, 23, 25, 26, 28, 29, 30, 31, 35, 36, 37, 38, 39), so that
 * clicking Next walks through the computation line by line, and
 * this repeats once per checkpoint iteration. Ends with a final
 * summary step.
 */
function runTraining() {
    if (!X_values.length || !y_values.length) {
        alert('Please load a dataset first (Preset, Random, or Custom).');
        return false;
    }

    const learningRate = parseFloat(document.getElementById('learningRateInput').value) || 0.01;
    const numIterations = parseInt(document.getElementById('iterationsInput').value) || 100000;

    const n = X_values.length;
    let weight = parseFloat(document.getElementById('weightInput').value) || 0;
    let bias = parseFloat(document.getElementById('biasInput').value) || 0;

    trainingSteps = [];
    weight_hist = [];   // reset for this run, mirrors Python's weight_hist
    bias_hist = [];      // reset for this run, mirrors Python's bias_hist
    checkpointSnapshots = []; // reset cumulative plot lines for this run
    let snapshotCounter = 0;  // how many checkpoints have been "plotted" so far

    // Initial state, before any updates (lines 7-11: init)
    trainingSteps.push({
        highlight: [7, 8, 9, 10, 11],
        type: 'info',
        weight: weight,
        bias: bias,
        snapshotsUpTo: snapshotCounter,
        label: 'Initialize weight, bias, and history lists<br>(weight = 0, bias = 0)'
    });

    for (let iter = 0; iter < numIterations; iter++) {
        const yPredicted = X_values.map(x => weight * x + bias);

        let dwSum = 0, dbSum = 0;
        for (let i = 0; i < n; i++) {
            dwSum += X_values[i] * (yPredicted[i] - y_values[i]);
            dbSum += (yPredicted[i] - y_values[i]);
        }
        const dw = (2 / n) * dwSum;
        const db = (2 / n) * dbSum;

        const oldWeight = weight;
        const oldBias = bias;
        const newWeight = weight - learningRate * dw;
        const newBias = bias - learningRate * db;

        // Safety check: stop early if the learning rate is too high for this
        // data's scale and gradient descent has started diverging, rather
        // than silently producing NaN/Infinity deeper into the run.
        if (!isFinite(newWeight) || !isFinite(newBias)) {
            alert(
                `Training diverged at iteration ${iter} (weight/bias became infinite or NaN).\n\n` +
                `This usually means the learning rate (${learningRate}) is too high for this dataset's scale. ` +
                `Try a smaller learning rate (e.g. 0.01, 0.001, or lower), or use a smaller dataset range.`
            );
            trainingSteps = [];
            return false;
        }

        if (plot_iterations.includes(iter)) {
            // One micro-step per line, in source order, for this checkpoint

            trainingSteps.push({
                highlight: [22],
                type: 'info',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
                label: `Loop: starting iteration ${iter} &nbsp; &mdash; &nbsp; weight and bias carried over from previous iteration: weight = ${oldWeight.toFixed(5)}, bias = ${oldBias.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [23],
                type: 'info',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
                label: `y_predicted = weight &times; X + bias &nbsp; (using weight = ${oldWeight.toFixed(5)}, bias = ${oldBias.toFixed(5)})`
            });

            trainingSteps.push({
                highlight: [25],
                type: 'info',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
                label: `dw = (2 / n) &times; &Sigma; X &times; (y_predicted &minus; y) = ${dw.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [26],
                type: 'info',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
                label: `db = (2 / n) &times; &Sigma; (y_predicted &minus; y) = ${db.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [28],
                type: 'weight_update',
                weight: oldWeight,
                bias: oldBias,
                dw: dw,
                newWeight: newWeight,
                learningRate: learningRate,
                snapshotsUpTo: snapshotCounter
            });

            trainingSteps.push({
                highlight: [29],
                type: 'bias_update',
                weight: newWeight,
                bias: oldBias,
                db: db,
                newBias: newBias,
                learningRate: learningRate,
                snapshotsUpTo: snapshotCounter
            });

            trainingSteps.push({
                highlight: [30],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `weight_hist.append(weight) &nbsp; &rarr; &nbsp; ${newWeight.toFixed(5)} added (history now has ${weight_hist.length + 1} entries; previous entry was ${weight_hist.length > 0 ? weight_hist[weight_hist.length - 1].toFixed(5) : 'none, this is the first'})`
            });

            trainingSteps.push({
                highlight: [31],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `bias_hist.append(bias) &nbsp; &rarr; &nbsp; ${newBias.toFixed(5)} added (history now has ${bias_hist.length + 1} entries; previous entry was ${bias_hist.length > 0 ? bias_hist[bias_hist.length - 1].toFixed(5) : 'none, this is the first'})`
            });

            trainingSteps.push({
                highlight: [35],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `if iter in plot_iterations: &nbsp; &rarr; &nbsp; True (iter = ${iter})`
            });

            trainingSteps.push({
                highlight: [36],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `current_weight = weight &nbsp; &rarr; &nbsp; ${newWeight.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [37],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `current_bias = bias &nbsp; &rarr; &nbsp; ${newBias.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [38],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `current_line = current_weight &times; x_plot + current_bias`
            });

            // This is the plotting line — a new snapshot line is added to the
            // graph here, so this step (and everything after it, until the
            // next checkpoint's line 39) shows one more cumulative line.
            checkpointSnapshots.push({ iter: iter, weight: newWeight, bias: newBias });
            snapshotCounter++;

            trainingSteps.push({
                highlight: [39],
                type: 'info',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `plt.plot(x_plot, current_line, label='Iteration ${iter}') &nbsp; &rarr; &nbsp; snapshot recorded`
            });
        }

        // Matches Python's weight_hist.append(weight) / bias_hist.append(bias),
        // which run every iteration, not just at checkpoints.
        weight_hist.push(newWeight);
        bias_hist.push(newBias);

        weight = newWeight;
        bias = newBias;
    }

    // Final state (lines 45, 51-57: return + testing)
    trainingSteps.push({
        highlight: [45, 51, 52, 56, 57],
        type: 'info',
        weight: weight,
        bias: bias,
        snapshotsUpTo: snapshotCounter,
        label: 'Training complete: final weight and bias returned'
    });

    currentStepIndex = 0;
    return true;
}

/**
 * Draws the scatter plot of X_values/y_values plus one regression line
 * per checkpoint snapshot reached so far (cumulative), onto the
 * #regressionCanvas element. Mirrors the Python plt.scatter() +
 * repeated plt.plot() calls building up the legend across checkpoints.
 */
const PLOT_COLORS = ['#e76f51', '#2a9d8f', '#e9c46a', '#264653', '#f4a261', '#8ab17d', '#6d597a', '#457b9d', '#9d4edd'];

function drawRegressionPlot(snapshots) {
    const canvas = document.getElementById('regressionCanvas');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (!X_values.length) return;

    const dataXMin = Math.min(...X_values);
    const dataXMax = Math.max(...X_values);
    const xRange = (dataXMax - dataXMin) || 1;
    const xMin = dataXMin - xRange * 0.1;
    const xMax = dataXMax + xRange * 0.1;

    let yMin = Math.min(...y_values);
    let yMax = Math.max(...y_values);
    snapshots.forEach(s => {
        const y1 = s.weight * xMin + s.bias;
        const y2 = s.weight * xMax + s.bias;
        yMin = Math.min(yMin, y1, y2);
        yMax = Math.max(yMax, y1, y2);
    });
    const yRange = (yMax - yMin) || 1;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;

    const marginL = 50, marginR = 12, marginT = 12, marginB = 36;
    const plotW = width - marginL - marginR;
    const plotH = height - marginT - marginB;

    const toPx = x => marginL + ((x - xMin) / (xMax - xMin)) * plotW;
    const toPy = y => marginT + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

    // Major gridlines + tick labels (matches plt.grid(True) in the Python source)
    const NUM_TICKS = 5; // number of major gridlines per axis

    ctx.font = '9px sans-serif';
    ctx.strokeStyle = '#dddddd';
    ctx.lineWidth = 1;

    // X-axis major ticks
    ctx.textAlign = 'center';
    for (let i = 0; i <= NUM_TICKS; i++) {
        const xVal = xMin + (xMax - xMin) * (i / NUM_TICKS);
        const px = toPx(xVal);

        ctx.beginPath();
        ctx.moveTo(px, marginT);
        ctx.lineTo(px, marginT + plotH);
        ctx.stroke();

        ctx.fillStyle = '#333';
        ctx.fillText(xVal.toFixed(2), px, marginT + plotH + 12);
    }

    // Y-axis major ticks
    ctx.textAlign = 'right';
    for (let i = 0; i <= NUM_TICKS; i++) {
        const yVal = yMin + (yMax - yMin) * (i / NUM_TICKS);
        const py = toPy(yVal);

        ctx.strokeStyle = '#dddddd';
        ctx.beginPath();
        ctx.moveTo(marginL, py);
        ctx.lineTo(marginL + plotW, py);
        ctx.stroke();

        ctx.fillStyle = '#333';
        ctx.fillText(yVal.toFixed(2), marginL - 4, py + 3);
    }

    // Axes (drawn on top of the gridlines for a crisp border)
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(marginL, marginT);
    ctx.lineTo(marginL, marginT + plotH);
    ctx.lineTo(marginL + plotW, marginT + plotH);
    ctx.stroke();

    // Axis titles (matches plt.xlabel('X') / plt.ylabel('Y') in the Python source)
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    ctx.fillText('X', marginL + plotW / 2, height - 4);

    ctx.save();
    ctx.translate(12, marginT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Y', 0, 0);
    ctx.restore();

    // Cumulative regression lines, one per checkpoint reached so far
    snapshots.forEach((s, idx) => {
        ctx.strokeStyle = PLOT_COLORS[idx % PLOT_COLORS.length];
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(toPx(xMin), toPy(s.weight * xMin + s.bias));
        ctx.lineTo(toPx(xMax), toPy(s.weight * xMax + s.bias));
        ctx.stroke();
    });

    // Scatter of the actual data points, drawn on top
    ctx.fillStyle = '#e63946';
    X_values.forEach((x, i) => {
        ctx.beginPath();
        ctx.arc(toPx(x), toPy(y_values[i]), 3, 0, Math.PI * 2);
        ctx.fill();
    });

    // Legend listing each plotted iteration and its color
    ctx.textAlign = 'left';
    ctx.font = '9px sans-serif';
    let legendY = marginT + 4;
    snapshots.forEach((s, idx) => {
        ctx.fillStyle = PLOT_COLORS[idx % PLOT_COLORS.length];
        ctx.fillRect(marginL + 4, legendY, 9, 9);
        ctx.fillStyle = '#000';
        ctx.fillText(`Iter ${s.iter}`, marginL + 16, legendY + 8);
        legendY += 12;
    });
}

/**
 * Renders the given micro-step: highlights its single code line,
 * fills the computation card according to the step's type, updates
 * the weight/bias display fields, and redraws the cumulative plot.
 */
function renderStep(step) {
    highlightGroup(step.highlight);

    if (step.type === 'weight_update') {
        document.getElementById('step').innerHTML = `
            <span class="calc-title">Weight Update</span>
            <span class="calc-body">weight = weight − lr × dw
weight = ${step.weight.toFixed(5)} − ${step.learningRate} × ${step.dw.toFixed(5)}
       = ${step.newWeight.toFixed(5)}</span>
        `;
        updateDisplays(step.newWeight.toFixed(5), step.bias.toFixed(5));
    } else if (step.type === 'bias_update') {
        document.getElementById('step').innerHTML = `
            <span class="calc-title">Bias Update</span>
            <span class="calc-body">bias = bias − lr × db
bias = ${step.bias.toFixed(5)} − ${step.learningRate} × ${step.db.toFixed(5)}
     = ${step.newBias.toFixed(5)}</span>
        `;
        updateDisplays(step.weight.toFixed(5), step.newBias.toFixed(5));
    } else {
        document.getElementById('step').innerHTML = `<span class="calc-body">${step.label}</span>`;
        updateDisplays(step.weight.toFixed(5), step.bias.toFixed(5));
    }

    drawRegressionPlot(checkpointSnapshots.slice(0, step.snapshotsUpTo));
}

document.getElementById('startBtn').addEventListener('click', () => {
    const ok = runTraining();
    if (ok) {
        renderStep(trainingSteps[currentStepIndex]);
    }
});

document.getElementById('nextBtn').addEventListener('click', () => {
    if (!trainingSteps.length) {
        alert('Click Start first to run training.');
        return;
    }
    if (currentStepIndex < trainingSteps.length - 1) {
        currentStepIndex++;
        renderStep(trainingSteps[currentStepIndex]);
    }
});

document.getElementById('prevBtn').addEventListener('click', () => {
    if (!trainingSteps.length) {
        alert('Click Start first to run training.');
        return;
    }
    if (currentStepIndex > 0) {
        currentStepIndex--;
        renderStep(trainingSteps[currentStepIndex]);
    }
});

document.getElementById('endBtn').addEventListener('click', () => {
    if (!trainingSteps.length) {
        alert('Click Start first to run training.');
        return;
    }
    currentStepIndex = trainingSteps.length - 1;
    renderStep(trainingSteps[currentStepIndex]);
});

// ---------------------------------------------------------------
// Dataset population: Preset / Random / Custom
// ---------------------------------------------------------------

let X_values = [];
let y_values = [];

const uploadContainer = document.getElementById('upload-container');
const dataPreviewContainer = document.getElementById('data-preview-container');
const dataTableBody = document.getElementById('dataTableBody');
const predefineBtn = document.getElementById('predefineBtn');
const randomBtn = document.getElementById('randomBtn');
const customBtn = document.getElementById('customBtn');

/**
 * Renders X_values/y_values into the data preview table
 * and reveals the preview panel.
 */
function renderDataPreview() {
    let rows = '';
    for (let i = 0; i < X_values.length; i++) {
        rows += `<tr><td>${X_values[i]}</td><td>${y_values[i]}</td></tr>`;
    }
    dataTableBody.innerHTML = rows;

    uploadContainer.classList.add('hidden');
    dataPreviewContainer.classList.remove('hidden');
}

/**
 * Highlights which of the three mode buttons is currently active,
 * matching the .btn-active style already defined in the page CSS.
 */
function setActiveModeButton(activeBtn) {
    [predefineBtn, randomBtn, customBtn].forEach(btn => btn.classList.remove('btn-active'));
    activeBtn.classList.add('btn-active');
}

// ---- Preset: fixed, reproducible dataset (same numbers every click) ----
predefineBtn.addEventListener('click', () => {
    X_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    y_values = [1.5, 3.7, 4.2, 6.1, 7.8, 9.0, 11.2, 12.5, 14.1, 15.9];

    renderDataPreview();
    setActiveModeButton(predefineBtn);
});

// ---- Random: freshly generated data each click, roughly linear + noise ----
randomBtn.addEventListener('click', () => {
    const n = 10;                                  // number of points
    const trueWeight = (Math.random() * 4 - 2);     // random slope, e.g. -2..2
    const trueBias = (Math.random() * 10 - 5);      // random intercept, e.g. -5..5
    const noiseScale = 1.5;

    X_values = [];
    y_values = [];

    for (let i = 1; i <= n; i++) {
        const x = i;
        const noise = (Math.random() * 2 - 1) * noiseScale;
        const y = trueWeight * x + trueBias + noise;

        X_values.push(x);
        y_values.push(parseFloat(y.toFixed(2)));
    }

    renderDataPreview();
    setActiveModeButton(randomBtn);
});

// ---- Custom: reveal the Excel upload UI instead of the data table ----
customBtn.addEventListener('click', () => {
    dataPreviewContainer.classList.add('hidden');
    uploadContainer.classList.remove('hidden');
    setActiveModeButton(customBtn);
});

// ---- Custom: handle the uploaded Excel file (expects columns X, Y) ----
document.getElementById('submitFileBtn').addEventListener('click', () => {
    const fileInput = document.getElementById('excelFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an Excel file first.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const rows = XLSX.utils.sheet_to_json(sheet, { header: 1 });

        // Assumes row 0 is a header (X, Y) and data starts at row 1
        X_values = [];
        y_values = [];
        for (let i = 1; i < rows.length; i++) {
            if (rows[i].length >= 2) {
                X_values.push(Number(rows[i][0]));
                y_values.push(Number(rows[i][1]));
            }
        }

        renderDataPreview();
    };
    reader.readAsArrayBuffer(file);
});