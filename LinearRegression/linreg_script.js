/* ── Python source (rendered as code-table like logistic regression) ── */
const CODE = `import numpy as np

# ── Run ───────────────────────────────────────────────
weight_hist, bias_hist = train_linear_regression(
    X_values, y_values,
    learning_rate=0.01,
    num_iterations=100000
)

# ── Loss function ─────────────────────────────────────
def compute_loss(y_true, y_pred):
    n = len(y_true)
    return (1 / n) * np.sum((y_pred - y_true) ** 2)

# ── Training function ──────────────────────────────────
def train_linear_regression(X, y, learning_rate, num_iterations):
    n_samples = X.shape[0]
    weight = 0
    bias   = 0
    weight_hist = []
    bias_hist   = []

    for iter in range(num_iterations):

        # Step 1: predict
        y_predicted = weight * X + bias

        # Step 2: compute gradients
        dw = (2 / n_samples) * np.sum(X * (y_predicted - y))
        db = (2 / n_samples) * np.sum(y_predicted - y)

        # Step 3: update weight
        weight -= learning_rate * dw

        # Step 4: update bias
        bias -= learning_rate * db

        weight_hist.append(weight)
        bias_hist.append(bias)

        # Step 5: draw regression line
        draw_line(weight, bias, iter)

    return weight_hist, bias_hist

# ── Evaluate ──────────────────────────────────────────
final_weight = weight_hist[-1]
final_bias   = bias_hist[-1]
predictions  = final_weight * X_values + final_bias
print(f"Predicted Values: {predictions}")`;

/* render as <table id="code-table"> rows with .ln and .lc cells */
(function renderCode() {
    const lines = CODE.split('\n');
    let html = '';
    lines.forEach((ln, i) => {
        const safe = ln.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') || ' ';
        html += `<tr id="row-${i}"><td class="ln">${i+1}</td><td class="lc">${safe}</td></tr>`;
    });
    document.getElementById('code-table').innerHTML = html;
})();

//-----------------------------------------------------

let iter = 0;
let plot_iterations = [0, 1, 10000, 25000, 40000, 50000, 60000, 80000, 99999];

/**
 * Highlights a group of lines (current step in the traversal)
 * @param {Array} lineNums - Array of 1-based line numbers
 */
function highlightGroup(lineNums) {
    if (!lineNums || lineNums.length === 0) return;
    /* clear all rows */
    document.querySelectorAll('#code-table tr').forEach(tr => {
        tr.classList.remove('highlighted-row', 'dimmed-row');
    });
    const hlSet = new Set(lineNums.map(n => `row-${n-1}`));
    document.querySelectorAll('#code-table tr').forEach(tr => {
        tr.classList.add(hlSet.has(tr.id) ? 'highlighted-row' : 'dimmed-row');
    });
    /* scroll into view */
    const first = document.getElementById(`row-${lineNums[0]-1}`);
    if (first) first.scrollIntoView({ block: 'center', behavior: 'smooth' });
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
        highlight: [3,4,5,6,7,8,16,17,18,19,20,21],
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
                highlight: [23],
                type: 'iteration',
                iter: iter,
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter
            });

            trainingSteps.push({
                highlight: [25,26],
                type: 'predict',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
               // label: `y_predicted = weight &times; X + bias &nbsp; (using weight = ${oldWeight.toFixed(5)}, bias = ${oldBias.toFixed(5)})`
            });

            trainingSteps.push({
                highlight: [29],
                type: 'dw',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
                label: `dw = (2 / n) &times; &Sigma; X &times; (y_predicted &minus; y) = ${dw.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [30],
                type: 'db',
                weight: oldWeight,
                bias: oldBias,
                snapshotsUpTo: snapshotCounter,
                label: `db = (2 / n) &times; &Sigma; (y_predicted &minus; y) = ${db.toFixed(5)}`
            });

            trainingSteps.push({
                highlight: [32,33],
                type: 'weight_update',
                weight: oldWeight,
                bias: oldBias,
                dw: dw,
                newWeight: newWeight,
                learningRate: learningRate,
                snapshotsUpTo: snapshotCounter
            });

            trainingSteps.push({
                highlight: [35,36],
                type: 'bias_update',
                weight: newWeight,
                bias: oldBias,
                db: db,
                newBias: newBias,
                learningRate: learningRate,
                snapshotsUpTo: snapshotCounter
            });

            trainingSteps.push({
                highlight: [38],
                type: 'weighthist',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `weight_hist.append(weight) &nbsp; &rarr; &nbsp; ${newWeight.toFixed(5)} added (history now has ${weight_hist.length + 1} entries; previous entry was ${weight_hist.length > 0 ? weight_hist[weight_hist.length - 1].toFixed(5) : 'none, this is the first'})`
            });

            trainingSteps.push({
                highlight: [39],
                type: 'biashist',
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter,
                label: `bias_hist.append(bias) &nbsp; &rarr; &nbsp; ${newBias.toFixed(5)} added (history now has ${bias_hist.length + 1} entries; previous entry was ${bias_hist.length > 0 ? bias_hist[bias_hist.length - 1].toFixed(5) : 'none, this is the first'})`
            });

            /* push the snapshot BEFORE the draw step so the canvas
               shows the new regression line when draw step is displayed */
            checkpointSnapshots.push({ iter, weight: newWeight, bias: newBias });
            snapshotCounter++;

            trainingSteps.push({
                highlight: [41, 42],
                type: 'draw',
                iter: iter,
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter   /* now includes this line */
            });

            trainingSteps.push({
                highlight: [23],
                type: 'iter',
                iter: iter,
                weight: newWeight,
                bias: newBias,
                snapshotsUpTo: snapshotCounter
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
    const fmt = (v, d=5) => Number(v).toFixed(d);
    const n   = X_values.length;
    const calc = document.getElementById('step');

    if (step.type === 'info') {
        /* ── Initialisation ─────────────────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">Initialise — Train Linear Regression</span>` +
            `<span class="calc-body">` +
            `n_samples = ${n}
` +
            `weight    = 0
` +
            `bias      = 0
` +
            `weight_hist = []
` +
            `bias_hist   = []

` +
            `Training will run for ${document.getElementById('iterationsInput').value || 100000} iterations
` +
            `Learning rate α = ${document.getElementById('learningRateInput').value || 0.01}` +
            `</span>`;
        updateDisplays('0.00000', '0.00000');

    } else if (step.type === 'iteration') {
        /* ── Loop header ────────────────────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">for iter in range(num_iterations)</span>` +
            `<span class="calc-body">` +
            `Current iteration : ${step.iter ?? '—'}

` +
            `Carrying over from previous iteration:
` +
            `  weight = ${fmt(step.weight)}
` +
            `  bias   = ${fmt(step.bias)}` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'predict') {
        /* ── y_predicted = weight × X + bias ────────────────────── */
        const yPreds = X_values.map(x => step.weight * x + step.bias);
        const preview = X_values.slice(0, 3).map((x, i) =>
            `  x=${fmt(x,2)}: ${fmt(step.weight)} × ${fmt(x,2)} + (${fmt(step.bias)}) = ${fmt(yPreds[i])}`
        ).join('\n');
        calc.innerHTML =
            `<span class="calc-title">Step 1 — Predict: y_predicted = weight × X + bias</span>` +
            `<span class="calc-body">` +
            `weight = ${fmt(step.weight)},  bias = ${fmt(step.bias)}

` +
            `First ${Math.min(3, n)} predictions:
${preview}` +
            (n > 3 ? `
  … (${n - 3} more)` : '') +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'dw') {
        /* ── dw = (2/n) Σ X×(ŷ−y) ───────────────────────────────── */
        const yPreds = X_values.map(x => step.weight * x + step.bias);
        const terms  = X_values.slice(0, 3).map((x, i) =>
            `  x=${fmt(x,2)}: ${fmt(x,2)} × (${fmt(yPreds[i])} − ${fmt(y_values[i],2)}) = ${fmt(x*(yPreds[i]-y_values[i]))}`
        ).join('\n');
        const dwSum  = X_values.reduce((s,x,i) => s + x*(yPreds[i]-y_values[i]), 0);
        calc.innerHTML =
            `<span class="calc-title">Step 2a — Gradient dw</span>` +
            `<span class="calc-body">` +
            `dw = (2 / n) × Σ X × (y_predicted − y)

` +
            `First ${Math.min(3,n)} terms:
${terms}` +
            (n > 3 ? `
  … (${n-3} more)` : '') +
            `

Σ = ${fmt(dwSum)}
` +
            `dw = (2 / ${n}) × ${fmt(dwSum)}
` +
            `   = ${fmt(step.dw)}` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'db') {
        /* ── db = (2/n) Σ (ŷ−y) ──────────────────────────────────── */
        const yPreds = X_values.map(x => step.weight * x + step.bias);
        const terms  = X_values.slice(0, 3).map((x, i) =>
            `  [${i}]: ${fmt(yPreds[i])} − ${fmt(y_values[i],2)} = ${fmt(yPreds[i]-y_values[i])}`
        ).join('\n');
        const dbSum  = X_values.reduce((s,x,i) => s + (yPreds[i]-y_values[i]), 0);
        calc.innerHTML =
            `<span class="calc-title">Step 2b — Gradient db</span>` +
            `<span class="calc-body">` +
            `db = (2 / n) × Σ (y_predicted − y)

` +
            `First ${Math.min(3,n)} terms:
${terms}` +
            (n > 3 ? `
  … (${n-3} more)` : '') +
            `

Σ = ${fmt(dbSum)}
` +
            `db = (2 / ${n}) × ${fmt(dbSum)}
` +
            `   = ${fmt(step.db)}` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'weight_update') {
        /* ── weight = weight − α × dw ────────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">Step 3 — Update Weight</span>` +
            `<span class="calc-body">` +
            `weight = weight − α × dw

` +
            `       = ${fmt(step.weight)}
` +
            `       − ${step.learningRate} × (${fmt(step.dw)})

` +
            `       = ${fmt(step.weight)} − ${fmt(step.learningRate * step.dw)}

` +
            `weight = ${fmt(step.newWeight)}` +
            `</span>`;
        updateDisplays(fmt(step.newWeight), fmt(step.bias));

    } else if (step.type === 'bias_update') {
        /* ── bias = bias − α × db ────────────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">Step 4 — Update Bias</span>` +
            `<span class="calc-body">` +
            `bias = bias − α × db

` +
            `     = ${fmt(step.bias)}
` +
            `     − ${step.learningRate} × (${fmt(step.db)})

` +
            `     = ${fmt(step.bias)} − ${fmt(step.learningRate * step.db)}

` +
            `bias = ${fmt(step.newBias)}` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.newBias));

    } else if (step.type === 'weighthist') {
        /* ── weight_hist.append(weight) ──────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">Append weight to weight_hist</span>` +
            `<span class="calc-body">` +
            `weight_hist.append(weight)

` +
            `Appended value : ${fmt(step.weight)}
` +
            `History length : ${step.histLen ?? weight_hist.length + 1} entries
` +
            `Previous entry : ${weight_hist.length > 0 ? fmt(weight_hist[weight_hist.length-1]) : 'none (first entry)'}` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'biashist') {
        /* ── bias_hist.append(bias) ──────────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">Append bias to bias_hist</span>` +
            `<span class="calc-body">` +
            `bias_hist.append(bias)

` +
            `Appended value : ${fmt(step.bias)}
` +
            `History length : ${step.histLen ?? bias_hist.length + 1} entries
` +
            `Previous entry : ${bias_hist.length > 0 ? fmt(bias_hist[bias_hist.length-1]) : 'none (first entry)'}` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'draw') {
        /* ── draw_line(weight, bias, iter) ───────────────────────── */
        const x0 = Math.min(...X_values) - 1;
        const x1 = Math.max(...X_values) + 1;
        calc.innerHTML =
            `<span class="calc-title">Step 5 — draw_line(weight, bias, iter)</span>` +
            `<span class="calc-body">` +
            `Regression line equation:
` +
            `  y = ${fmt(step.weight)} × x + (${fmt(step.bias)})

` +
            `Plot range:
` +
            `  x from ${fmt(x0,2)} to ${fmt(x1,2)}

` +
            `  y at x=${fmt(x0,2)} : ${fmt(step.weight*x0+step.bias)}
` +
            `  y at x=${fmt(x1,2)} : ${fmt(step.weight*x1+step.bias)}

` +
            `← Canvas redrawn with updated line` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else if (step.type === 'iter') {
        /* ── End of iteration summary ────────────────────────────── */
        calc.innerHTML =
            `<span class="calc-title">End of Iteration</span>` +
            `<span class="calc-body">` +
            `Iteration complete.

` +
            `Updated parameters:
` +
            `  weight = ${fmt(step.weight)}
` +
            `  bias   = ${fmt(step.bias)}

` +
            `These values carry into the next iteration.` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));

    } else {
        /* ── fallback for any remaining label-based steps ─────────── */
        calc.innerHTML =
            `<span class="calc-title">Training Complete</span>` +
            `<span class="calc-body">` +
            `Final weight = ${fmt(step.weight)}
` +
            `Final bias   = ${fmt(step.bias)}

` +
            `The model is ready to make predictions.
` +
            `  y = ${fmt(step.weight)} × x + (${fmt(step.bias)})` +
            `</span>`;
        updateDisplays(fmt(step.weight), fmt(step.bias));
    }

    drawRegressionPlot(checkpointSnapshots.slice(0, step.snapshotsUpTo));
}

document.getElementById('startBtn').addEventListener('click', () => {
    const ok = runTraining();
    if (ok) {
        renderStep(trainingSteps[currentStepIndex]);
        /* compute and render validation metrics */
        setTimeout(() => {
            if (!X_values.length || !weight_hist.length) return;
            const w    = weight_hist[weight_hist.length - 1];
            const b    = bias_hist[bias_hist.length - 1];
            const n    = X_values.length;
            const preds = X_values.map(x => w * x + b);
            const yMean = y_values.reduce((a,c) => a+c, 0) / n;
            const mse   = preds.reduce((s,p,i) => s + (p-y_values[i])**2, 0) / n;
            const mae   = preds.reduce((s,p,i) => s + Math.abs(p-y_values[i]), 0) / n;
            const ssTot = y_values.reduce((s,y) => s + (y-yMean)**2, 0);
            const ssRes = preds.reduce((s,p,i) => s + (y_values[i]-p)**2, 0);
            const r2    = 1 - ssRes/ssTot;
            document.getElementById('mse-val').textContent  = mse.toFixed(3);
            document.getElementById('rmse-val').textContent = Math.sqrt(mse).toFixed(3);
            document.getElementById('mae-val').textContent  = mae.toFixed(3);
            document.getElementById('r2-val').textContent   = r2.toFixed(4);
        }, 300);
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

/* ── Keyboard navigation with button flash ── */
function flashBtn(id) {
    ['startBtn','prevBtn','nextBtn','endBtn'].forEach(bid => {
        const b = document.getElementById(bid);
        if (b) b.classList.remove('nav-btn-flash');
    });
    const btn = document.getElementById(id);
    if (!btn) return;
    void btn.offsetWidth;
    btn.classList.add('nav-btn-flash');
    btn.addEventListener('animationend', () => btn.classList.remove('nav-btn-flash'), { once: true });
}

document.addEventListener('keydown', e => {
    const tag = document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowRight') {
        e.preventDefault(); flashBtn('nextBtn');
        if (trainingSteps.length && currentStepIndex < trainingSteps.length - 1) {
            currentStepIndex++; renderStep(trainingSteps[currentStepIndex]);
        }
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault(); flashBtn('prevBtn');
        if (trainingSteps.length && currentStepIndex > 0) {
            currentStepIndex--; renderStep(trainingSteps[currentStepIndex]);
        }
    } else if (e.key === 'Home') {
        e.preventDefault(); flashBtn('startBtn');
        if (trainingSteps.length) { currentStepIndex = 0; renderStep(trainingSteps[0]); }
    } else if (e.key === 'End') {
        e.preventDefault(); flashBtn('endBtn');
        if (trainingSteps.length) {
            currentStepIndex = trainingSteps.length - 1;
            renderStep(trainingSteps[currentStepIndex]);
        }
    }
});
