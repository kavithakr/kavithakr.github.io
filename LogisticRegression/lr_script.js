/* ══════════════════════════════════════════════════════════════════
   Logistic Regression Visualization — lr_script.js
   Fully dynamic: trains in real-time from any x/y dataset.
   ══════════════════════════════════════════════════════════════════ */

/* ── Python code displayed in the editor ── */
const CODE = `import numpy as np

w, b = train(X_train, y_train, lr, iterations)

# ── 1. Sigmoid function ──────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ── 2. Binary cross-entropy loss ─────────────────
def compute_loss(y_true, y_pred):
    eps = 1e-9
    return -np.mean(
        y_true * np.log(y_pred + eps) +
        (1 - y_true) * np.log(1 - y_pred + eps)
    )

# ── 3. Training loop ─────────────────────────────
def train(X, y, lr, iterations):
    w, b = 0.0, 0.0
    for iteration in range(iterations):

        # ── 4. Linear combination ─────────────────
        z = X * w + b

        # ── 5. Apply sigmoid → probabilities ──────
        y_pred = sigmoid(z)

        # ── 6. Compute loss ───────────────────────
        loss = compute_loss(y, y_pred)

        # ── 7. Gradients ──────────────────────────
        dw = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)

        # ── 8. Update parameters ──────────────────
        w = w - lr * dw
        b = b - lr * db

        # ── 9. Draw sigmoid curve ─────────────────
        draw_sigmoid_curve(X, w, b, threshold)

    return w, b

# ── 9. Predict ───────────────────────────────────
def predict(X, w, b, threshold=0.5):
    return (sigmoid(X * w + b) >= threshold).astype(int)

predictions = predict(X_test, w, b, threshold)`;

/* maps each step kind to the corresponding math section ID in the HTML */
const MATH_MAP = {
    call:          'math-init',
    train_fn:      'math-init',
    linear:        'math-linear',
    apply_sig_call:'math-sigmoid',
    sigmoid_fn:    'math-sigmoid',
    loss_call:     'math-loss',
    loss_fn:       'math-loss',
    gradients:     'math-gradients',
    update:        'math-update',
    draw_curve:    'math-boundary',
    done:          'math-predict',
};

/* line ranges matched to each step kind — aligned to CODE string line numbers */
const LINE = {
    call:      [3],                          // w, b = train(X_train, y_train, lr, iterations)
    train_fn:  [18, 19],                // def train + w,b=0 + for loop
    linear:    [22, 23],                    // z = X * w + b
    apply_sig: [25,26],                    // y_pred = sigmoid(z)
    sigmoid:   [5, 6, 7],                   // def sigmoid(z) body
    loss:      [28,29],                    // loss = compute_loss(y, y_pred)
    loss_fn:   [9, 10, 11, 12, 13, 14, 15],// def compute_loss body
    gradients: [31,32,33],               // dw, db
    update:    [35,36,37],               // w = w - lr*dw  b = b - lr*db
  
    draw_call: [39,40],                   // draw_sigmoid_curve(X, w, b, threshold)
    done:      [45,46,47,48],       // predict
};

/* ── render code table ── */
(function renderCode() {
    const lines = CODE.split('\n');
    let html = '';
    lines.forEach((ln, i) => {
        const safe = ln.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') || ' ';
        html += `<tr id="row-${i}"><td class="ln" id="gutter-${i}">${i+1}</td><td class="lc" id="line-${i}">${safe}</td></tr>`;
    });
    document.getElementById('code-table').innerHTML = html;
})();

/* ── highlight / dim ── */
function highlight(nums) {
    document.querySelectorAll('.highlighted-row').forEach(e => e.classList.remove('highlighted-row'));
    document.querySelectorAll('.dimmed-row').forEach(e => e.classList.remove('dimmed-row'));
    if (!nums) return;
    const hl = new Set(nums.map(n => `row-${n-1}`));
    document.querySelectorAll('#code-table tr').forEach(tr => {
        tr.classList.add(hl.has(tr.id) ? 'highlighted-row' : 'dimmed-row');
    });
    document.getElementById(`row-${nums[0]-1}`)?.scrollIntoView({block:'center',behavior:'smooth'});
}

/* ── highlight active math section ── */
function highlightMath(kind) {
    /* remove active class from all sections */
    document.querySelectorAll('.math-section').forEach(el => {
        el.classList.remove('math-active');
    });
    const sectionId = MATH_MAP[kind];
    if (!sectionId) return;
    const el = document.getElementById(sectionId);
    if (!el) return;
    el.classList.add('math-active');
    /* scroll the math section into view if math tab is visible */
    const mathView = document.getElementById('view-math');
    if (mathView && !mathView.classList.contains('hidden')) {
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
}

/* ══════════════════════════════════════════════════════════════════
   CORE MATH  (runs in-browser, no external libraries)
   ══════════════════════════════════════════════════════════════════ */
const sigmoid  = z => 1 / (1 + Math.exp(-z));
const fmt      = (v, d=6) => Number(v).toFixed(d);

function computeLoss(yTrue, yPred) {
    const eps = 1e-9, n = yTrue.length;
    let s = 0;
    for (let i = 0; i < n; i++)
        s += yTrue[i] * Math.log(yPred[i] + eps) + (1 - yTrue[i]) * Math.log(1 - yPred[i] + eps);
    return -s / n;
}

function trainAll(X, y, lr, iterations, nSteps) {
    /* Pre-compute the full training run.
       Always capture iter 0 and iter 1 as the first two snapshots so the
       learner sees consecutive iterations. Remaining nSteps-2 snapshots are
       spread evenly from iter 2 to the final iteration. */
    let w = 0, b = 0;
    const snapshots = [];
    const checkAt = [0, 1];   // always include iter 0 and iter 1
    const remaining = Math.max(0, nSteps - 2);
    for (let s = 0; s < remaining; s++) {
        const idx = Math.round(2 + (s / Math.max(remaining - 1, 1)) * (iterations - 3));
        checkAt.push(Math.min(idx, iterations - 1));
    }
    const checkSet = new Set(checkAt);

    for (let iter = 0; iter < iterations; iter++) {
        const n   = X.length;
        const z   = X.map(x => x * w + b);
        const yp  = z.map(sigmoid);
        const loss= computeLoss(y, yp);
        let dw = 0, db = 0;
        for (let i = 0; i < n; i++) { dw += (yp[i] - y[i]) * X[i]; db += (yp[i] - y[i]); }
        dw /= n; db /= n;

        if (checkSet.has(iter)) {
            snapshots.push({
                iter, w, b, loss, dw, db,
                z: [...z], yp: [...yp],
                wNext: w - lr * dw,
                bNext: b - lr * db,
            });
        }
        w -= lr * dw;
        b -= lr * db;
    }
    return snapshots;
}

/* ══════════════════════════════════════════════════════════════════
   STEP TRACE
   Each snapshot produces 9 sub-steps mirroring the code sections.
   ══════════════════════════════════════════════════════════════════ */
let STEPS = [], cursor = -1;
let X_all = [], y_all = [], X_train = [], y_train = [], X_test = [], y_test = [];
let snapshots = [], finalW = 0, finalB = 0;

function buildSteps(snaps) {
    STEPS = [];
    const lr       = parseFloat(document.getElementById('lrInput').value) || 0.01;
    const iterTotal= document.getElementById('iterInput').value;
    const n        = X_train.length;

    /* helper — ALL samples, no truncation */
    const allZ  = s => X_train.map((x,i) => `  z[${i}]      = ${fmt(x,2)} × (${fmt(s.w,6)}) + (${fmt(s.b,6)}) = ${fmt(s.z[i],6)}`).join('\n');
    const allYP = s => X_train.map((x,i) =>
        `  ŷ[${i}]      = 1/(1+e^(${fmt(-s.z[i],4)})) = ${fmt(s.yp[i],6)}   y=${y_train[i]}`
    ).join('\n');
    const allLoss = s => X_train.map((x,i) => {
        const eps=1e-9;
        const term = y_train[i]*Math.log(s.yp[i]+eps) + (1-y_train[i])*Math.log(1-s.yp[i]+eps);
        return `  [${i}]  y=${y_train[i]}  ŷ=${fmt(s.yp[i],6)}  → ${fmt(term,6)}`;
    }).join('\n');
    const allDW  = s => X_train.map((x,i) =>
        `  [${i}]  (${fmt(s.yp[i],6)} - ${y_train[i]}) × ${fmt(x,4)} = ${fmt((s.yp[i]-y_train[i])*x,6)}`
    ).join('\n');
    const allDB  = s => X_train.map((x,i) =>
        `  [${i}]  ${fmt(s.yp[i],6)} - ${y_train[i]} = ${fmt(s.yp[i]-y_train[i],6)}`
    ).join('\n');

    /* ─────────────────────────────────────────────
       STEP 0 — line 3: w,b = train(...)  [Start]
    ───────────────────────────────────────────── */
    STEPS.push({
        lines: LINE.call, kind: 'call',
        title: 'Call train() — line 3',
        w: 0, b: 0, curves: [],
        detail:
            `w, b = train(X_train, y_train, lr, iterations)\n\n` +
            `Training samples : ${n}\n` +
            `Test samples     : ${X_test.length}\n` +
            `Learning rate    : ${lr}\n` +
            `Iterations       : ${iterTotal}\n\n` +
            `Control jumps to def train() on line 18 →`
    });

    /* ─────────────────────────────────────────────
       STEP 1 — lines 18-20: enter train(), w=b=0
    ───────────────────────────────────────────── */
    STEPS.push({
        lines: LINE.train_fn, kind: 'train_fn',
        title: 'Enter train() — initialise w=0, b=0',
        w: 0, b: 0, curves: [],
        detail:
            `def train(X, y, lr, iterations):\n` +
            `    w, b = 0.0, 0.0\n` +
            `    for iteration in range(iterations):\n\n` +
            `Initialised:\n` +
            `  w = 0.0\n` +
            `  b = 0.0\n` +
            `  lr          = ${lr}\n` +
            `  iterations  = ${iterTotal}`
    });

    /* ─────────────────────────────────────────────
       ITERATION BLOCKS
       iter 0 and iter 1 → full detail, sigmoid and
         compute_loss functions highlighted when called.
       iter 2+ → steps 4-8 only (no function jump)
    ───────────────────────────────────────────── */
    const curvesAccum = [];

    snaps.forEach((s, si) => {
        const iterLabel = si === 0 ? 'Iteration 0'
                        : si === snaps.length - 1 ? `Iteration ${s.iter} (final)`
                        : `Iteration ${s.iter}`;
        const showFull = si <= 1;  // show sigmoid+loss fn jump only for iter 0 and 1

        /* Steps 4-8 pass curvesAccum (previous iterations only) as the faded
           history. The current iteration is drawn as the bold current curve
           via s.w/s.b — not as a past curve — so nothing overlaps. */

        /* Step 4 — z = X * w + b */
        STEPS.push({
            lines: LINE.linear, kind: 'linear',
            title: `Step 4 — z = X·w + b  [${iterLabel}]`,
            w: s.w, b: s.b, curves: [...curvesAccum],
            detail:
                `z = X · w + b\n\n` +
                `w = ${fmt(s.w,6)}   b = ${fmt(s.b,6)}\n\n` +
                allZ(s)
        });

        /* Step 5a — call site: y_pred = sigmoid(z) */
        STEPS.push({
            lines: LINE.apply_sig, kind: 'apply_sig_call',
            title: `Step 5 — y_pred = sigmoid(z)  [${iterLabel}]`,
            w: s.w, b: s.b, curves: [...curvesAccum],
            detail:
                `y_pred = sigmoid(z)\n\n` +
                (showFull ? `→ calling def sigmoid(z) on line 6…\n\n` : '') +
                `sigmoid(z) = 1 / (1 + e^(−z))\n\n` +
                allYP(s)
        });

        /* Step 5b — highlight sigmoid function body (iter 0 & 1 only) */
        if (showFull) {
            STEPS.push({
                lines: LINE.sigmoid, kind: 'sigmoid_fn',
                title: `def sigmoid(z)  [${iterLabel}]`,
                w: s.w, b: s.b, curves: [...curvesAccum],
                detail:
                    `def sigmoid(z):\n` +
                    `    return 1 / (1 + np.exp(-z))\n\n` +
                    `Computes probability for each z:\n\n` +
                    allYP(s) +
                    `\n\n← returns ŷ to line 26`
            });
        }

        /* Step 6a — call site: loss = compute_loss(y, y_pred) */
        const prevLoss = si > 0 ? snaps[si-1].loss : null;
        STEPS.push({
            lines: LINE.loss, kind: 'loss_call',
            title: `Step 6 — loss = compute_loss(y, y_pred)  [${iterLabel}]`,
            w: s.w, b: s.b, curves: [...curvesAccum],
            detail:
                `loss = compute_loss(y, y_pred)\n\n` +
                (showFull ? `→ calling def compute_loss() on line 10…\n\n` : '') +
                `L = −mean( y·log(ŷ) + (1−y)·log(1−ŷ) )\n\n` +
                allLoss(s) +
                `\n\nLoss = ${fmt(s.loss, 8)}` +
                (prevLoss !== null
                    ? `\nPrev = ${fmt(prevLoss, 8)}\nΔ    = ${fmt(s.loss - prevLoss, 8)}`
                    : '')
        });

        /* Step 6b — highlight compute_loss function body (iter 0 & 1 only) */
        if (showFull) {
            STEPS.push({
                lines: LINE.loss_fn, kind: 'loss_fn',
                title: `def compute_loss(y_true, y_pred)  [${iterLabel}]`,
                w: s.w, b: s.b, curves: [...curvesAccum],
                detail:
                    `def compute_loss(y_true, y_pred):\n` +
                    `    eps = 1e-9\n` +
                    `    return -np.mean(\n` +
                    `        y_true * np.log(y_pred + eps) +\n` +
                    `        (1-y_true) * np.log(1-y_pred + eps))\n\n` +
                    `Per-sample terms:\n` +
                    allLoss(s) +
                    `\n\nLoss = ${fmt(s.loss, 8)}\n← returns to line 29`
            });
        }

        /* Step 7 — dw, db */
        STEPS.push({
            lines: LINE.gradients, kind: 'gradients',
            title: `Step 7 — Compute gradients  [${iterLabel}]`,
            w: s.w, b: s.b, curves: [...curvesAccum],
            detail:
                `dw = mean( (ŷ − y) · X )\n\n` +
                allDW(s) +
                `\n\ndw = ${fmt(s.dw, 8)}\n\n` +
                `db = mean( ŷ − y )\n\n` +
                allDB(s) +
                `\n\ndb = ${fmt(s.db, 8)}`
        });

        /* Step 8 — update w, b */
        STEPS.push({
            lines: LINE.update, kind: 'update',
            title: `Step 8 — Update w, b  [${iterLabel}]`,
            w: s.wNext, b: s.bNext, curves: [...curvesAccum],
            detail:
                `w ← w − lr × dw\n` +
                `  = ${fmt(s.w,6)} − ${lr} × (${fmt(s.dw,8)})\n` +
                `  = ${fmt(s.w,6)} − (${fmt(lr*s.dw,8)})\n` +
                `  = ${fmt(s.wNext,6)}\n\n` +
                `b ← b − lr × db\n` +
                `  = ${fmt(s.b,6)} − ${lr} × (${fmt(s.db,8)})\n` +
                `  = ${fmt(s.b,6)} − (${fmt(lr*s.db,8)})\n` +
                `  = ${fmt(s.bNext,6)}`
        });

        /* Step 9 — draw_sigmoid_curve(X, w, b, threshold) → canvas redraws
           At this point w and b have just been updated (wNext/bNext).
           Add the updated curve to curvesAccum NOW so it becomes visible
           as faded history in all future steps, and pass it to this step
           so the canvas shows it as part of the accumulated past curves. */
        curvesAccum.push({ w: s.wNext, b: s.bNext, label: iterLabel });
        STEPS.push({
            lines: LINE.draw_call,  kind: 'draw_curve',
            title: `Step 9 — draw_sigmoid_curve()  [${iterLabel}]`,
            w: s.wNext, b: s.bNext, curves: [...curvesAccum],
            detail:
                `draw_sigmoid_curve(X, w, b, threshold)\n\n` +
                `w         = ${fmt(s.wNext,6)}\n` +
                `b         = ${fmt(s.bNext,6)}\n` +
                `threshold = ${document.getElementById('thrInput').value}\n\n` +
                `x_plot = np.linspace(${fmt(Math.min(...X_all)-5,2)}, ` +
                                     `${fmt(Math.max(...X_all)+5,2)}, 300)\n` +
                `y_plot = 1 / (1 + np.exp(-(x_plot × ${fmt(s.wNext,4)} + (${fmt(s.bNext,4)}))))\n\n` +
                `← Sigmoid curve drawn on canvas with updated w and b`
        });
    });

    /* ─────────────────────────────────────────────
       FINAL — predict
    ───────────────────────────────────────────── */
    curvesAccum.push({ w: finalW, b: finalB, label: 'Final' });
    STEPS.push({
        lines: LINE.done, kind: 'done',
        title: 'Step 9 — Training complete → Predict',
        w: finalW, b: finalB, curves: [...curvesAccum],
        detail:
            `Training complete!\n\n` +
            `Final  w = ${fmt(finalW, 6)}\n` +
            `Final  b = ${fmt(finalB, 6)}\n\n` +
            `predictions = predict(X_test, w, b, threshold)\n` +
            `           = (sigmoid(X·w + b) >= ${document.getElementById('thrInput').value}).astype(int)\n\n` +
            `Test samples: ${X_test.length}`
    });
}

/* ══════════════════════════════════════════════════════════════════
   CANVAS — draws multiple sigmoid curves
   ══════════════════════════════════════════════════════════════════ */
function drawCanvas(curves, currentW, currentB) {
    const canvas = document.getElementById('sigmoid-canvas');
    const W = canvas.offsetWidth  || 400;
    const H = canvas.offsetHeight || 280;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    if (!X_train.length) return;

    const xMin = Math.min(...X_all) - 5;
    const xMax = Math.max(...X_all) + 5;
    const px = x => ((x - xMin) / (xMax - xMin)) * (W - 60) + 30;
    const py = p => H - 30 - p * (H - 50);

    /* axes */
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(30, 10); ctx.lineTo(30, H-30); ctx.lineTo(W-10, H-30); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('x', W-12, H-28); ctx.fillText('P', 6, 14);

    /* threshold value — read once, used by both grid and threshold line */
    const thr = parseFloat(document.getElementById('thrInput').value) || 0.5;

    /* y-axis ticks — skip the grid line at the threshold value to avoid overlap */
    [0, 0.25, 0.5, 0.75, 1].forEach(v => {
        const y = py(v);
        ctx.fillStyle = '#94a3b8'; ctx.fillText(v.toFixed(2), 0, y+3);
        if (Math.abs(v - thr) < 0.001) return;   /* threshold line drawn separately */
        ctx.setLineDash([3,3]); ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 0.7;
        ctx.beginPath(); ctx.moveTo(30, y); ctx.lineTo(W-10, y); ctx.stroke();
        ctx.setLineDash([]);
    });

    /* threshold line — single amber dashed line, one style only */
    ctx.setLineDash([6,3]); ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(30, py(thr)); ctx.lineTo(W-10, py(thr)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f59e0b'; ctx.fillText(`threshold=${thr}`, W-110, py(thr)-3);

    /* data points */
    X_train.forEach((x, i) => {
        ctx.beginPath();
        ctx.arc(px(x), py(y_train[i]), 4, 0, 2*Math.PI);
        ctx.fillStyle   = y_train[i] === 1 ? '#16a34a' : '#dc2626';
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
        ctx.fill(); ctx.stroke();
    });

    /* past curves — dark distinct colours, visible line weight */
    const palette = ['#1d4ed8','#7c3aed','#047857','#b45309','#be185d','#0369a1','#b91c1c','#15803d'];
    const xs = [];
    for (let i = 0; i <= 200; i++) xs.push(xMin + i * (xMax - xMin) / 200);

    const pastCurves = curves.filter(c =>
        !(Math.abs(c.w - currentW) < 1e-9 && Math.abs(c.b - currentB) < 1e-9)
    );

    pastCurves.forEach((c, ci) => {
        ctx.beginPath();
        ctx.lineWidth   = 1.4;
        ctx.strokeStyle = palette[ci % palette.length];
        ctx.globalAlpha = 0.7;
        xs.forEach((x, xi) => {
            const y = py(sigmoid(x * c.w + c.b));
            xi === 0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
    });

    /* current curve (bold) */
    ctx.beginPath(); ctx.lineWidth = 2; ctx.strokeStyle = '#1e3a5f';
    xs.forEach((x, xi) => {
        const y = py(sigmoid(x * currentW + currentB));
        xi === 0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
    });
    ctx.stroke();

    /* ── 1. Yes / No class labels — bottom-right corner ── */
    const brX = W - 10, brY = H - 36;   /* anchor: bottom-right */
    ctx.font = '9px sans-serif';
    [['#16a34a', '● Yes (y=1)'], ['#dc2626', '● No  (y=0)']].forEach(([col, lbl], i) => {
        const tw = ctx.measureText(lbl).width;
        ctx.fillStyle = col;
        ctx.fillText(lbl, brX - tw, brY + i * 13);
    });

    /* ── 3. Iteration legend — top-left corner ── */
    if (pastCurves.length) {
        const legX = 34, legY0 = 12, lineH = 13;
        /* background box */
        const boxW = 115, boxH = pastCurves.length * lineH + 6;
        ctx.fillStyle = 'rgba(255,255,255,0.82)';
        ctx.fillRect(legX - 3, legY0 - 9, boxW, boxH);
        ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 0.7;
        ctx.strokeRect(legX - 3, legY0 - 9, boxW, boxH);

        ctx.font = '8.5px Consolas, monospace';
        pastCurves.forEach((c, ci) => {
            const col   = palette[ci % palette.length];
            const shortLabel = c.label
                .replace('Iteration ', 'Iter ')
                .replace(' (initial)', '')
                .replace(' (final)',   '');
            const y = legY0 + ci * lineH;
            /* colour swatch */
            ctx.fillStyle = col;
            ctx.fillRect(legX, y - 7, 18, 7);
            /* label text */
            ctx.fillStyle = '#1e293b';
            ctx.fillText(shortLabel, legX + 22, y);
        });
    }
}

/* ══════════════════════════════════════════════════════════════════
   STEP DISPLAY
   ══════════════════════════════════════════════════════════════════ */
function show(i) {
    if (!STEPS.length) return;
    cursor = Math.max(0, Math.min(i, STEPS.length-1));
    const s = STEPS[cursor];

    highlight(s.lines);
    document.getElementById('step-counter').textContent = `Step ${cursor+1} / ${STEPS.length}`;
    document.getElementById('weightDisplay').value = fmt(s.w, 6);
    document.getElementById('biasDisplay').value   = fmt(s.b, 6);

    const loss = s.kind === 'loss' || s.kind === 'loop'
        ? fmt(STEPS.slice(0, cursor+1).reverse().find(st => st.kind==='loss'||st.kind==='loop')?.detail?.match(/Loss\s+=\s+([\d.\-eE]+)/)?.[1] ?? s.detail?.match(/Loss\s+=\s+([\d.\-eE]+)/)?.[1] ?? '—')
        : '—';
    document.getElementById('lossDisplay').value = s.detail?.match(/L\s*=\s*([\d.\-eE]+)/)?.[1] ?? s.detail?.match(/Loss\s+=\s*([\d.\-eE]+)/)?.[1] ?? '—';

    document.getElementById('calc').innerHTML =
        `<span class="calc-title">${s.title}</span>` +
        `<span class="calc-body">${s.detail.replace(/</g,'&lt;')}</span>`;

    document.getElementById('step-summary').innerHTML =
        `<span style="font-weight:700;color:var(--navy);">Step ${cursor+1}</span>: ${s.title}<br>` +
        `<span style="color:var(--accent);font-family:var(--mono);font-size:10px;">w = ${fmt(s.w,4)}</span>&nbsp;&nbsp;` +
        `<span style="color:var(--red);font-family:var(--mono);font-size:10px;">b = ${fmt(s.b,4)}</span>`;

    drawCanvas(s.curves, s.w, s.b);
    highlightMath(s.kind);
}

/* ══════════════════════════════════════════════════════════════════
   METRICS
   ══════════════════════════════════════════════════════════════════ */
function renderMetrics(w, b) {
    const thr = parseFloat(document.getElementById('thrInput').value) || 0.5;
    const preds = X_test.map(x => sigmoid(x * w + b) >= thr ? 1 : 0);
    const tp = preds.filter((p,i) => p===1 && y_test[i]===1).length;
    const tn = preds.filter((p,i) => p===0 && y_test[i]===0).length;
    const fp = preds.filter((p,i) => p===1 && y_test[i]===0).length;
    const fn = preds.filter((p,i) => p===0 && y_test[i]===1).length;
    const acc  = (tp+tn)/(tp+tn+fp+fn) || 0;
    const prec = tp/(tp+fp) || 0;
    const rec  = tp/(tp+fn) || 0;
    const spec = tn/(tn+fp) || 0;
    const f1   = 2*prec*rec/(prec+rec) || 0;

    const card = (l,v,c) =>
        `<div class="metric-card" style="border-color:${c};background:${c}15;">
           <span class="mc-label" style="color:${c};">${l}</span>
           <span class="mc-value" style="color:${c};">${(v*100).toFixed(2)}%</span>
         </div>`;
    document.getElementById('metrics').innerHTML =
        card('Accuracy',    acc,  '#3b82f6') +
        card('Precision',   prec, '#7c3aed') +
        card('Recall',      rec,  '#0891b2') +
        card('Specificity', spec, '#d97706') +
        card('F1 Score',    f1,   '#059669') +
        `<div class="border rounded px-3 py-1 text-center" style="border-color:#64748b!important;background:#64748b18;">
           <div style="font-size:.6rem;font-weight:700;color:#64748b;">TP/TN/FP/FN</div>
           <div style="font-size:.8rem;font-weight:700;">${tp}/${tn}/${fp}/${fn}</div>
         </div>`;
}

/* ══════════════════════════════════════════════════════════════════
   TRAIN — called on Start
   ══════════════════════════════════════════════════════════════════ */
function runTraining() {
    if (!X_all.length) { alert('Load a dataset first.'); return; }

    const lr         = parseFloat(document.getElementById('lrInput').value)   || 0.01;
    const iterations = parseInt(document.getElementById('iterInput').value)    || 100000;
    const nSteps     = parseInt(document.getElementById('stepsInput').value)   || 8;
    const splitPct   = parseInt(document.getElementById('splitInput').value)   || 80;
    const splitIdx   = Math.floor(X_all.length * splitPct / 100);

    X_train = X_all.slice(0, splitIdx); y_train = y_all.slice(0, splitIdx);
    X_test  = X_all.slice(splitIdx);   y_test  = y_all.slice(splitIdx);

    if (!X_train.length) { alert('Not enough data for selected split.'); return; }

    /* highlight test rows */
    document.querySelectorAll('#tbody tr').forEach((tr, i) => {
        tr.classList.toggle('table-secondary', i >= splitIdx);
    });

    snapshots = trainAll(X_train, y_train, lr, iterations, nSteps);
    finalW = snapshots[snapshots.length-1].wNext;
    finalB = snapshots[snapshots.length-1].bNext;

    buildSteps(snapshots);
    renderMetrics(finalW, finalB);
    show(0);
}

/* ══════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════ */
document.getElementById('startBtn').onclick = runTraining;
document.getElementById('nextBtn').onclick  = () => { if (STEPS.length) show(cursor+1); };
document.getElementById('prevBtn').onclick  = () => { if (STEPS.length) show(cursor-1); };
document.getElementById('endBtn').onclick   = () => {
    if (!STEPS.length) runTraining(); else show(STEPS.length-1);
};

/* ── Keyboard navigation — Left/Right arrow keys ── */
function flashBtn(id) {
    /* first remove flash from ALL nav buttons so switching between
       prev and next always triggers a clean animation */
    ['startBtn','prevBtn','nextBtn','endBtn'].forEach(bid => {
        const b = document.getElementById(bid);
        if (b) b.classList.remove('nav-btn-flash');
    });
    const btn = document.getElementById(id);
    if (!btn) return;
    void btn.offsetWidth;   /* force reflow to restart animation */
    btn.classList.add('nav-btn-flash');
    btn.addEventListener('animationend', () => btn.classList.remove('nav-btn-flash'), { once: true });
}

document.addEventListener('keydown', e => {
    /* ignore when user is typing inside an input or select */
    const tag = document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

    if (e.key === 'ArrowRight') {
        e.preventDefault();
        flashBtn('nextBtn');
        if (STEPS.length) show(cursor + 1);
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        flashBtn('prevBtn');
        if (STEPS.length) show(cursor - 1);
    } else if (e.key === 'Home') {
        e.preventDefault();
        flashBtn('startBtn');
        if (!STEPS.length) runTraining(); else show(0);
    } else if (e.key === 'End') {
        e.preventDefault();
        flashBtn('endBtn');
        if (!STEPS.length) runTraining(); else show(STEPS.length - 1);
    }
});

/* ══════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════ */
function loadData(arr) {
    if (!arr.length) { alert('Empty dataset'); return; }
    const cols = Object.keys(arr[0]);
    const xCol = cols[0], yCol = cols[cols.length-1];
    X_all = arr.map(r => parseFloat(r[xCol]));
    y_all = arr.map(r => parseInt(r[yCol]));

    if (X_all.some(isNaN) || y_all.some(isNaN)) {
        alert('Could not parse data. Make sure x column is numeric and y column is 0/1.');
        return;
    }

    /* render table */
    const thead = document.getElementById('thead');
    const tbody = document.getElementById('tbody');
    thead.innerHTML = `<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
    tbody.innerHTML = arr.map(r=>
        `<tr>${cols.map(c=>`<td>${r[c]??''}</td>`).join('')}</tr>`).join('');
    document.getElementById('table-box').classList.remove('hidden');

    STEPS = []; cursor = -1;
    document.getElementById('calc').innerHTML =
        `<span style="color:#475569;">${arr.length} rows loaded. Press <span style="color:#38bdf8;">Start</span> to train.</span>`;
    document.getElementById('step-counter').textContent = `Step 0 / 0`;
    document.getElementById('metrics').innerHTML = '';
    drawCanvas([], 0, 0);
}

/* ── preset dataset ── */
const PRESET = [
    {x:10,y:0},{x:15,y:0},{x:20,y:0},{x:25,y:0},{x:30,y:0},
    {x:35,y:0},{x:40,y:1},{x:55,y:1},{x:60,y:1},{x:70,y:1},
    {x:75,y:1},{x:80,y:1},{x:85,y:1},
];

/*const setActive = id => {
    if (typeof window.setActive === 'function') window.setActive(id);
    ['presetBtn','randomBtn','customBtn'].forEach(b =>
        document.getElementById(b).classList.toggle('btn-active', b===id));
};
*/

document.getElementById('presetBtn').onclick = () => {
    setActive('presetBtn');
    document.getElementById('upload-box').classList.add('hidden');
    loadData(JSON.parse(JSON.stringify(PRESET)));
};

document.getElementById('randomBtn').onclick = () => {
    setActive('randomBtn');
    document.getElementById('upload-box').classList.add('hidden');
    const data = Array.from({length:14}, (_,i) => {
        const x = (i+1)*6;
        return {x, y: x >= 40 ? (Math.random()<0.85?1:0) : (Math.random()<0.15?1:0)};
    });
    loadData(data);
};

document.getElementById('customBtn').onclick = () => {
    setActive('customBtn');
    document.getElementById('upload-box').classList.remove('hidden');
};

let chosenFile = null;
document.getElementById('fileInput').onchange = e => { chosenFile = e.target.files[0]; };
document.getElementById('submitFile').onclick  = () => {
    if (!chosenFile) { alert('Please choose a file first.'); return; }
    const fr = new FileReader();
    fr.onload = ev => {
        try {
            const wb = XLSX.read(new Uint8Array(ev.target.result), {type:'array'});
            const json = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], {defval:''});
            loadData(json);
            document.getElementById('upload-box').classList.add('hidden');
        } catch(err) { alert('Could not read file: ' + err.message); }
    };
    fr.readAsArrayBuffer(chosenFile);
};

/* auto-load preset */
document.getElementById('presetBtn').click();
