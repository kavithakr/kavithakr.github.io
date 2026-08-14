/* ══════════════════════════════════════════════════════════════════
   K-Nearest Neighbours Visualization — knn_script.js
   Fully dynamic: works on any 2-feature (x1, x2, y) dataset.
   ══════════════════════════════════════════════════════════════════ */

/* ── Python code displayed in the editor ────────────────────────── */
const CODE = `import numpy as np
from collections import Counter

# ── 1. Load data and split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ── 2. Select a query point ──────────────────────────────
query = X_test[i]

# ── 3. Compute distance to every training point ──────────
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

distances = [(euclidean(query, x), y)
             for x, y in zip(X_train, y_train)]

# ── 4. Sort by distance (ascending) ─────────────────────
distances.sort(key=lambda d: d[0])

# ── 5. Select k nearest neighbours ──────────────────────
k_nearest = distances[:k]

# ── 6. Majority vote → predicted class ───────────────────
votes  = Counter(label for _, label in k_nearest)
y_pred = votes.most_common(1)[0][0]

# ── 7. Collect all predictions ───────────────────────────
predictions.append(y_pred)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)`;

/* line ranges aligned to each step kind */
const LINE = {
    init:      [1, 2, 3, 4,5],
    query:     [ 7,8],
    distance:  [10, 11, 12, 13, 14,15],
    sort:      [ 17,18],
    kneighbors:[20, 21],
    vote:      [23, 24, 25],
    done:      [27, 28, 29,30],
};

/* ── render code table ─────────────────────────────────────────── */
(function renderCode() {
    const lines = CODE.split('\n');
    let html = '';
    lines.forEach((ln, i) => {
        const safe = ln.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') || ' ';
        html += `<tr id="row-${i}"><td class="ln">${i+1}</td><td class="lc" id="line-${i}">${safe}</td></tr>`;
    });
    document.getElementById('code-table').innerHTML = html;
})();

/* ── highlight / dim ───────────────────────────────────────────── */
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

/* ══════════════════════════════════════════════════════════════════
   CORE KNN ALGORITHM
   ══════════════════════════════════════════════════════════════════ */
function euclidean(a, b) {
    return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2);
}
function manhattan(a, b) {
    return Math.abs(a[0]-b[0]) + Math.abs(a[1]-b[1]);
}
function distFn() {
    return document.getElementById('distInput').value === 'manhattan' ? manhattan : euclidean;
}
function fmt(v, d=4) { return Number(v).toFixed(d); }

function knnPredict(query, trainX, trainY, k) {
    const dist = distFn();
    const dists = trainX.map((x, i) => ({ d: dist(query, x), label: trainY[i], idx: i }));
    dists.sort((a, b) => a.d - b.d);
    const knn = dists.slice(0, k);
    const votes = {};
    knn.forEach(n => { votes[n.label] = (votes[n.label] || 0) + 1; });
    const pred = Object.keys(votes).reduce((a,b) => votes[a] >= votes[b] ? a : b);
    return { pred: parseInt(pred), knn, allDists: dists };
}

/* ══════════════════════════════════════════════════════════════════
   GLOBAL STATE
   ══════════════════════════════════════════════════════════════════ */
let ALL_DATA = [], TRAIN_X = [], TRAIN_Y = [], TEST_X = [], TEST_Y = [];
let STEPS = [], cursor = -1;

/* ══════════════════════════════════════════════════════════════════
   BUILD STEP TRACE
   ══════════════════════════════════════════════════════════════════ */
function buildSteps() {
    STEPS = [];
    const k        = parseInt(document.getElementById('kInput').value) || 5;
    const nQuery   = parseInt(document.getElementById('queryInput').value) || 5;
    const distName = document.getElementById('distInput').value;
    const dist     = distFn();

    /* randomly pick nQuery test points (or all if fewer) */
    const queryIdxs = [];
    const pool = [...Array(TEST_X.length).keys()];
    while (queryIdxs.length < Math.min(nQuery, pool.length)) {
        const r = Math.floor(Math.random() * pool.length);
        queryIdxs.push(pool.splice(r, 1)[0]);
    }

    /* ── step 0: init ── */
    STEPS.push({
        kind: 'init', lines: LINE.init,
        title: 'Load data and initialise KNN',
        detail:
            `Algorithm      : K-Nearest Neighbours\n` +
            `k              : ${k}\n` +
            `Distance metric: ${distName}\n` +
            `Training points: ${TRAIN_X.length}\n` +
            `Query points   : ${queryIdxs.length}\n` +
            `Classes        : 0, 1`,
        queryPt: null, knn: [], allDists: [], pred: null,
        activePtIdx: -1, shownDists: [],
        classified: [], queryIdxs
    });

    const classified = [];   // accumulates as we go

    queryIdxs.forEach((qi, si) => {
        const query      = TEST_X[qi];
        const trueLabel  = TEST_Y[qi];
        const { pred, knn, allDists } = knnPredict(query, TRAIN_X, TRAIN_Y, k);

        /* ── step: select query point ── */
        STEPS.push({
            kind: 'query', lines: LINE.query,
            title: `Query point ${si+1} / ${queryIdxs.length}`,
            detail:
                `Query point    : (${fmt(query[0],3)}, ${fmt(query[1],3)})\n` +
                `True label     : ${trueLabel}\n` +
                `Total training : ${TRAIN_X.length} points\n\n` +
                `Computing distance from query\nto every training point…`,
            queryPt: query, knn: [], allDists: [], pred: null,
            activePtIdx: -1, shownDists: [],
            classified: [...classified], queryIdxs
        });

        /* ── ONE STEP PER TRAINING POINT — distance computation ── */
        const shownDistsSoFar = [];
        TRAIN_X.forEach((tpt, ti) => {
            const d   = dist(query, tpt);
            const lbl = TRAIN_Y[ti];

            /* build the formula string */
            const dx = fmt(tpt[0] - query[0], 4);
            const dy = fmt(tpt[1] - query[1], 4);
            let formula;
            if (distName === 'euclidean') {
                formula =
                    `d = √( (x1ᵢ − q1)² + (x2ᵢ − q2)² )\n` +
                    `  = √( (${fmt(tpt[0],2)} − ${fmt(query[0],2)})² + (${fmt(tpt[1],2)} − ${fmt(query[1],2)})² )\n` +
                    `  = √( ${fmt((tpt[0]-query[0])**2,4)} + ${fmt((tpt[1]-query[1])**2,4)} )\n` +
                    `  = √${fmt((tpt[0]-query[0])**2+(tpt[1]-query[1])**2,4)}\n` +
                    `  = ${fmt(d,6)}`;
            } else {
                formula =
                    `d = |x1ᵢ − q1| + |x2ᵢ − q2|\n` +
                    `  = |${fmt(tpt[0],2)} − ${fmt(query[0],2)}|  + |${fmt(tpt[1],2)} − ${fmt(query[1],2)}|\n` +
                    `  = ${fmt(Math.abs(tpt[0]-query[0]),4)} + ${fmt(Math.abs(tpt[1]-query[1]),4)}\n` +
                    `  = ${fmt(d,6)}`;
            }

            shownDistsSoFar.push({ idx: ti, d, label: lbl });

            STEPS.push({
                kind: 'dist_one', lines: LINE.distance,
                title: `Distance to point [${ti+1}/${TRAIN_X.length}]  class=${lbl}`,
                detail:
                    `Training point [${ti}]\n` +
                    `  x = (${fmt(tpt[0],2)}, ${fmt(tpt[1],2)})  class=${lbl}\n\n` +
                    `Query\n` +
                    `  q = (${fmt(query[0],2)}, ${fmt(query[1],2)})\n\n` +
                    formula +
                    `\n\nDistances computed so far: ${shownDistsSoFar.length} / ${TRAIN_X.length}`,
                queryPt: query, knn: [], allDists: [],
                activePtIdx: ti,                            /* ← current point */
                shownDists: [...shownDistsSoFar],           /* ← all lines drawn so far */
                pred: null,
                classified: [...classified], queryIdxs
            });
        });

        /* ── step: sort distances ── */
        STEPS.push({
            kind: 'sort', lines: LINE.sort,
            title: `Sort distances (ascending)`,
            detail:
                `Sorted by distance (ascending):\n\n` +
                allDists.slice(0, k+2).map((d,i)=>
                    `  ${i < k ? '►' : ' '} [${d.idx}]` +
                    ` d=${fmt(d.d,4)}  y=${d.label}` +
                    (i === k-1 ? `  ← k=${k} boundary` : '')
                ).join('\n'),
            queryPt: query, knn: [], allDists,
            activePtIdx: -1, shownDists: [...shownDistsSoFar],
            pred: null,
            classified: [...classified], queryIdxs
        });

        /* ── step: select k nearest ── */
        STEPS.push({
            kind: 'kneighbors', lines: LINE.kneighbors,
            title: `Select k=${k} nearest neighbours`,
            detail:
                `k = ${k}\n\n` +
                `k nearest neighbours:\n` +
                knn.map((n,i)=>
                    `  ${i+1}. [${n.idx}]` +
                    ` (${fmt(TRAIN_X[n.idx][0],2)},${fmt(TRAIN_X[n.idx][1],2)})` +
                    `  d=${fmt(n.d,4)}  class=${n.label}`
                ).join('\n'),
            queryPt: query, knn, allDists,
            activePtIdx: -1, shownDists: [...shownDistsSoFar],
            pred: null,
            classified: [...classified], queryIdxs
        });

        /* ── step: vote ── */
        const votes = {};
        knn.forEach(n => { votes[n.label] = (votes[n.label]||0)+1; });
        STEPS.push({
            kind: 'vote', lines: LINE.vote,
            title: `Majority vote → class ${pred}`,
            detail:
                `Vote count:\n` +
                Object.entries(votes).map(([cls,cnt])=>
                    `  Class ${cls}: ${cnt} vote${cnt>1?'s':''} ${'█'.repeat(cnt)}`
                ).join('\n') +
                `\n\nMajority class : ${pred}\n` +
                `True label     : ${trueLabel}\n` +
                `Result         : ${pred === trueLabel ? '✓ Correct' : '✗ Wrong'}`,
            queryPt: query, knn, allDists,
            activePtIdx: -1, shownDists: [...shownDistsSoFar],
            pred,
            classified: [...classified], queryIdxs
        });

        classified.push({ pt: query, pred, trueLabel, qi });
    });

    /* ── final step ── */
    const correct = classified.filter(c => c.pred === c.trueLabel).length;
    const acc = classified.length ? correct / classified.length : 0;
    STEPS.push({
        kind: 'done', lines: LINE.done,
        title: 'Classification complete',
        detail:
            `All ${classified.length} query points classified.\n\n` +
            classified.map((c,i)=>
                `  Q${i+1}: pred=${c.pred}  true=${c.trueLabel}  ${c.pred===c.trueLabel?'✓':'✗'}`
            ).join('\n') +
            `\n\nAccuracy : ${correct}/${classified.length} = ${(acc*100).toFixed(1)}%`,
        queryPt: null, knn: [], allDists: [],
        activePtIdx: -1, shownDists: [],
        pred: null,
        classified: [...classified], queryIdxs
    });
}

/* ══════════════════════════════════════════════════════════════════
   CANVAS DRAWING
   ══════════════════════════════════════════════════════════════════ */
const CLASS_COL  = ['#dc2626', '#2563eb'];   // class 0 = red, 1 = blue
const CLASS_DARK = ['#991b1b', '#1e3a5f'];
const CLASS_NAME = ['Class 0', 'Class 1'];

function drawCanvas(step) {
    const canvas = document.getElementById('knn-canvas');
    const W = canvas.offsetWidth  || 420;
    const H = canvas.offsetHeight || 310;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    if (!TRAIN_X.length) return;

    /* coordinate mapping with padding */
    const pad = 36;
    const allX1 = ALL_DATA.map(d=>d.x1), allX2 = ALL_DATA.map(d=>d.x2);
    const x1Min = Math.min(...allX1), x1Max = Math.max(...allX1);
    const x2Min = Math.min(...allX2), x2Max = Math.max(...allX2);
    const rx = r => pad + (r - x1Min) / (x1Max - x1Min + 1e-9) * (W - 2*pad);
    const ry = r => H - pad - (r - x2Min) / (x2Max - x2Min + 1e-9) * (H - 2*pad);

    /* axes */
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, H-pad); ctx.lineTo(W-10, H-pad); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '9px sans-serif';
    ctx.fillText('x1', W-14, H-pad+2);
    ctx.save(); ctx.translate(12, H/2); ctx.rotate(-Math.PI/2); ctx.fillText('x2', 0, 0); ctx.restore();

    /* grid lines */
    ctx.setLineDash([2,3]); ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 0.7;
    for (let t = 0; t <= 4; t++) {
        const gx = pad + t*(W-2*pad)/4;
        const gy = (H-pad) - t*(H-2*pad)/4;
        ctx.beginPath(); ctx.moveTo(gx, 10); ctx.lineTo(gx, H-pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W-10, gy); ctx.stroke();
        const xv = (x1Min + t*(x1Max-x1Min)/4).toFixed(1);
        const yv = (x2Min + t*(x2Max-x2Min)/4).toFixed(1);
        ctx.fillStyle='#94a3b8';
        ctx.fillText(xv, gx-8, H-pad+10);
        ctx.fillText(yv, 2, gy+3);
    }
    ctx.setLineDash([]);

    /* ── already classified query points ── */
    if (step && step.classified.length) {
        step.classified.forEach(c => {
            const cx = rx(c.pt[0]), cy = ry(c.pt[1]);
            ctx.beginPath(); ctx.arc(cx, cy, 9, 0, 2*Math.PI);
            ctx.fillStyle = CLASS_COL[c.pred];
            ctx.strokeStyle = c.pred === c.trueLabel ? '#1e293b' : '#f59e0b';
            ctx.lineWidth = c.pred === c.trueLabel ? 1.5 : 2.5;
            ctx.fill(); ctx.stroke();
            /* star symbol */
            drawStar(ctx, cx, cy, 5, 7, 3.5);
            ctx.fillStyle = '#fff'; ctx.fill();
        });
    }

    /* ── training points ── */
    TRAIN_X.forEach((x, i) => {
        const cx = rx(x[0]), cy = ry(x[1]);
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2*Math.PI);
        ctx.fillStyle   = CLASS_COL[TRAIN_Y[i]];
        ctx.strokeStyle = CLASS_DARK[TRAIN_Y[i]];
        ctx.lineWidth   = 1;
        ctx.fill(); ctx.stroke();
    });

    /* ── distance lines — past points (faded) ── */
    if (step && step.queryPt && step.shownDists && step.shownDists.length) {
        const qx = rx(step.queryPt[0]), qy = ry(step.queryPt[1]);
        step.shownDists.forEach((sd, si) => {
            const tpt = TRAIN_X[sd.idx];
            if (!tpt) return;
            const tx = rx(tpt[0]), ty = ry(tpt[1]);
            const isCurrent = (step.activePtIdx === sd.idx);
            const col = CLASS_COL[sd.label];

            /* line */
            ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty);
            ctx.strokeStyle = isCurrent ? col : col + '55';  /* faded past, vivid current */
            ctx.lineWidth   = isCurrent ? 1.6 : 0.7;
            ctx.setLineDash(isCurrent ? [] : [3,3]);
            ctx.stroke();
            ctx.setLineDash([]);

            /* distance label — only for current active point */
            if (isCurrent) {
                const mx = (qx + tx) / 2 + 6, my = (qy + ty) / 2 - 4;
                ctx.font = 'bold 8.5px Consolas, monospace';
                const dlbl = `d=${fmt(sd.d, 4)}`;
                const tw   = ctx.measureText(dlbl).width;
                ctx.fillStyle = 'rgba(255,255,255,0.88)';
                ctx.fillRect(mx - 2, my - 9, tw + 4, 12);
                ctx.fillStyle = CLASS_DARK[sd.label] || '#1e293b';
                ctx.fillText(dlbl, mx, my);
            }
        });
    }

    /* ── distance lines to k-NN (highlighted, from sort/select/vote steps) ── */
    if (step && step.queryPt && step.knn.length) {
        const qx = rx(step.queryPt[0]), qy = ry(step.queryPt[1]);

        /* radius circle — measured in PIXELS from query to each k-NN,
           then take the maximum so all k points are always inside */
        let rPx = 0;
        step.knn.forEach(n => {
            const tx = rx(TRAIN_X[n.idx][0]), ty = ry(TRAIN_X[n.idx][1]);
            const dPx = Math.sqrt((tx - qx)**2 + (ty - qy)**2);
            if (dPx > rPx) rPx = dPx;
        });
        rPx = rPx + 8;   /* small padding so points sit inside the ring */
        ctx.beginPath(); ctx.arc(qx, qy, rPx, 0, 2*Math.PI);
        ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([5,3]);
        ctx.stroke(); ctx.setLineDash([]);
        /* light fill */
        ctx.beginPath(); ctx.arc(qx, qy, rPx, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(245,158,11,0.07)'; ctx.fill();

        /* lines to k-NN */
        step.knn.forEach((n, ni) => {
            const tx = rx(TRAIN_X[n.idx][0]), ty = ry(TRAIN_X[n.idx][1]);
            ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty);
            ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.75;
            ctx.stroke(); ctx.globalAlpha = 1;

            /* highlight the neighbour */
            ctx.beginPath(); ctx.arc(tx, ty, 7, 0, 2*Math.PI);
            ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2;
            ctx.stroke();

            /* rank label */
            ctx.fillStyle = '#92400e'; ctx.font = 'bold 8px sans-serif';
            ctx.fillText(ni+1, tx+7, ty-5);
        });
    }

    /* ── current query point (star — NO prediction label) ── */
    if (step && step.queryPt) {
        const qx = rx(step.queryPt[0]), qy = ry(step.queryPt[1]);
        /* outer glow */
        ctx.beginPath(); ctx.arc(qx, qy, 11, 0, 2*Math.PI);
        ctx.fillStyle = '#f59e0b';
        ctx.globalAlpha = 0.25; ctx.fill(); ctx.globalAlpha = 1;
        /* star — always gold, no class colour or label */
        drawStar(ctx, qx, qy, 5, 9, 4.5);
        ctx.fillStyle   = '#f59e0b';
        ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1.5;
        ctx.fill(); ctx.stroke();
        /* query coordinate label — small, unobtrusive */
        ctx.fillStyle = '#475569'; ctx.font = '7.5px sans-serif';
        ctx.fillText(`(${fmt(step.queryPt[0],1)},${fmt(step.queryPt[1],1)})`, qx+11, qy+4);
    }

    /* ── legend: top-right ── */
    const legX = W - 90, legY = 14;
    ctx.fillStyle = 'rgba(255,255,255,0.88)';
    ctx.fillRect(legX-4, legY-10, 88, 62);
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 0.7;
    ctx.strokeRect(legX-4, legY-10, 88, 62);
    ctx.font = '8.5px sans-serif';
    [
        [CLASS_COL[0], '● Class 0 (train)'],
        [CLASS_COL[1], '● Class 1 (train)'],
        ['#f59e0b',    '★ Query point'],
        ['#f59e0b',    '- - k-NN radius'],
    ].forEach(([col, lbl], i) => {
        ctx.fillStyle = col; ctx.fillText(lbl, legX, legY + i*13);
    });
}

/* ── draw a 5-point star ── */
function drawStar(ctx, cx, cy, pts, outerR, innerR) {
    ctx.beginPath();
    for (let i = 0; i < pts * 2; i++) {
        const angle = (i * Math.PI) / pts - Math.PI / 2;
        const r = i % 2 === 0 ? outerR : innerR;
        i === 0 ? ctx.moveTo(cx + r*Math.cos(angle), cy + r*Math.sin(angle))
                : ctx.lineTo(cx + r*Math.cos(angle), cy + r*Math.sin(angle));
    }
    ctx.closePath();
}

/* ══════════════════════════════════════════════════════════════════
   NEIGHBOUR TABLE
   ══════════════════════════════════════════════════════════════════ */
function renderNbrTable(step) {
    const box = document.getElementById('nbr-table-wrap');
    const k   = parseInt(document.getElementById('kInput').value) || 5;

    /* ── no query point yet (init / done steps) ── */
    if (!step || !step.queryPt) {
        box.innerHTML =
            '<span style="color:#475569;font-size:9px;font-family:var(--mono);">Distances will fill here…</span>';
        return;
    }

    /* ── choose which distances to show ──
       During dist_one: shownDists (unsorted, fills row-by-row)
       After sort/select/vote: allDists (sorted) with k-nearest highlighted */
    let rows = [];
    let sorted = false;

    if (step.kind === 'dist_one' && step.shownDists && step.shownDists.length) {
        /* show distances in the order they were computed (unsorted) */
        rows = step.shownDists.map(sd => ({
            rank: null,
            idx:  sd.idx,
            d:    sd.d,
            label: sd.label,
            isKnn: false,
            isCurrent: sd.idx === step.activePtIdx
        }));
    } else if (step.allDists && step.allDists.length) {
        /* sorted — from sort/kneighbors/vote steps */
        sorted = true;
        const knnIdxSet = new Set((step.knn || []).map(n => n.idx));
        rows = step.allDists.map((sd, ri) => ({
            rank: ri + 1,
            idx:  sd.idx,
            d:    sd.d,
            label: sd.label,
            isKnn: knnIdxSet.has(sd.idx),
            isCurrent: false
        }));
    } else if (step.knn && step.knn.length) {
        /* only knn available (vote step) */
        sorted = true;
        rows = step.knn.map((n, ri) => ({
            rank: ri + 1,
            idx:  n.idx,
            d:    n.d,
            label: n.label,
            isKnn: true,
            isCurrent: false
        }));
    } else {
        box.innerHTML =
            '<span style="color:#475569;font-size:9px;font-family:var(--mono);">Distances will fill here…</span>';
        return;
    }

    const totalComputed = step.shownDists ? step.shownDists.length : rows.length;
    const headerNote = sorted
        ? `<span style="font-size:.6rem;color:#475569;">Sorted ascending — top ${k} highlighted</span>`
        : `<span style="font-size:.6rem;color:#475569;">Computing… ${totalComputed} / ${TRAIN_X.length} done</span>`;

    box.innerHTML =
        `<div class="d-flex justify-content-between align-items-center mb-1">${headerNote}</div>` +
        `<table class="table table-sm table-bordered mb-0" style="font-size:.63rem;">
           <thead><tr class="table-primary">
             <th>${sorted ? 'Rank' : '#'}</th>
             <th>Idx</th><th>x1</th><th>x2</th>
             <th>Distance</th><th>Class</th>
           </tr></thead>
           <tbody>` +
        rows.map((r) => {
            const col     = CLASS_COL[r.label] || '#64748b';
            const nnStyle = r.isKnn     ? 'background:#fef9c3;font-weight:700;' : '';
            const curStyle= r.isCurrent ? 'background:#d1fae5;outline:2px solid #059669;' : '';
            const rowStyle= nnStyle || curStyle;
            const rankCell= sorted
                ? `<td>${r.isKnn ? `<b>★${r.rank}</b>` : r.rank}</td>`
                : `<td>${r.isCurrent ? '→' : r.idx + 1}</td>`;
            return `<tr style="${rowStyle}">
                ${rankCell}
                <td>${r.idx}</td>
                <td>${fmt(TRAIN_X[r.idx]?.[0] ?? 0, 2)}</td>
                <td>${fmt(TRAIN_X[r.idx]?.[1] ?? 0, 2)}</td>
                <td><b>${fmt(r.d, 4)}</b></td>
                <td><span class="badge" style="background:${col};font-size:.6rem;">${r.label}</span></td>
            </tr>`;
        }).join('') +
        `</tbody></table>`;

    /* auto-scroll to the currently computed row */
    if (step.kind === 'dist_one') {
        const trs = box.querySelectorAll('tbody tr');
        if (trs.length) trs[trs.length - 1].scrollIntoView({ block: 'nearest' });
    }
}

/* ══════════════════════════════════════════════════════════════════
   METRICS
   ══════════════════════════════════════════════════════════════════ */
function renderMetrics() {
    const k    = parseInt(document.getElementById('kInput').value) || 5;
    const preds = TEST_X.map(x => knnPredict(x, TRAIN_X, TRAIN_Y, k).pred);
    const tp = preds.filter((p,i)=>p===1&&TEST_Y[i]===1).length;
    const tn = preds.filter((p,i)=>p===0&&TEST_Y[i]===0).length;
    const fp = preds.filter((p,i)=>p===1&&TEST_Y[i]===0).length;
    const fn = preds.filter((p,i)=>p===0&&TEST_Y[i]===1).length;
    const acc  = (tp+tn)/(tp+tn+fp+fn)||0;
    const prec = tp/(tp+fp)||0;
    const rec  = tp/(tp+fn)||0;
    const f1   = 2*prec*rec/(prec+rec)||0;
    const card = (l,v,c) =>
        `<div class="metric-card" style="border-color:${c};background:${c}15;">
           <span class="mc-label" style="color:${c};">${l}</span>
           <span class="mc-value" style="color:${c};">${(v*100).toFixed(2)}%</span>
         </div>`;
    document.getElementById('metrics').innerHTML =
        card('Accuracy',  acc,  '#3b82f6') +
        card('Precision', prec, '#7c3aed') +
        card('Recall',    rec,  '#0891b2') +
        card('F1 Score',  f1,   '#059669') +
        `<div class="metric-card" style="border-color:#64748b;background:#64748b15;">
           <span class="mc-label" style="color:#64748b;">TP/TN/FP/FN</span>
           <span class="mc-value" style="color:#64748b;font-size:12px;">${tp}/${tn}/${fp}/${fn}</span>
         </div>` +
        `<div class="metric-card" style="border-color:#d97706;background:#d9770615;">
           <span class="mc-label" style="color:#d97706;">k</span>
           <span class="mc-value" style="color:#d97706;">${k}</span>
         </div>`;
}

/* ══════════════════════════════════════════════════════════════════
   SHOW STEP
   ══════════════════════════════════════════════════════════════════ */
function show(i) {
    if (!STEPS.length) return;
    cursor = Math.max(0, Math.min(i, STEPS.length-1));
    const s = STEPS[cursor];

    highlight(s.lines);
    document.getElementById('step-counter').textContent = `Step ${cursor+1} / ${STEPS.length}`;
    document.getElementById('calc').innerHTML =
        `<span class="calc-title">${s.title}</span>` +
        `<span class="calc-body">${s.detail.replace(/</g,'&lt;')}</span>`;

    /* left panel current state */
    document.getElementById('queryDisplay').value =
        s.queryPt ? `(${fmt(s.queryPt[0],2)}, ${fmt(s.queryPt[1],2)})` : '—';
    document.getElementById('distDisplay').value  =
        s.knn.length ? fmt(s.knn[0].d, 4) : '—';
    document.getElementById('predDisplay').value  =
        s.pred !== null ? `Class ${s.pred}` : '—';

    drawCanvas(s);
    renderNbrTable(s);
}

/* ══════════════════════════════════════════════════════════════════
   TRAINING — called on Start
   ══════════════════════════════════════════════════════════════════ */
function runKNN() {
    if (!ALL_DATA.length) { alert('Load a dataset first.'); return; }
    const splitPct = parseInt(document.getElementById('splitInput').value) || 80;
    const splitIdx = Math.floor(ALL_DATA.length * splitPct / 100);

    TRAIN_X = ALL_DATA.slice(0, splitIdx).map(d => [d.x1, d.x2]);
    TRAIN_Y = ALL_DATA.slice(0, splitIdx).map(d => d.y);
    TEST_X  = ALL_DATA.slice(splitIdx).map(d => [d.x1, d.x2]);
    TEST_Y  = ALL_DATA.slice(splitIdx).map(d => d.y);

    if (!TRAIN_X.length || !TEST_X.length) { alert('Not enough data for the selected split.'); return; }

    buildSteps();
    renderMetrics();
    show(0);
}

/* ══════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════ */
document.getElementById('startBtn').onclick = runKNN;
document.getElementById('nextBtn').onclick  = () => { if (STEPS.length) show(cursor+1); };
document.getElementById('prevBtn').onclick  = () => { if (STEPS.length) show(cursor-1); };
document.getElementById('endBtn').onclick   = () => {
    if (!STEPS.length) runKNN(); else show(STEPS.length-1);
};

/* ══════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════ */
function loadData(arr) {
    if (!arr.length) { alert('Empty dataset'); return; }
    const cols = Object.keys(arr[0]);
    const x1c = cols[0], x2c = cols[1], yc = cols[cols.length-1];
    ALL_DATA = arr.map(r => ({
        x1: parseFloat(r[x1c]),
        x2: parseFloat(r[x2c]),
        y:  parseInt(r[yc])
    })).filter(d => !isNaN(d.x1) && !isNaN(d.x2) && !isNaN(d.y));

    if (!ALL_DATA.length) { alert('Could not parse dataset. Ensure columns are numeric.'); return; }

    const thead = document.getElementById('thead');
    const tbody = document.getElementById('tbody');
    thead.innerHTML = `<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
    tbody.innerHTML = ALL_DATA.map(d=>
        `<tr><td>${fmt(d.x1,2)}</td><td>${fmt(d.x2,2)}</td>
             <td><span class="badge" style="background:${CLASS_COL[d.y]}">${d.y}</span></td></tr>`
    ).join('');
    document.getElementById('table-box').classList.remove('hidden');

    STEPS = []; cursor = -1;
    TRAIN_X = []; TRAIN_Y = []; TEST_X = []; TEST_Y = [];
    document.getElementById('calc').innerHTML =
        `<span class="text-muted">Dataset loaded (${ALL_DATA.length} rows). Press <b>Start</b> to run KNN.</span>`;
    document.getElementById('step-counter').textContent = 'Step 0 / 0';
    document.getElementById('metrics').innerHTML = '';
    document.getElementById('nbr-table-wrap').innerHTML =
        '<span style="color:#475569;font-size:9px;font-family:var(--mono);">Neighbours will appear during classification…</span>';
    drawCanvas(null);
}

/* ── preset dataset — two 2D clusters ── */
const PRESET = [
    /* class 0 — spread across lower-left quadrant */
    {x1: 1.0, x2: 2.0, y:0}, {x1: 2.5, x2: 0.8, y:0}, {x1: 0.5, x2: 5.5, y:0},
    {x1: 3.8, x2: 1.5, y:0}, {x1: 1.5, x2: 7.0, y:0}, {x1: 4.5, x2: 0.5, y:0},
    {x1: 0.8, x2: 3.8, y:0}, {x1: 3.0, x2: 4.5, y:0}, {x1: 5.5, x2: 2.0, y:0},
    {x1: 2.0, x2: 6.5, y:0}, {x1: 6.0, x2: 1.0, y:0}, {x1: 4.0, x2: 6.0, y:0},
    /* class 1 — spread across upper-right quadrant */
    {x1: 6.5, x2: 8.5, y:1}, {x1: 8.0, x2: 5.0, y:1}, {x1: 9.0, x2: 7.5, y:1},
    {x1: 7.5, x2: 3.5, y:1}, {x1: 5.0, x2: 9.0, y:1}, {x1: 8.5, x2: 9.0, y:1},
    {x1: 9.5, x2: 4.5, y:1}, {x1: 6.0, x2: 6.5, y:1}, {x1: 7.0, x2: 8.0, y:1},
    {x1: 8.8, x2: 6.5, y:1}, {x1: 5.5, x2: 7.5, y:1}, {x1: 9.2, x2: 3.0, y:1},
    /* boundary region — intentionally mixed */
    {x1: 5.0, x2: 4.5, y:0}, {x1: 4.8, x2: 5.5, y:1}, {x1: 5.5, x2: 4.0, y:1},
    {x1: 4.2, x2: 4.8, y:0}, {x1: 5.8, x2: 5.2, y:1}, {x1: 4.5, x2: 3.5, y:0},
    {x1: 6.2, x2: 4.8, y:1}, {x1: 3.5, x2: 5.0, y:0},
];

const setActive = id => ['presetBtn','randomBtn','customBtn']
    .forEach(b => document.getElementById(b).classList.toggle('btn-active', b===id));

document.getElementById('presetBtn').onclick = () => {
    setActive('presetBtn');
    document.getElementById('upload-box').classList.add('hidden');
    loadData(JSON.parse(JSON.stringify(PRESET)));
};

document.getElementById('randomBtn').onclick = () => {
    setActive('randomBtn');
    document.getElementById('upload-box').classList.add('hidden');
    const data = [];
    for (let i = 0; i < 30; i++) {
        const cls = i < 15 ? 0 : 1;
        const cx  = cls === 0 ? 2 : 7, cy = cls === 0 ? 2 : 7;
        data.push({
            x1: +(cx + (Math.random()-0.5)*3).toFixed(2),
            x2: +(cy + (Math.random()-0.5)*3).toFixed(2),
            y: cls
        });
    }
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
            const wb   = XLSX.read(new Uint8Array(ev.target.result), {type:'array'});
            const json = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], {defval:''});
            loadData(json);
            document.getElementById('upload-box').classList.add('hidden');
        } catch(err) { alert('Could not read file: ' + err.message); }
    };
    fr.readAsArrayBuffer(chosenFile);
};


/* ── click on canvas → set query point and re-run ── */
(function attachCanvasClick() {
    const canvas = document.getElementById('knn-canvas');
    canvas.style.cursor = 'crosshair';
    canvas.addEventListener('click', function(e) {
        if (!TRAIN_X.length) return;

        /* pixel → data coordinates */
        const rect = canvas.getBoundingClientRect();
        const mx   = (e.clientX - rect.left)  * (canvas.width  / rect.width);
        const my   = (e.clientY - rect.top)   * (canvas.height / rect.height);
        const pad  = 36;
        const W    = canvas.width, H = canvas.height;
        const allX1 = ALL_DATA.map(d=>d.x1), allX2 = ALL_DATA.map(d=>d.x2);
        const x1Min = Math.min(...allX1), x1Max = Math.max(...allX1);
        const x2Min = Math.min(...allX2), x2Max = Math.max(...allX2);

        const qx1 = x1Min + (mx - pad) / (W - 2*pad) * (x1Max - x1Min);
        const qx2 = x2Min + (1 - (my - 10) / (H - pad - 10)) * (x2Max - x2Min);

        /* clamp to data range */
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const cq1 = clamp(qx1, x1Min, x1Max);
        const cq2 = clamp(qx2, x2Min, x2Max);

        /* inject a custom single-query step trace */
        runSingleQuery([cq1, cq2]);
    });
})();

/* ── run KNN for one manually-clicked query point ── */
function runSingleQuery(query) {
    if (!TRAIN_X.length) return;
    const k        = parseInt(document.getElementById('kInput').value) || 5;
    const distName = document.getElementById('distInput').value;
    const dist     = distFn();
    const { pred, knn, allDists } = knnPredict(query, TRAIN_X, TRAIN_Y, k);

    /* build a small focused step trace for this single query */
    STEPS = [];
    const shownDistsSoFar = [];

    /* query step */
    STEPS.push({
        kind: 'query', lines: LINE.query,
        title: 'Manual query point selected',
        detail: `Query (clicked) : (${fmt(query[0],3)}, ${fmt(query[1],3)})\nk               : ${k}\nMetric          : ${distName}\n\nComputing distances…`,
        queryPt: query, knn: [], allDists: [], pred: null,
        activePtIdx: -1, shownDists: [], classified: [], queryIdxs: []
    });

    /* one dist_one step per training point */
    TRAIN_X.forEach((tpt, ti) => {
        const d = dist(query, tpt);
        const lbl = TRAIN_Y[ti];
        let formula;
        if (distName === 'euclidean') {
            formula =
                `d = √( (${fmt(tpt[0],2)} − ${fmt(query[0],2)})²  ` +
                `     + (${fmt(tpt[1],2)} − ${fmt(query[1],2)})² )\n` +
                `  = √( ${fmt((tpt[0]-query[0])**2,4)} + ${fmt((tpt[1]-query[1])**2,4)} )\n` +
                `  = ${fmt(d,6)}`;
        } else {
            formula =
                `d = |${fmt(tpt[0],2)} − ${fmt(query[0],2)}| + |${fmt(tpt[1],2)} − ${fmt(query[1],2)}|\n` +
                `  = ${fmt(Math.abs(tpt[0]-query[0]),4)} + ${fmt(Math.abs(tpt[1]-query[1]),4)}\n` +
                `  = ${fmt(d,6)}`;
        }
        shownDistsSoFar.push({ idx: ti, d, label: lbl });
        STEPS.push({
            kind: 'dist_one', lines: LINE.distance,
            title: `Distance to point [${ti+1}/${TRAIN_X.length}]  class=${lbl}`,
            detail:
                `Point [${ti}]: (${fmt(tpt[0],2)}, ${fmt(tpt[1],2)})  class=${lbl}\n` +
                `Query       : (${fmt(query[0],2)}, ${fmt(query[1],2)})\n\n` +
                formula +
                `\n\nComputed: ${shownDistsSoFar.length} / ${TRAIN_X.length}`,
            queryPt: query, knn: [], allDists: [],
            activePtIdx: ti, shownDists: [...shownDistsSoFar],
            pred: null, classified: [], queryIdxs: []
        });
    });

    /* sort */
    STEPS.push({
        kind: 'sort', lines: LINE.sort,
        title: 'Sort distances ascending',
        detail: `Sorted (top ${k+2}):\n\n` +
            allDists.slice(0, k+2).map((d,i)=>
                `  ${i < k ? '►' : ' '} [${d.idx}] d=${fmt(d.d,4)}  y=${d.label}` +
                (i===k-1 ? `  ← k=${k}` : '')).join('\n'),
        queryPt: query, knn: [], allDists,
        activePtIdx: -1, shownDists: [...shownDistsSoFar],
        pred: null, classified: [], queryIdxs: []
    });

    /* kneighbors */
    STEPS.push({
        kind: 'kneighbors', lines: LINE.kneighbors,
        title: `Select k=${k} nearest`,
        detail: `k nearest:\n` +
            knn.map((n,i)=>
                `  ${i+1}. [${n.idx}] (${fmt(TRAIN_X[n.idx][0],2)},${fmt(TRAIN_X[n.idx][1],2)}) d=${fmt(n.d,4)} class=${n.label}`
            ).join('\n'),
        queryPt: query, knn, allDists,
        activePtIdx: -1, shownDists: [...shownDistsSoFar],
        pred: null, classified: [], queryIdxs: []
    });

    /* vote */
    const votes = {};
    knn.forEach(n => { votes[n.label] = (votes[n.label]||0)+1; });
    STEPS.push({
        kind: 'vote', lines: LINE.vote,
        title: `Majority vote → class ${pred}`,
        detail: `Votes:\n` +
            Object.entries(votes).map(([cls,cnt])=>
                `  Class ${cls}: ${cnt} ${'█'.repeat(cnt)}`).join('\n') +
            `\n\nPredicted: Class ${pred}`,
        queryPt: query, knn, allDists,
        activePtIdx: -1, shownDists: [...shownDistsSoFar],
        pred, classified: [], queryIdxs: []
    });

    cursor = -1;
    document.getElementById('step-counter').textContent = `Step 0 / ${STEPS.length}`;
    document.getElementById('calc').innerHTML =
        `<span style="color:#34d399;font-weight:700;font-family:var(--mono);">Query set by click → Press Start / Next to step through.</span>`;
    renderMetrics();
    show(0);
}

/* auto-load preset on open */
document.getElementById('presetBtn').click();