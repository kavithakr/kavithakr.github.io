/* ══════════════════════════════════════════════════════════════════
   K-Means Clustering Visualization — kmeans_script.js
   Fully dynamic: works on any x1,x2 dataset.
   ══════════════════════════════════════════════════════════════════ */

/* ── Python code displayed in the editor ── */
const CODE = `import numpy as np

# ── 1. Initialise k centroids ────────────────────────
def init_centroids(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx].copy()

# ── 2. Assign each point to nearest centroid ─────────
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        dists = [euclidean(point, c)
                 for c in centroids]
        clusters.append(np.argmin(dists))
    return np.array(clusters)

# ── 3. Euclidean distance ─────────────────────────────
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# ── 4. Recompute centroids as cluster means ───────────
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        members = X[clusters == i]
        new_centroids.append(members.mean(axis=0))
    return np.array(new_centroids)

# ── 5. Compute WCSS (within-cluster sum of squares) ──
def compute_wcss(X, clusters, centroids):
    wcss = 0
    for i, point in enumerate(X):
        c = centroids[clusters[i]]
        wcss += euclidean(point, c) ** 2
    return wcss

# ── 6. K-Means main loop ──────────────────────────────
def kmeans(X, k, max_iter=10):
    centroids = init_centroids(X, k)
    for iteration in range(max_iter):
        clusters  = assign_clusters(X, centroids)
        new_cents = update_centroids(X, clusters, k)
        wcss      = compute_wcss(X, clusters, new_cents)
        if np.allclose(centroids, new_cents):
            break          # converged
        centroids = new_cents
    return clusters, centroids, wcss`;

/* line ranges aligned to step kinds */
const LINE = {
    init:    [1, 2],
    init_c:  [4, 5, 6],
    assign:  [8, 9, 10, 11, 12, 13, 14],
    euclid:  [16, 17, 18],
    update:  [20, 21, 22, 23, 24, 25],
    wcss:    [27, 28, 29, 30, 31, 32],
    loop:    [34, 35, 36, 37, 38, 39, 40, 41, 42],
    done:    [43],
};

/* ── render code table ── */
(function renderCode() {
    const lines = CODE.split('\n');
    let html = '';
    lines.forEach((ln, i) => {
        const safe = ln.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') || ' ';
        html += `<tr id="row-${i}"><td class="ln">${i+1}</td><td class="lc">${safe}</td></tr>`;
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

/* ══════════════════════════════════════════════════════════════════
   COLOURS
   ══════════════════════════════════════════════════════════════════ */
const PALETTE = [
    '#2563eb', '#dc2626', '#16a34a',
    '#d97706', '#7c3aed', '#0891b2',
    '#db2777', '#65a30d',
];
const PALETTE_L = [
    '#bfdbfe', '#fee2e2', '#dcfce7',
    '#fef3c7', '#ede9fe', '#e0f2fe',
    '#fce7f3', '#f7fee7',
];
const fmt = (v, d=4) => Number(v).toFixed(d);

/* ══════════════════════════════════════════════════════════════════
   CORE K-MEANS ALGORITHM
   ══════════════════════════════════════════════════════════════════ */
function euclidean(a, b) {
    return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2);
}

function assignClusters(X, centroids) {
    return X.map(pt => {
        let best = 0, bestD = Infinity;
        centroids.forEach((c, i) => {
            const d = euclidean(pt, c);
            if (d < bestD) { bestD = d; best = i; }
        });
        return best;
    });
}

function updateCentroids(X, clusters, k) {
    return Array.from({length: k}, (_, i) => {
        const members = X.filter((_, j) => clusters[j] === i);
        if (!members.length) return X[Math.floor(Math.random()*X.length)].slice();
        return [
            members.reduce((s,p) => s+p[0], 0) / members.length,
            members.reduce((s,p) => s+p[1], 0) / members.length,
        ];
    });
}

function computeWCSS(X, clusters, centroids) {
    return X.reduce((s, pt, i) => {
        const d = euclidean(pt, centroids[clusters[i]]);
        return s + d*d;
    }, 0);
}

function centroidsEqual(a, b, tol=1e-6) {
    return a.every((c, i) => Math.abs(c[0]-b[i][0]) < tol && Math.abs(c[1]-b[i][1]) < tol);
}

/* ══════════════════════════════════════════════════════════════════
   STEP TRACE
   ══════════════════════════════════════════════════════════════════ */
let DATA = [], STEPS = [], cursor = -1;
let manualCentroids = [], manualK = 3;

function buildSteps(initCentroids) {
    STEPS = [];
    const k       = parseInt(document.getElementById('kInput').value) || 3;
    const maxIter = parseInt(document.getElementById('maxIterInput').value) || 10;

    /* ── step 0: init ── */
    STEPS.push({
        kind: 'init', lines: LINE.init,
        title: 'Import numpy — initialise K-Means',
        centroids: null, clusters: null, wcss: null,
        iteration: 0, converged: false,
        detail:
            `import numpy as np\n\n` +
            `Dataset points : ${DATA.length}\n` +
            `k              : ${k}\n` +
            `Max iterations : ${maxIter}\n` +
            `Init method    : ${initCentroids ? 'manual (clicked)' : 'random'}`
    });

    /* ── step 1: initialise centroids ── */
    const cents0 = initCentroids
        ? initCentroids.map(c => c.slice())
        : (() => {
            const idx = [];
            while (idx.length < k) {
                const r = Math.floor(Math.random() * DATA.length);
                if (!idx.includes(r)) idx.push(r);
            }
            return idx.map(i => DATA[i].slice());
          })();

    STEPS.push({
        kind: 'init_c', lines: LINE.init_c,
        title: `Initialise ${k} centroids`,
        centroids: cents0.map(c=>c.slice()), clusters: null, wcss: null,
        iteration: 0, converged: false,
        detail:
            `Randomly selected ${k} starting centroids:\n\n` +
            cents0.map((c,i)=>
                `  C${i+1} = (${fmt(c[0],3)}, ${fmt(c[1],3)})`
            ).join('\n')
    });

    /* ── main loop ── */
    let cents = cents0.map(c => c.slice());
    let converged = false;
    const wcssHistory = [];

    for (let iter = 0; iter < maxIter; iter++) {

        /* ── assign step (one per point for first iter, grouped after) ── */
        const clusters = assignClusters(DATA, cents);

        if (iter === 0) {
            /* show assignment for each point individually in iteration 1 */
            /* Build assignedSoFar incrementally — only colour a point AFTER
           its distance has been computed and cluster decided.
           Points not yet processed stay as -1 (grey). */
        const assignedSoFar = new Array(DATA.length).fill(-1);
        DATA.forEach((pt, pi) => {
                const dists = cents.map((c,ci) => ({ ci, d: euclidean(pt, c) }));
                dists.sort((a,b)=>a.d-b.d);
                /* colour this point NOW that its cluster is decided */
                assignedSoFar[pi] = clusters[pi];
                /* restore original order (by ci) for canvas labelling */
                const distsOrdered = cents.map((c,ci) => ({ ci, d: euclidean(pt, c) }));
                STEPS.push({
                    kind: 'assign', lines: LINE.assign,
                    title: `Iteration 1 — Assign point [${pi+1}/${DATA.length}]`,
                    centroids: cents.map(c=>c.slice()),
                    clusters: assignedSoFar.slice(),
                    highlightPt: pi,
                    /* store distances so canvas can label each edge */
                    ptDists: distsOrdered,
                    wcss: null, iteration: iter+1, converged: false,
                    detail:
                        `Point [${pi}] = (${fmt(pt[0],3)}, ${fmt(pt[1],3)})\n\n` +
                        dists.map(({ci,d})=>
                            `  dist to C${ci+1} = ${fmt(d,4)}` +
                            (ci === clusters[pi] ? '  ← nearest' : '')
                        ).join('\n') +
                        `\n\nAssigned to Cluster ${clusters[pi]+1}\n` +
                        `Points coloured: ${pi+1} / ${DATA.length}`
                });
            });
        } else {
            /* subsequent iterations: SAME granular per-point steps as iteration 1 */
            const assignedSoFar2 = new Array(DATA.length).fill(-1);
            DATA.forEach((pt, pi) => {
                const distsOrdered2 = cents.map((c, ci) => ({ ci, d: euclidean(pt, c) }));
                const distsSorted2  = distsOrdered2.slice().sort((a,b) => a.d - b.d);
                assignedSoFar2[pi]  = clusters[pi];
                STEPS.push({
                    kind: 'assign', lines: LINE.assign,
                    title: `Iteration ${iter+1} — Assign point [${pi+1}/${DATA.length}]`,
                    centroids: cents.map(c=>c.slice()),
                    clusters: assignedSoFar2.slice(),
                    highlightPt: pi,
                    ptDists: distsOrdered2,
                    wcss: null, iteration: iter+1, converged: false,
                    detail:
                        `Point [${pi}] = (${fmt(pt[0],3)}, ${fmt(pt[1],3)})\n\n` +
                        distsSorted2.map(({ci,d})=>
                            `  dist to C${ci+1} = ${fmt(d,4)}` +
                            (ci === clusters[pi] ? '  ← nearest' : '')
                        ).join('\n') +
                        `\n\nAssigned to Cluster ${clusters[pi]+1}\n` +
                        `Points coloured: ${pi+1} / ${DATA.length}`
                });
            });
        }

        /* ── update centroids ── */
        const newCents = updateCentroids(DATA, clusters, k);
        const wcss     = computeWCSS(DATA, clusters, newCents);
        wcssHistory.push(wcss);
        converged = centroidsEqual(cents, newCents);

        STEPS.push({
            kind: 'update', lines: LINE.update,
            title: `Iteration ${iter+1} — Update centroids`,
            centroids: newCents.map(c=>c.slice()),
            clusters: clusters.slice(),
            highlightPt: -1,
            wcss, iteration: iter+1, converged,
            detail:
                `New centroid positions:\n\n` +
                newCents.map((c,i)=>{
                    const old = cents[i];
                    const move = euclidean(old, c);
                    return `  C${i+1}: (${fmt(old[0],3)},${fmt(old[1],3)}) → (${fmt(c[0],3)},${fmt(c[1],3)})  Δ=${fmt(move,4)}`;
                }).join('\n')
        });

        /* ── wcss step ── */
        STEPS.push({
            kind: 'wcss', lines: LINE.wcss,
            title: `Iteration ${iter+1} — Compute WCSS`,
            centroids: newCents.map(c=>c.slice()),
            clusters: clusters.slice(),
            highlightPt: -1,
            wcss, wcssHistory: [...wcssHistory],
            iteration: iter+1, converged,
            detail:
                `WCSS = Σ ||xᵢ − centroid(xᵢ)||²\n\n` +
                `Iteration ${iter+1} WCSS = ${fmt(wcss, 4)}\n` +
                (wcssHistory.length > 1
                    ? `Previous WCSS     = ${fmt(wcssHistory[wcssHistory.length-2], 4)}\n` +
                      `Change            = ${fmt(wcss - wcssHistory[wcssHistory.length-2], 4)}`
                    : '') +
                (converged ? '\n\n✓ Centroids did not move — CONVERGED!' : '')
        });

        cents = newCents;
        if (converged) break;
    }

    /* ── done step ── */
    const finalClusters = assignClusters(DATA, cents);
    const finalWCSS     = computeWCSS(DATA, finalClusters, cents);
    STEPS.push({
        kind: 'done', lines: LINE.done,
        title: 'K-Means complete',
        centroids: cents.map(c=>c.slice()),
        clusters: finalClusters.slice(),
        highlightPt: -1,
        wcss: finalWCSS, wcssHistory,
        iteration: wcssHistory.length, converged,
        detail:
            `K-Means finished.\n\n` +
            `Iterations run : ${wcssHistory.length}\n` +
            `Final WCSS     : ${fmt(finalWCSS, 4)}\n` +
            `Converged      : ${converged ? 'Yes ✓' : 'No (max iter reached)'}\n\n` +
            `Final centroids:\n` +
            cents.map((c,i)=>`  C${i+1} = (${fmt(c[0],3)}, ${fmt(c[1],3)})`).join('\n') +
            `\n\nCluster sizes:\n` +
            Array.from({length: parseInt(document.getElementById('kInput').value)||3}, (_,ci)=>{
                const n = finalClusters.filter(c=>c===ci).length;
                return `  Cluster ${ci+1}: ${n} points`;
            }).join('\n')
    });
}

/* ══════════════════════════════════════════════════════════════════
   CANVAS — scatter plot
   ══════════════════════════════════════════════════════════════════ */
function drawCanvas(step) {
    const canvas = document.getElementById('kmeans-canvas');
    const W = canvas.offsetWidth || 480;
    const H = canvas.offsetHeight || 300;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    if (!DATA.length) return;

    const pad = 36;
    const x1s = DATA.map(d=>d[0]), x2s = DATA.map(d=>d[1]);
    const x1Min = Math.min(...x1s)-1, x1Max = Math.max(...x1s)+1;
    const x2Min = Math.min(...x2s)-1, x2Max = Math.max(...x2s)+1;
    const px = x => pad + (x-x1Min)/(x1Max-x1Min)*(W-2*pad);
    const py = y => H-pad - (y-x2Min)/(x2Max-x2Min)*(H-2*pad);

    /* axes */
    ctx.strokeStyle='#94a3b8'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad,8); ctx.lineTo(pad,H-pad); ctx.lineTo(W-8,H-pad); ctx.stroke();
    ctx.fillStyle='#64748b'; ctx.font='9px sans-serif';
    ctx.fillText('x1',W-14,H-pad+2);
    ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2); ctx.fillText('x2',0,0); ctx.restore();

    /* grid */
    ctx.setLineDash([2,3]); ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=0.6;
    for (let t=0;t<=4;t++){
        const gx=pad+t*(W-2*pad)/4, gy=(H-pad)-t*(H-2*pad)/4;
        ctx.beginPath(); ctx.moveTo(gx,8); ctx.lineTo(gx,H-pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-8,gy); ctx.stroke();
        ctx.fillStyle='#94a3b8'; ctx.font='8px sans-serif';
        ctx.fillText((x1Min+t*(x1Max-x1Min)/4).toFixed(1),gx-8,H-pad+10);
        ctx.fillText((x2Min+t*(x2Max-x2Min)/4).toFixed(1),2,gy+3);
    }
    ctx.setLineDash([]);

    /* data points:
       - ci === -1  → not yet assigned this iteration → grey
       - ci >= 0    → assigned → cluster colour
       - isHL       → currently being processed → full vivid colour + larger */
    DATA.forEach((pt, i) => {
        const ci   = (step && step.clusters) ? step.clusters[i] : -1;
        const isHL = step && step.highlightPt === i;
        let fillC, strokeC, radius;
        if (isHL) {
            /* point currently being distance-computed — vivid, large */
            fillC   = ci >= 0 ? PALETTE[ci % PALETTE.length] : '#f59e0b';
            strokeC = '#1e293b';
            radius  = 8;
        } else if (ci >= 0) {
            /* already assigned — light cluster colour */
            fillC   = PALETTE_L[ci % PALETTE_L.length];
            strokeC = PALETTE[ci % PALETTE.length];
            radius  = 5;
        } else {
            /* not yet processed this iteration — grey */
            fillC   = '#e2e8f0';
            strokeC = '#94a3b8';
            radius  = 5;
        }
        ctx.beginPath(); ctx.arc(px(pt[0]), py(pt[1]), radius, 0, 2*Math.PI);
        ctx.fillStyle   = fillC;
        ctx.strokeStyle = strokeC;
        ctx.lineWidth   = isHL ? 2.5 : 1;
        ctx.fill(); ctx.stroke();
    });

    /* centroid-to-point lines + distance labels for highlighted point */
    if (step && step.highlightPt >= 0 && step.centroids) {
        const pt = DATA[step.highlightPt];
        const qx = px(pt[0]), qy = py(pt[1]);
        step.centroids.forEach((c, ci) => {
            const tx  = px(c[0]), ty  = py(c[1]);
            const col = PALETTE[ci % PALETTE.length];
            /* line */
            ctx.beginPath();
            ctx.moveTo(qx, qy); ctx.lineTo(tx, ty);
            ctx.strokeStyle = col + '99';
            ctx.lineWidth   = 1.3;
            ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([]);
            /* perpendicular offset for label */
            if (step.ptDists) {
                const d   = step.ptDists[ci].d;
                const mx2 = (qx + tx) / 2;
                const my2 = (qy + ty) / 2;
                const dx  = tx - qx, dy = ty - qy;
                const len = Math.sqrt(dx*dx + dy*dy) || 1;
                const ox  = (-dy / len) * 12;
                const oy  = ( dx / len) * 12;
                const lbl = `d=${d.toFixed(3)}`;
                ctx.font  = 'bold 8.5px Consolas, monospace';
                const tw  = ctx.measureText(lbl).width;
                /* white backing box */
                ctx.fillStyle = 'rgba(255,255,255,0.88)';
                ctx.fillRect(mx2 + ox - tw/2 - 2, my2 + oy - 9, tw + 4, 12);
                /* coloured text */
                ctx.fillStyle = col;
                ctx.textAlign = 'center';
                ctx.fillText(lbl, mx2 + ox, my2 + oy);
                ctx.textAlign = 'left';
            }
        });
    }

    /* centroids — large star markers */
    if (step && step.centroids) {
        step.centroids.forEach((c, ci) => {
            const col = PALETTE[ci%PALETTE.length];
            /* glow */
            ctx.beginPath(); ctx.arc(px(c[0]),py(c[1]),14,0,2*Math.PI);
            ctx.fillStyle=col+'33'; ctx.fill();
            /* cross */
            ctx.strokeStyle=col; ctx.lineWidth=2.5;
            ctx.beginPath(); ctx.moveTo(px(c[0])-10,py(c[1])); ctx.lineTo(px(c[0])+10,py(c[1])); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(px(c[0]),py(c[1])-10); ctx.lineTo(px(c[0]),py(c[1])+10); ctx.stroke();
            /* circle */
            ctx.beginPath(); ctx.arc(px(c[0]),py(c[1]),6,0,2*Math.PI);
            ctx.fillStyle=col; ctx.strokeStyle='#fff'; ctx.lineWidth=2;
            ctx.fill(); ctx.stroke();
            /* label */
            ctx.fillStyle='#1e293b'; ctx.font='bold 9px sans-serif';
            ctx.fillText(`C${ci+1}`,px(c[0])+10,py(c[1])-8);
        });
    }

    /* manual centroid placeholders */
    if (manualCentroids.length && (!step || !step.centroids)) {
        manualCentroids.forEach((c, ci) => {
            ctx.beginPath(); ctx.arc(px(c[0]),py(c[1]),8,0,2*Math.PI);
            ctx.fillStyle=PALETTE[ci%PALETTE.length]+'66';
            ctx.strokeStyle=PALETTE[ci%PALETTE.length]; ctx.lineWidth=2;
            ctx.fill(); ctx.stroke();
            ctx.fillStyle='#1e293b'; ctx.font='bold 9px sans-serif';
            ctx.fillText(`C${ci+1}`,px(c[0])+10,py(c[1])-8);
        });
    }
}

/* ── WCSS chart ── */
function drawWCSSChart(step) {
    const canvas = document.getElementById('wcss-canvas');
    const W = canvas.offsetWidth || 200;
    const H = canvas.offsetHeight || 100;
    canvas.width=W; canvas.height=H;
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);

    const hist = (step && step.wcssHistory) ? step.wcssHistory : [];
    if (hist.length < 2) {
        ctx.fillStyle='#94a3b8'; ctx.font='9px sans-serif';
        ctx.fillText('WCSS chart appears after 1st iteration',8,H/2);
        return;
    }
    const pad=28, maxV=Math.max(...hist), minV=Math.min(...hist)*0.95;
    const px2=i => pad+(i/(hist.length-1))*(W-pad-8);
    const py2=v => H-pad-(v-minV)/(maxV-minV+1e-9)*(H-2*pad);

    /* axes */
    ctx.strokeStyle='#94a3b8'; ctx.lineWidth=0.8;
    ctx.beginPath(); ctx.moveTo(pad,4); ctx.lineTo(pad,H-pad); ctx.lineTo(W-4,H-pad); ctx.stroke();
    ctx.fillStyle='#64748b'; ctx.font='7px sans-serif';
    ctx.fillText('Iter',W-18,H-pad+8); ctx.fillText('WCSS',2,10);

    /* line */
    ctx.beginPath(); ctx.strokeStyle='#2563eb'; ctx.lineWidth=1.5;
    hist.forEach((v,i)=>{ i===0?ctx.moveTo(px2(i),py2(v)):ctx.lineTo(px2(i),py2(v)); });
    ctx.stroke();

    /* dots */
    hist.forEach((v,i)=>{
        ctx.beginPath(); ctx.arc(px2(i),py2(v),3,0,2*Math.PI);
        ctx.fillStyle='#2563eb'; ctx.fill();
        ctx.fillStyle='#64748b'; ctx.font='7px sans-serif';
        ctx.fillText(i+1,px2(i)-3,H-pad+8);
    });
}

/* ── cluster sizes ── */
function renderClusterSizes(step) {
    const box = document.getElementById('cluster-sizes');
    if (!step || !step.clusters) { box.innerHTML='<span style="color:#475569;font-size:9px;font-family:var(--mono);">—</span>'; return; }
    const k = step.centroids ? step.centroids.length : 0;
    const sizes = Array.from({length:k},(_,i)=>step.clusters.filter(c=>c===i).length);
    box.innerHTML = sizes.map((n,i)=>
        `<div style="display:flex;align-items:center;margin-bottom:3px;font-size:10px;font-family:var(--mono);">
           <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                        background:${PALETTE[i%PALETTE.length]};margin-right:5px;flex-shrink:0;"></span>
           <span>C${i+1}: <b>${n}</b> pts</span>
         </div>`
    ).join('');
}

/* ── centroid info in left panel ── */
function renderCentroidInfo(step) {
    const box = document.getElementById('centroid-info');
    if (!step || !step.centroids) { box.innerHTML=''; return; }
    box.innerHTML = step.centroids.map((c,i)=>
        `<div style="display:flex;align-items:center;margin-bottom:3px;font-size:10px;font-family:var(--mono);">
           <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                        background:${PALETTE[i%PALETTE.length]};margin-right:5px;flex-shrink:0;"></span>
           <span style="color:var(--ink-2);">C${i+1}: <b style="color:var(--navy);">(${fmt(c[0],2)}, ${fmt(c[1],2)})</b></span>
         </div>`
    ).join('');
}

/* ── metrics ── */
function renderMetrics(step) {
    if (!step || !step.clusters || !step.centroids) {
        document.getElementById('metrics').innerHTML=''; return;
    }
    const wcss = step.wcss ?? 0;
    const k = step.centroids.length;
    const n = DATA.length;
    /* silhouette score approximation */
    let silSum = 0;
    DATA.forEach((pt, i) => {
        const ci = step.clusters[i];
        const members = DATA.filter((_,j)=>step.clusters[j]===ci);
        const a = members.length>1
            ? members.reduce((s,p)=>s+euclidean(pt,p),0)/(members.length-1) : 0;
        let b = Infinity;
        step.centroids.forEach((_,cj)=>{
            if (cj===ci) return;
            const other = DATA.filter((_,j)=>step.clusters[j]===cj);
            if (!other.length) return;
            const avgD = other.reduce((s,p)=>s+euclidean(pt,p),0)/other.length;
            if (avgD < b) b = avgD;
        });
        silSum += b===Infinity ? 0 : (b-a)/Math.max(a,b);
    });
    const sil = silSum/n;

    const card=(l,v,c)=>
        `<div class="metric-card" style="border-color:${c};background:${c}15;">
           <span class="mc-label" style="color:${c};">${l}</span>
           <span class="mc-value" style="color:${c};">${v}</span>
         </div>`;
    document.getElementById('metrics').innerHTML =
        card('WCSS',         fmt(wcss,2),         '#3b82f6') +
        card('Silhouette',   fmt(sil,3),           '#059669') +
        card('k',            k,                    '#7c3aed') +
        card('Points',       n,                    '#0891b2') +
        card('Iterations',   step.iteration||'—', '#d97706') +
        card('Converged',    step.converged?'Yes':'No', step.converged?'#16a34a':'#dc2626');
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

    document.getElementById('iterDisplay').value = s.iteration || '—';
    document.getElementById('wcssDisplay').value = s.wcss != null ? fmt(s.wcss,4) : '—';
    document.getElementById('convDisplay').value = s.converged ? 'Yes ✓' : (s.iteration?'No':'—');

    drawCanvas(s);
    drawWCSSChart(s);
    renderClusterSizes(s);
    renderCentroidInfo(s);
    renderMetrics(s);
}

/* ══════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════ */
function runKMeans() {
    if (!DATA.length) { alert('Load a dataset first.'); return; }
    const initMethod = document.getElementById('initInput').value;
    const k = parseInt(document.getElementById('kInput').value) || 3;
    let initCents = null;
    if (initMethod === 'manual') {
        if (manualCentroids.length < k) {
            alert(`Please click ${k} points on the plot to set centroids first.`); return;
        }
        initCents = manualCentroids.slice(0, k);
    }
    buildSteps(initCents);
    show(0);
}

document.getElementById('startBtn').onclick = runKMeans;
document.getElementById('nextBtn').onclick  = () => { if (STEPS.length) show(cursor+1); };
document.getElementById('prevBtn').onclick  = () => { if (STEPS.length) show(cursor-1); };
document.getElementById('endBtn').onclick   = () => {
    if (!STEPS.length) runKMeans(); else show(STEPS.length-1);
};

/* ══════════════════════════════════════════════════════════════════
   CANVAS CLICK — manual centroid placement
   ══════════════════════════════════════════════════════════════════ */
document.getElementById('kmeans-canvas').addEventListener('click', function(e) {
    if (document.getElementById('initInput').value !== 'manual') return;
    const k = parseInt(document.getElementById('kInput').value) || 3;
    if (manualCentroids.length >= k) manualCentroids = [];

    const rect = this.getBoundingClientRect();
    const W = this.offsetWidth || 480, H = this.offsetHeight || 300;
    const pad = 36;
    const x1s = DATA.map(d=>d[0]), x2s = DATA.map(d=>d[1]);
    const x1Min=Math.min(...x1s)-1, x1Max=Math.max(...x1s)+1;
    const x2Min=Math.min(...x2s)-1, x2Max=Math.max(...x2s)+1;
    const mx = (e.clientX-rect.left)*(this.width/rect.width);
    const my = (e.clientY-rect.top)*(this.height/rect.height);
    const cx = x1Min + (mx-pad)/(W-2*pad)*(x1Max-x1Min);
    const cy = x2Min + (1-(my-8)/(H-pad-8))*(x2Max-x2Min);
    manualCentroids.push([+cx.toFixed(3), +cy.toFixed(3)]);
    document.getElementById('calc').innerHTML =
        `<span style="color:#34d399;font-family:var(--mono);font-size:10px;">C${manualCentroids.length}/${k} placed at (${cx.toFixed(2)}, ${cy.toFixed(2)}).` +
        (manualCentroids.length < k ? ` Click ${k-manualCentroids.length} more.` : ' Press <b>Start</b>.') +
        `</span>`;
    drawCanvas(null);
});

/* ══════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════ */
function loadData(arr) {
    if (!arr.length) { alert('Empty dataset'); return; }
    const cols = Object.keys(arr[0]);
    DATA = arr.map(r => [parseFloat(r[cols[0]]), parseFloat(r[cols[1]])]).filter(d => !isNaN(d[0])&&!isNaN(d[1]));
    if (!DATA.length) { alert('Could not parse numeric x1,x2 columns.'); return; }

    const thead = document.getElementById('thead');
    const tbody = document.getElementById('tbody');
    thead.innerHTML = `<tr><th>${cols[0]}</th><th>${cols[1]}</th></tr>`;
    tbody.innerHTML = DATA.map(d=>`<tr><td>${fmt(d[0],2)}</td><td>${fmt(d[1],2)}</td></tr>`).join('');
    document.getElementById('table-box').classList.remove('hidden');

    manualCentroids = []; STEPS = []; cursor = -1;
    document.getElementById('calc').innerHTML =
        `<span style="color:#475569;">${DATA.length} pts loaded. Press <span style="color:#38bdf8;">Start</span>.</span>`;
    document.getElementById('step-counter').textContent = 'Step 0 / 0';
    document.getElementById('metrics').innerHTML = '';
    drawCanvas(null); drawWCSSChart(null);
}

/* ── preset ── */
const PRESET = [
    /* cluster A — lower left */
    [1.2,1.5],[1.8,2.2],[0.8,2.8],[2.5,1.2],[1.5,3.0],[3.0,1.8],[0.5,1.5],[2.0,0.8],[2.8,2.5],[1.0,3.5],
    /* cluster B — upper right */
    [7.5,8.0],[8.2,7.5],[6.8,8.5],[8.8,8.2],[7.0,7.0],[9.0,7.8],[8.5,9.0],[6.5,7.5],[9.2,8.8],[7.8,9.2],
    /* cluster C — lower right */
    [8.0,1.5],[9.0,2.5],[7.5,0.8],[9.5,1.8],[8.5,3.0],[7.0,2.0],[9.8,2.8],[8.2,0.5],[9.2,3.5],[7.8,1.2],
];

const setActive = id => ['presetBtn','randomBtn','customBtn']
    .forEach(b => document.getElementById(b).classList.toggle('btn-active', b===id));

document.getElementById('presetBtn').onclick = () => {
    setActive('presetBtn');
    document.getElementById('upload-box').classList.add('hidden');
    loadData(PRESET.map(p=>({x1:p[0],x2:p[1]})));
};

document.getElementById('randomBtn').onclick = () => {
    setActive('randomBtn');
    document.getElementById('upload-box').classList.add('hidden');
    const k = parseInt(document.getElementById('kInput').value)||3;
    const centres = Array.from({length:k},()=>[Math.random()*8+1, Math.random()*8+1]);
    const data=[];
    centres.forEach(c=>{
        for(let i=0;i<12;i++){
            data.push({
                x1:+(c[0]+(Math.random()-0.5)*3).toFixed(2),
                x2:+(c[1]+(Math.random()-0.5)*3).toFixed(2)
            });
        }
    });
    loadData(data);
};

document.getElementById('customBtn').onclick = () => {
    setActive('customBtn');
    document.getElementById('upload-box').classList.remove('hidden');
};

let chosenFile=null;
document.getElementById('fileInput').onchange=e=>{chosenFile=e.target.files[0];};
document.getElementById('submitFile').onclick=()=>{
    if(!chosenFile){alert('Choose a file first.');return;}
    const fr=new FileReader();
    fr.onload=ev=>{
        try{
            const wb=XLSX.read(new Uint8Array(ev.target.result),{type:'array'});
            const json=XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]],{defval:''});
            loadData(json);
            document.getElementById('upload-box').classList.add('hidden');
        }catch(err){alert('Could not read file: '+err.message);}
    };
    fr.readAsArrayBuffer(chosenFile);
};

/* auto-load preset */
document.getElementById('presetBtn').click();