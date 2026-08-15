/* ══════════════════════════════════════════════════════════════════
   Decision Tree Visualization — works on ANY dataset
   Everything (entropy, gain, tree, steps) is computed at runtime.
   ══════════════════════════════════════════════════════════════════ */

/* ---------- global state ---------- */
let COLS = [], ROWS = [], FEATURES = [], TARGET = '';
let FEATURE_TYPE = {};            // 'nominal' | 'numeric'
let STEPS = [];                   // execution trace
let cursor = -1;

/* ══════════════ 1. CORE ALGORITHM ══════════════ */

const log2 = x => Math.log(x) / Math.LN2;

function counts(rows) {
    const c = {};
    rows.forEach(r => { c[r[TARGET]] = (c[r[TARGET]] || 0) + 1; });
    return c;
}

function entropy(rows) {
    if (!rows.length) return 0;
    const c = counts(rows), n = rows.length;
    let h = 0;
    for (const k in c) { const p = c[k] / n; h -= p * log2(p); }
    return h;
}

function gini(rows) {
    if (!rows.length) return 0;
    const c = counts(rows), n = rows.length;
    let g = 1;
    for (const k in c) { const p = c[k] / n; g -= p * p; }
    return g;
}

const impurity = rows =>
    document.getElementById('criterion').value === 'gini' ? gini(rows) : entropy(rows);

const impurityName = () =>
    document.getElementById('criterion').value === 'gini' ? 'Gini' : 'H';

/* Detect whether a column is numeric */
function detectType(col) {
    const vals = ROWS.map(r => r[col]).filter(v => v !== '' && v != null);
    const numeric = vals.every(v => !isNaN(parseFloat(v)) && isFinite(v));
    const distinct = new Set(vals).size;
    // treat as numeric only if genuinely numeric AND reasonably many distinct values
    return (numeric && distinct > 4) ? 'numeric' : 'nominal';
}

/* Split rows on a feature. Returns [{label, rows, test}] */
function splitOn(rows, feat) {
    if (FEATURE_TYPE[feat] === 'numeric') {
        // find best threshold among midpoints
        const vals = [...new Set(rows.map(r => parseFloat(r[feat])))].sort((a, b) => a - b);
        let best = null;
        for (let i = 0; i < vals.length - 1; i++) {
            const t = (vals[i] + vals[i + 1]) / 2;
            const L = rows.filter(r => parseFloat(r[feat]) <= t);
            const R = rows.filter(r => parseFloat(r[feat]) > t);
            if (!L.length || !R.length) continue;
            const w = (L.length / rows.length) * impurity(L) + (R.length / rows.length) * impurity(R);
            if (!best || w < best.weighted) {
                best = {
                    weighted: w, threshold: t,
                    groups: [{ label: `<= ${t.toFixed(2)}`, rows: L },
                    { label: `> ${t.toFixed(2)}`, rows: R }]
                };
            }
        }
        return best ? best.groups : null;
    }
    // nominal
    const map = {};
    rows.forEach(r => { (map[r[feat]] = map[r[feat]] || []).push(r); });
    const keys = Object.keys(map).sort();
    if (keys.length < 2) return null;
    return keys.map(k => ({ label: String(k), rows: map[k] }));
}

function gainOf(rows, feat) {
    const groups = splitOn(rows, feat);
    if (!groups) return null;
    const parent = impurity(rows);
    let weighted = 0;
    groups.forEach(g => { weighted += (g.rows.length / rows.length) * impurity(g.rows); });
    return { gain: parent - weighted, groups, parent, weighted };
}

function majority(rows) {
    const c = counts(rows);
    return Object.keys(c).reduce((a, b) => (c[a] >= c[b] ? a : b));
}

/* ══════════════ 2. TREE BUILD + STEP TRACE ══════════════ */

let nodeSeq = 0;

function buildTree() {
    STEPS = [];
    nodeSeq = 0;
    const maxDepth = parseInt(document.getElementById('maxDepth').value);
    const minSamples = parseInt(document.getElementById('minSamples').value);

    STEPS.push({
        kind: 'init',
        lines: [1, 2, 3],
        rows: ROWS,
        title: 'Load dataset and initialise',
        detail:
            `Samples      : ${ROWS.length}\n` +
            `Features     : ${FEATURES.join(', ')}\n` +
            `Target       : ${TARGET}\n` +
            `Classes      : ${Object.entries(counts(ROWS)).map(([k, v]) => `${k}=${v}`).join(', ')}\n` +
            `Criterion    : ${impurityName() === 'Gini' ? 'Gini index' : 'Information gain (entropy)'}\n` +
            `Max depth    : ${maxDepth === 99 ? 'unlimited' : maxDepth}\n` +
            `Min samples  : ${minSamples}`
    });

    const root = grow(ROWS, new Set(), 0, null, null, maxDepth, minSamples);

    // final tree state for the last step
    STEPS.push({
        kind: 'done',
        lines: [55, 56, 57],
        rows: ROWS,
        node: null,
        title: 'Tree construction complete',
        detail: `Total nodes : ${nodeSeq}\nLeaves      : ${countLeaves(root)}\nDepth       : ${treeDepth(root)}`,
        tree: cloneTree(root)
    });

    return root;
}

function grow(rows, used, depth, parent, edgeLabel, maxDepth, minSamples) {
    const node = { id: nodeSeq++, depth, rows, children: [], parent, edgeLabel };
    if (parent) parent.children.push(node);

    const imp = impurity(rows);
    const cls = counts(rows);
    const distTxt = Object.entries(cls).map(([k, v]) => `${k}=${v}`).join(', ');

    /* --- step: arrive at node, compute impurity --- */
    STEPS.push({
        kind: 'impurity',
        lines: [6, 7, 8, 9, 10, 11, 12, 13],
        rows, node,
        title: `Compute ${impurityName()}(S) at ${depth === 0 ? 'root' : 'node'}${edgeLabel ? ` [${edgeLabel}]` : ''}`,
        detail: impurityDetail(rows),
        tree: cloneTree(rootRef(node))
    });

    /* --- stopping conditions --- */
    const pure = Object.keys(cls).length === 1;
    if (pure || depth >= maxDepth || rows.length < minSamples || used.size === FEATURES.length) {
        node.leaf = true;
        node.value = majority(rows);
        let why = pure ? `all ${rows.length} samples belong to class "${node.value}"`
            : depth >= maxDepth ? `max depth ${maxDepth} reached`
                : rows.length < minSamples ? `only ${rows.length} samples (< min ${minSamples})`
                    : 'all features already used';
        STEPS.push({
            kind: 'leaf',
            lines: [16, 17, 18, 19],
            rows, node,
            title: `Leaf node → ${TARGET} = ${node.value}`,
            detail: `Stopping condition met: ${why}.\n\nSamples      : ${rows.length}\n` +
                `Distribution : ${distTxt}\n${impurityName()}(S)        : ${imp.toFixed(4)}\n` +
                `Predicted    : ${node.value}`,
            tree: cloneTree(rootRef(node))
        });
        return node;
    }

    /* --- step per candidate feature --- */
    const results = {};
    const avail = FEATURES.filter(f => !used.has(f));
    avail.forEach(f => {
        const g = gainOf(rows, f);
        if (!g) return;
        results[f] = g;
        STEPS.push({
            kind: 'gain',
            lines: [22, 23, 24, 25, 26, 27, 28, 29],
            rows, node, feature: f,
            gains: { ...results },
            title: `${impurityName() === 'Gini' ? 'Gini gain' : 'Information gain'} for "${f}"`,
            detail: gainDetail(rows, f, g),
            tree: cloneTree(rootRef(node))
        });
    });

    const names = Object.keys(results);
    if (!names.length) {                       // no usable split
        node.leaf = true;
        node.value = majority(rows);
        STEPS.push({
            kind: 'leaf', lines: [16, 17, 18, 19], rows, node,
            title: `Leaf node → ${TARGET} = ${node.value}`,
            detail: 'No feature produces a valid split.',
            tree: cloneTree(rootRef(node))
        });
        return node;
    }

    const best = names.reduce((a, b) => (results[a].gain >= results[b].gain ? a : b));
    node.feature = best;
    node.gain = results[best].gain;
    node.groups = results[best].groups;

    /* --- step: select best feature --- */
    STEPS.push({
        kind: 'select',
        lines: [32, 33, 34, 35],
        rows, node, feature: best,
        gains: { ...results },
        title: `Select split feature → "${best}"`,
        detail:
            names.sort((a, b) => results[b].gain - results[a].gain)
                .map(f => `${(f === best ? '► ' : '  ')}${f.padEnd(16)} gain = ${results[f].gain.toFixed(4)}`)
                .join('\n') +
            `\n\nBranches: ${results[best].groups.map(g => g.label).join(' | ')}`,
        tree: cloneTree(rootRef(node))
    });

    /* --- recurse --- */
    const nextUsed = FEATURE_TYPE[best] === 'numeric' ? used : new Set([...used, best]);
    node.groups.forEach(g => {
        STEPS.push({
            kind: 'branch',
            lines: [38, 39, 40, 41, 42],
            rows: g.rows, node, feature: best, branch: g.label,
            title: `Branch "${best} = ${g.label}"`,
            detail: `Subset size  : ${g.rows.length}\n` +
                `Distribution : ${Object.entries(counts(g.rows)).map(([k, v]) => `${k}=${v}`).join(', ')}\n` +
                `${impurityName()}(subset)   : ${impurity(g.rows).toFixed(4)}\n\n` +
                `Recursing into build_tree() at depth ${depth + 1}…`,
            tree: cloneTree(rootRef(node))
        });
        grow(g.rows, nextUsed, depth + 1, node, g.label, maxDepth, minSamples);
    });

    return node;
}

/* ---------- detail text builders ---------- */
function impurityDetail(rows) {
    const c = counts(rows), n = rows.length;
    const isGini = impurityName() === 'Gini';
    // Name the actual classes in the formula rather than a generic p(c),
    // built dynamically so it holds for any dataset and any number of classes.
    const ks = Object.keys(c);
    let s = isGini
        ? `Gini(S) = 1 − (${ks.map(k => `p(${k})²`).join(' + ')})\n\n`
        : `H(S) = ${ks.map(k => `(− p(${k}) · log₂ p(${k}))`).join(' + ')}\n\n`;
    s += `Samples : ${n}\n`;
    for (const k in c) s += `  P(${k}) = ${c[k]}/${n} = ${(c[k] / n).toFixed(4)}\n`;
    s += '\n';
    if (isGini) {
        s += '1 − (' + Object.keys(c).map(k => `${(c[k] / n).toFixed(4)}²`).join(' + ') + ')\n';
    } else {
        s += Object.keys(c).map(k => {
            const p = c[k] / n;
            return `  (− p(${k}) · log₂ p(${k})) = −(${p.toFixed(4)} × ${log2(p).toFixed(4)}) = ${(-p * log2(p)).toFixed(4)}`;
        }).join('\n') + '\n';
    }
    s += `\n${isGini ? 'Gini' : 'H'}(S) = ${impurity(rows).toFixed(4)}`;
    return s;
}

function gainDetail(rows, feat, g) {
    const nm = impurityName();
    let s = `Gain = ${nm}(S) − Σ (|Sv|/|S|) · ${nm}(Sv)\n\n`;
    s += `${nm}(S) = ${g.parent.toFixed(4)}\n\n`;
    g.groups.forEach(grp => {
        s += `  ${feat} = ${grp.label}\n`;
        s += `    n = ${grp.rows.length}/${rows.length}` +
            `  → ${nm} = ${impurity(grp.rows).toFixed(4)}` +
            `  (${Object.entries(counts(grp.rows)).map(([k, v]) => `${k}:${v}`).join(' ')})\n`;
    });
    s += `\nWeighted ${nm} = ` +
        g.groups.map(grp => `(${grp.rows.length}/${rows.length})×${impurity(grp.rows).toFixed(4)}`).join(' + ') +
        `\n              = ${g.weighted.toFixed(4)}\n\n`;
    s += `Gain(${feat}) = ${g.parent.toFixed(4)} − ${g.weighted.toFixed(4)} = ${g.gain.toFixed(4)}`;
    return s;
}

/* ---------- tree helpers ---------- */
function rootRef(n) { while (n.parent) n = n.parent; return n; }

function cloneTree(n) {
    if (!n) return null;
    return {
        id: n.id, depth: n.depth, feature: n.feature, gain: n.gain,
        leaf: n.leaf, value: n.value, edgeLabel: n.edgeLabel,
        count: n.rows.length, dist: counts(n.rows),
        children: n.children.map(cloneTree)
    };
}
function countLeaves(n) { return n.leaf ? 1 : n.children.reduce((s, c) => s + countLeaves(c), 0); }
function treeDepth(n) { return n.leaf ? n.depth : Math.max(...n.children.map(treeDepth)); }

/* ══════════════ 3. PREDICTION + METRICS ══════════════ */
function predictRow(node, row) {
    if (node.leaf) return node.value;
    // find matching child by edge label
    for (const c of node.children) {
        const lbl = c.edgeLabel;
        if (lbl.startsWith('<= ')) { if (parseFloat(row[node.feature]) <= parseFloat(lbl.slice(3))) return predictRow(c, row); }
        else if (lbl.startsWith('> ')) { if (parseFloat(row[node.feature]) > parseFloat(lbl.slice(2))) return predictRow(c, row); }
        else if (String(row[node.feature]) === lbl) return predictRow(c, row);
    }
    return node.children.length ? predictRow(node.children[0], row) : null;
}

function renderMetrics(root) {
    const preds = ROWS.map(r => predictRow(root, r));
    const acts = ROWS.map(r => String(r[TARGET]));
    const classes = [...new Set(acts)];
    const correct = preds.filter((p, i) => p === acts[i]).length;
    const acc = correct / ROWS.length;

    // macro precision / recall / F1
    let P = 0, R = 0;
    classes.forEach(c => {
        const tp = preds.filter((p, i) => p === c && acts[i] === c).length;
        const fp = preds.filter((p, i) => p === c && acts[i] !== c).length;
        const fn = preds.filter((p, i) => p !== c && acts[i] === c).length;
        P += tp + fp ? tp / (tp + fp) : 0;
        R += tp + fn ? tp / (tp + fn) : 0;
    });
    P /= classes.length; R /= classes.length;
    const F1 = P + R ? 2 * P * R / (P + R) : 0;

    const card = (lbl, val, col) =>
        `<div class="metric-card" style="border-color:${col};background:${col}15;">
           <span class="mc-label" style="color:${col};">${lbl}</span>
           <span class="mc-value" style="color:${col};">${val}</span>
         </div>`;

    document.getElementById('metrics').innerHTML =
        card('Accuracy', (acc * 100).toFixed(2) + '%', '#3b82f6') +
        card('Precision', (P * 100).toFixed(2) + '%', '#7c3aed') +
        card('Recall', (R * 100).toFixed(2) + '%', '#0891b2') +
        card('F1 Score', (F1 * 100).toFixed(2) + '%', '#059669') +
        card('Correct', `${correct} / ${ROWS.length}`, '#d97706');
}

/* ══════════════ 4. CODE PANEL ══════════════ */
const CODE = `def build_decision_tree(rows, features, target):
    # ── 1. Initialise ──────────────────────────────
    tree = grow(rows, features, used=set(), depth=0)

    # ── 2. Entropy / Impurity of a node ────────────
    def impurity(rows):
        counts = Counter(r[target] for r in rows)
        n = len(rows)
        h = 0.0
        for cls, cnt in counts.items():
            p  = cnt / n
            h -= p * log2(p)     # H = −Σ p·log₂(p)
        return h

    # ── 3. Stopping conditions → create a leaf ──────
    def is_leaf(rows, depth, used):
        if len(set(r[target] for r in rows)) == 1:
            return True   # pure node
        if depth >= max_depth: return True

    # ── 4. Information gain for one feature ─────────
    def gain(rows, feat):
        parent   = impurity(rows)
        groups   = split_on(rows, feat)
        weighted = sum(
            len(g) / len(rows) * impurity(g)
            for g in groups
        )   # IG = parent − Σ (|Sv|/|S|)·H(Sv)
        return parent - weighted

    # ── 5. Choose the best feature ───────────────────
    def best_feature(rows, used):
        scores = {f: gain(rows, f)
                  for f in features if f not in used}
        return max(scores, key=scores.get)

    # ── 6. Recurse into every branch ─────────────────
    def grow(rows, features, used, depth):
        feat = best_feature(rows, used)
        for label, subset in split_on(rows, feat):
            node.children[label] = grow(subset, features, used | {feat}, depth + 1)
        return node

    return tree

# ── Prediction ─────────────────────────────────────
def predict(node, row):
    while not node.is_leaf:
        node = node.children[row[node.feature]]
    return node.value


# ── Run ────────────────────────────────────────────
tree  = build_decision_tree(rows, features, target)
preds = [predict(tree, r) for r in rows]`;

(function renderCode() {
    const lines = CODE.split('\n');
    let html = '';
    lines.forEach((ln, i) => {
        const safe = ln.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') || ' ';
        html += `<tr id="row-${i}">` +
            `<td class="ln" id="gutter-${i}">${i + 1}</td>` +
            `<td class="lc" id="line-${i}">${safe}</td>` +
            `</tr>`;
    });
    document.getElementById('code-table').innerHTML = html;
})();

function highlight(nums) {
    // remove all highlights
    document.querySelectorAll('.highlighted-row').forEach(e => e.classList.remove('highlighted-row'));
    if (!nums) return;
    // highlight entire row so gutter and code are always in sync
    nums.forEach(n => {
        document.getElementById(`row-${n - 1}`)?.classList.add('highlighted-row');
    });
    document.getElementById(`row-${nums[0] - 1}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

/* ══════════════ 5. TREE RENDERING (auto-layout) ══════════════ */
function layout(node, depth, xCounter, positions) {
    if (!node) return 0;
    if (node.leaf) {
        /* true leaf — assign next x slot */
        node._x = xCounter.v++;
        node._d = depth;
        positions.push(node);
        return node._x;
    }
    if (!node.children.length) {
        /* internal node not yet split (partial tree during step-by-step build)
           assign next x slot like a leaf so it stays visible and centred     */
        node._x = xCounter.v++;
        node._d = depth;
        positions.push(node);
        return node._x;
    }
    /* fully expanded internal node — centre over its children */
    const xs = node.children.map(c => layout(c, depth + 1, xCounter, positions));
    node._x = (Math.min(...xs) + Math.max(...xs)) / 2;
    node._d = depth;
    positions.push(node);
    return node._x;
}

function renderTree(tree, currentId) {
    const g = document.getElementById('tree-g');
    g.innerHTML = '';
    if (!tree) return;

    const positions = [];
    layout(tree, 0, { v: 0 }, positions);

    const maxX = Math.max(...positions.map(n => n._x), 1);
    const maxD = Math.max(...positions.map(n => n._d), 1);

    const W = 560, H = 340, padX = 45, padY = 40;
    const sx = x => padX + (maxX ? (x / maxX) * (W - 2 * padX) : (W - 2 * padX) / 2);
    /* when only root exists (maxD=0) draw it centred; otherwise spread depths */
    const sy = d => maxD
        ? padY + (d / maxD) * (H - 2 * padY - 20)
        : H / 2;

    const NS = 'http://www.w3.org/2000/svg';
    const el = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };

    /* edges */
    (function drawEdges(n) {
        n.children.forEach(c => {
            const parentOff = 19;                     /* rect hh = 19  */
            const childOff  = c.leaf ? 20 : 19;      /* circle r ≈ 20 */
            const x1 = sx(n._x), y1 = sy(n._d) + parentOff;
            const x2 = sx(c._x), y2 = sy(c._d) - childOff;
            g.appendChild(el('line', { x1, y1, x2, y2, class: 'edge', 'marker-end': 'url(#ah)' }));
            /* perpendicular offset so label sits beside the edge, not on it */
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2;
            const dx = x2 - x1;
            const dy = y2 - y1;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            /* unit perpendicular vector — offset 10px to the right of the edge direction */
            const ox = (-dy / len) * 10;
            const oy = (dx / len) * 10;
            const t = el('text', {
                x: mx + ox, y: my + oy,
                class: 'edge-lbl', 'text-anchor': 'middle'
            });
            t.textContent = c.edgeLabel;
            g.appendChild(t);
            drawEdges(c);
        });
    })(tree);

    /* approximate character widths for auto-sizing nodes */
    const charW    = 7.2;   /* 13px bold  */
    const subCharW = 5.8;   /* 10px normal */

    /* nodes */
    positions.forEach(n => {
        const x = sx(n._x), y = sy(n._d);
        const isLeaf = n.leaf;

        let cls;
        if (!isLeaf) cls = 'nd-dec';
        else {
            const keys = Object.keys(n.dist);
            const positive = keys.length &&
                String(n.value).toLowerCase().match(/^(yes|1|true|y)$/);
            cls = positive ? 'nd-yes' : 'nd-no';
        }
        if (n.id === currentId) cls += ' nd-cur';

        if (isLeaf) {
            /* ── LEAF: circle sized to fit label ── */
            const label    = String(n.value);
            const subLabel = `n=${n.count}`;
            const r = Math.max(18, Math.ceil(
                Math.max(label.length * charW, subLabel.length * subCharW) / 2) + 10);

            g.appendChild(el('circle', { cx: x, cy: y, r, class: cls }));

            const t1 = el('text', { x, y: y - 5,
                'text-anchor': 'middle', 'dominant-baseline': 'central', class: 't-leaf' });
            t1.textContent = label;
            g.appendChild(t1);

            const t2 = el('text', { x, y: y + 9,
                'text-anchor': 'middle', 'dominant-baseline': 'central', class: 't-leafsub' });
            t2.textContent = subLabel;
            g.appendChild(t2);

        } else {
            /* ── INTERNAL: rounded rect sized to fit feature name ── */
            const label    = n.feature || '?';
            const subLabel = n.gain != null ? `g=${n.gain.toFixed(3)}` : '';
            const hw = Math.max(32, Math.ceil(
                Math.max(label.length * charW, subLabel.length * subCharW) / 2) + 12);
            const hh = 19;

            g.appendChild(el('rect', {
                x: x - hw, y: y - hh, width: hw * 2, height: hh * 2,
                rx: 14, ry: 14, class: cls
            }));

            const t1 = el('text', { x, y: y - 5,
                'text-anchor': 'middle', 'dominant-baseline': 'central', class: 't-dec' });
            t1.textContent = label;
            g.appendChild(t1);

            if (subLabel) {
                const t2 = el('text', { x, y: y + 10,
                    'text-anchor': 'middle', 'dominant-baseline': 'central', class: 't-sub' });
                t2.textContent = subLabel;
                g.appendChild(t2);
            }
        }
    });
}

/* ══════════════ 6. SIDE PANELS ══════════════ */
function renderIG(step) {
    const box = document.getElementById('ig-chart');
    if (!step || !step.gains || !Object.keys(step.gains).length) {
        box.innerHTML = '<span class="text-muted small">Awaiting split computation…</span>';
        return;
    }
    const entries = Object.entries(step.gains).sort((a, b) => b[1].gain - a[1].gain);
    const max = Math.max(...entries.map(e => e[1].gain), 1e-9);
    const bestName = entries[0][0];
    const chosen = step.kind === 'select' || step.kind === 'branch' ? step.feature : null;

    box.innerHTML = entries.map(([f, v]) => {
        const pct = Math.max(2, (v.gain / max) * 100);
        const isBest = f === bestName;
        const col = isBest ? '#1e3a5f' : '#94a3b8';
        const mark = (chosen === f) ? '<span class="badge bg-danger ms-1" style="font-size:.55rem;">selected</span>'
            : (isBest ? '<span class="badge bg-primary ms-1" style="font-size:.55rem;">best</span>' : '');
        return `<div class="mb-1 ig-row">
                  <div class="d-flex justify-content-between ${isBest ? 'fw-bold' : ''}">
                    <span>${f}${mark}</span><span>${v.gain.toFixed(4)}</span>
                  </div>
                  <div class="progress" style="height:9px;">
                    <div class="progress-bar" style="width:${pct}%;background:${col};"></div>
                  </div>
                </div>`;
    }).join('');
}

function renderDist(step) {
    const box = document.getElementById('dist-box');
    if (!step || !step.rows) { box.innerHTML = '<span class="text-muted small">—</span>'; return; }
    const c = counts(step.rows), n = step.rows.length;
    const palette = ['#16a34a', '#dc2626', '#2563eb', '#d97706', '#7c3aed', '#0891b2'];
    const keys = Object.keys(c).sort();

    let bar = '<div class="progress mb-2" style="height:16px;">';
    keys.forEach((k, i) => {
        bar += `<div class="progress-bar" style="width:${(c[k] / n * 100)}%;background:${palette[i % palette.length]};"
                     title="${k}">${c[k]}</div>`;
    });
    bar += '</div>';

    const legend = keys.map((k, i) =>
        `<span class="me-2" style="font-size:.68rem;">
           <span style="display:inline-block;width:9px;height:9px;background:${palette[i % palette.length]};border-radius:2px;"></span>
           ${k} = ${c[k]} (${(c[k] / n * 100).toFixed(1)}%)
         </span>`).join('');

    box.innerHTML = `<div class="small mb-1"><b>${n}</b> samples at this node</div>${bar}${legend}`;
}

function renderNodeInfo(step) {
    const box = document.getElementById('node-info');
    if (!step) { box.innerHTML = ''; return; }
    const n = step.node;
    const rows = step.rows || [];
    const kv = (k, v, col = '#3b82f6') =>
        `<div class="row gx-2 mb-1"><div class="col-5"><label class="form-label mb-0 small">${k}</label></div>
         <div class="col-7"><input class="form-control form-control-sm text-center" readonly
              style="border-color:${col};font-size:.7rem;" value="${v}"></div></div>`;

    box.innerHTML =
        kv('Step type', step.kind) +
        kv('Depth', n ? n.depth : '—') +
        kv('Samples', rows.length) +
        kv(impurityName() + '(S)', rows.length ? impurity(rows).toFixed(4) : '—', '#d97706') +
        kv('Feature', step.feature || (n && n.feature) || '—', '#059669') +
        kv('Gain', step.kind === 'gain' && step.gains[step.feature]
            ? step.gains[step.feature].gain.toFixed(4)
            : (n && n.gain != null ? n.gain.toFixed(4) : '—'), '#059669');
}

/* ══════════════ 7. TABLE HIGHLIGHTING ══════════════ */
function renderTable(highlightRows) {
    const thead = document.getElementById('thead');
    const tbody = document.getElementById('tbody');
    thead.innerHTML = `<tr>${COLS.map(c =>
        `<th class="${c === TARGET ? 'table-warning' : ''}">${c}</th>`).join('')}</tr>`;

    const set = highlightRows ? new Set(highlightRows.map(r => r.__i)) : null;
    tbody.innerHTML = ROWS.map(r =>
        `<tr class="${set ? (set.has(r.__i) ? 'row-active' : 'row-dim') : ''}">
           ${COLS.map(c => `<td>${r[c] ?? ''}</td>`).join('')}
         </tr>`).join('');
    document.getElementById('table-box').classList.remove('hidden');
}

/* ══════════════ 8. DATA LOADING ══════════════ */
function loadData(arr) {
    if (!arr.length) { alert('Empty dataset'); return; }
    COLS = Object.keys(arr[0]);
    TARGET = COLS[COLS.length - 1];
    FEATURES = COLS.slice(0, -1);
    ROWS = arr.map((r, i) => ({ ...r, __i: i }));
    FEATURE_TYPE = {};
    FEATURES.forEach(f => FEATURE_TYPE[f] = detectType(f));

    renderTable(null);
    document.getElementById('ds-summary').innerHTML =
        `<b>${ROWS.length}</b> rows · <b>${FEATURES.length}</b> features · target <b>${TARGET}</b><br>` +
        FEATURES.map(f => `${f}<span class="text-secondary">(${FEATURE_TYPE[f][0]})</span>`).join(', ');

    STEPS = []; cursor = -1;
    document.getElementById('calc').innerHTML =
        '<span class="text-muted">Dataset loaded. Press <b>Start</b> to begin execution.</span>';
    document.getElementById('tree-g').innerHTML = '';
    document.getElementById('metrics').innerHTML = '';
    document.getElementById('step-counter').textContent = 'Step 0 / 0';
}

const PRESET = [
    { Weather: 'Sunny', Wind: 'Weak', Humidity: 'High', Play: 'No' },
    { Weather: 'Sunny', Wind: 'Strong', Humidity: 'High', Play: 'No' },
    { Weather: 'Cloudy', Wind: 'Weak', Humidity: 'High', Play: 'Yes' },
    { Weather: 'Rainy', Wind: 'Weak', Humidity: 'High', Play: 'Yes' },
    { Weather: 'Rainy', Wind: 'Weak', Humidity: 'Low', Play: 'Yes' },
    { Weather: 'Rainy', Wind: 'Strong', Humidity: 'Low', Play: 'No' },
    { Weather: 'Cloudy', Wind: 'Strong', Humidity: 'Low', Play: 'Yes' },
    { Weather: 'Sunny', Wind: 'Weak', Humidity: 'High', Play: 'No' },
    { Weather: 'Sunny', Wind: 'Weak', Humidity: 'Low', Play: 'Yes' },
    { Weather: 'Rainy', Wind: 'Weak', Humidity: 'Low', Play: 'Yes' },
    { Weather: 'Sunny', Wind: 'Strong', Humidity: 'Low', Play: 'Yes' },
    { Weather: 'Cloudy', Wind: 'Weak', Humidity: 'High', Play: 'Yes' },
    { Weather: 'Cloudy', Wind: 'Strong', Humidity: 'High', Play: 'Yes' },
    { Weather: 'Rainy', Wind: 'Strong', Humidity: 'High', Play: 'No' },
];

const setActive = id => ['presetBtn', 'randomBtn', 'customBtn']
    .forEach(b => document.getElementById(b).classList.toggle('btn-active', b === id));

document.getElementById('presetBtn').onclick = () => {
    setActive('presetBtn');
    document.getElementById('upload-box').classList.add('hidden');
    loadData(JSON.parse(JSON.stringify(PRESET)));
};

document.getElementById('randomBtn').onclick = () => {
    setActive('randomBtn');
    document.getElementById('upload-box').classList.add('hidden');
    const A = ['Low', 'Medium', 'High'], B = ['Yes', 'No'], C = ['Red', 'Green', 'Blue'];
    const pick = a => a[Math.floor(Math.random() * a.length)];
    const data = Array.from({ length: 16 }, () => {
        const size = pick(A), avail = pick(B), colour = pick(C);
        // give it some learnable structure plus noise
        let buy = (size === 'High' || (avail === 'Yes' && colour !== 'Blue')) ? 'Yes' : 'No';
        if (Math.random() < 0.15) buy = buy === 'Yes' ? 'No' : 'Yes';
        return { Size: size, Available: avail, Colour: colour, Buy: buy };
    });
    loadData(data);
};

document.getElementById('customBtn').onclick = () => {
    setActive('customBtn');
    document.getElementById('upload-box').classList.remove('hidden');
};

let chosenFile = null;
document.getElementById('fileInput').onchange = e => { chosenFile = e.target.files[0]; };
document.getElementById('submitFile').onclick = () => {
    if (!chosenFile) { alert('Please choose a file first.'); return; }
    const fr = new FileReader();
    fr.onload = ev => {
        try {
            const wb = XLSX.read(new Uint8Array(ev.target.result), { type: 'array' });
            const json = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: '' });
            loadData(json);
            document.getElementById('upload-box').classList.add('hidden');
        } catch (err) { alert('Could not read that file: ' + err.message); }
    };
    fr.readAsArrayBuffer(chosenFile);
};

/* ══════════════ 9. NAVIGATION ══════════════ */
function show(i) {
    if (!STEPS.length) return;
    cursor = Math.max(0, Math.min(i, STEPS.length - 1));
    const s = STEPS[cursor];

    highlight(s.lines);
    document.getElementById('calc').innerHTML =
        `<span class="calc-title">${s.title}</span>` +
        `<span class="calc-body">${s.detail.replace(/</g,'&lt;')}</span>`;
    document.getElementById('step-counter').textContent = `Step ${cursor + 1} / ${STEPS.length}`;

    renderTree(s.tree || null, s.node ? s.node.id : -1);
    renderIG(s);
    renderDist(s);
    renderNodeInfo(s);
    renderTable(s.rows);
}

/* Start = build the tree, then jump straight to the first execution step.
   Nothing is drawn ahead of time: the tree grows only as the learner steps. */
document.getElementById('startBtn').onclick = () => {
    if (!ROWS.length) { alert('Load a dataset first.'); return; }
    const root = buildTree();
    renderMetrics(root);
    show(0);
};

document.getElementById('nextBtn').onclick = () => { if (STEPS.length) show(cursor + 1); };
document.getElementById('prevBtn').onclick = () => { if (STEPS.length) show(cursor - 1); };
document.getElementById('endBtn').onclick = () => {
    if (!STEPS.length) {                    // allow End without pressing Start first
        if (!ROWS.length) { alert('Load a dataset first.'); return; }
        renderMetrics(buildTree());
    }
    show(STEPS.length - 1);
};

/* auto-load preset on first open */
document.getElementById('presetBtn').click();