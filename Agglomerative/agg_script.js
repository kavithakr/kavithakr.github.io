/* ══════════════════════════════════════════════════════════════════
   Agglomerative Clustering Visualization — agg_script.js
   Fully dynamic: works on any x1,x2 dataset.
   ══════════════════════════════════════════════════════════════════ */

const CODE = `import numpy as np
from scipy.spatial.distance import cdist

# ── 1. Compute initial distance matrix ───────────────
def compute_distance_matrix(X, metric):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i,j] = D[j,i] = metric(X[i], X[j])
    return D

# ── 2. Find the closest pair of clusters ─────────────
def find_min_distance(D, clusters):
    min_dist = np.inf
    pair = (None, None)
    ids  = list(clusters.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            d = D[ids[i]][ids[j]]
            if d < min_dist:
                min_dist = d
                pair = (ids[i], ids[j])
    return pair, min_dist

# ── 3. Merge two clusters ────────────────────────────
def merge_clusters(clusters, a, b):
    new_cluster = clusters[a] + clusters[b]
    del clusters[a], clusters[b]
    new_id = max(clusters.keys(), default=-1) + 1
    clusters[new_id] = new_cluster
    return clusters, new_id

# ── 4. Update distance matrix (linkage) ──────────────
def update_distances(D, clusters, a, b, new_id, linkage):
    for c in clusters:
        if c == new_id: continue
        da = D[a][c]
        db = D[b][c]
        if   linkage == 'single':   D[new_id][c] = D[c][new_id] = min(da, db)
        elif linkage == 'complete': D[new_id][c] = D[c][new_id] = max(da, db)
        elif linkage == 'average':  D[new_id][c] = D[c][new_id] = (da+db)/2
    return D

# ── 5. Main agglomerative loop ───────────────────────
def agglomerative(X, n_clusters, linkage, metric):
    D        = compute_distance_matrix(X, metric)
    clusters = {i:[i] for i in range(len(X))}
    history  = []
    while len(clusters) > n_clusters:
        (a,b), dist = find_min_distance(D, clusters)
        clusters, new_id = merge_clusters(clusters, a, b)
        D = update_distances(D, clusters, a, b, new_id, linkage)
        history.append((a, b, dist, new_id))
    return clusters, history`;

const LINE = {
    init:    [46, 47],
    dist_mx: [4, 5, 6, 7, 8, 9, 10,11],
    find_min:[13, 14, 15, 16, 17, 18, 19, 20,21,22, 23, 24],
    merge:   [26, 27, 28, 29, 30, 31, 32],
    update:  [ 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    loop:    [50, 51, 52, 53, 54],
    done:    [55],
};

/* ── render code ── */
(function renderCode() {
    const lines = CODE.split('\n');
    let html = '';
    lines.forEach((ln, i) => {
        const safe = ln.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') || ' ';
        html += `<tr id="row-${i}"><td class="ln">${i+1}</td><td class="lc">${safe}</td></tr>`;
    });
    document.getElementById('code-table').innerHTML = html;
})();

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
const PALETTE = ['#2563eb','#dc2626','#16a34a','#d97706','#7c3aed','#0891b2','#db2777','#65a30d'];
const fmt = (v,d=4) => Number(v).toFixed(d);

/* ══════════════════════════════════════════════════════════════════
   CORE ALGORITHM
   ══════════════════════════════════════════════════════════════════ */
function euclidean(a,b){ return Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2); }
function manhattan(a,b){ return Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1]); }

function distFn(){
    return document.getElementById('distInput').value==='manhattan' ? manhattan : euclidean;
}

/* build n×n distance matrix between all DATA points */
function buildDistMatrix(data, metric) {
    const n = data.length;
    const D = Array.from({length:n},()=>new Array(n).fill(0));
    for (let i=0;i<n;i++)
        for (let j=i+1;j<n;j++)
            D[i][j]=D[j][i]=metric(data[i],data[j]);
    return D;
}

/* cluster-level distance matrix using linkage */
function clusterDist(cidA, cidB, clusters, baseD, linkage) {
    const mA = clusters[cidA], mB = clusters[cidB];
    const dists = [];
    mA.forEach(a => mB.forEach(b => dists.push(baseD[a][b])));
    if (linkage==='single')   return Math.min(...dists);
    if (linkage==='complete') return Math.max(...dists);
    return dists.reduce((s,v)=>s+v,0)/dists.length; // average
}

/* find the pair of clusters with minimum linkage distance */
function findMinPair(clusters, baseD, linkage) {
    const ids = Object.keys(clusters).map(Number);
    let minD=Infinity, pair=[null,null];
    for (let i=0;i<ids.length;i++)
        for (let j=i+1;j<ids.length;j++) {
            const d = clusterDist(ids[i],ids[j],clusters,baseD,linkage);
            if (d<minD) { minD=d; pair=[ids[i],ids[j]]; }
        }
    return {pair, minD};
}

/* build cluster-level distance table for display */
function buildClusterDistTable(clusters, baseD, linkage) {
    const ids = Object.keys(clusters).map(Number).sort((a,b)=>a-b);
    const table = {};
    ids.forEach(a => {
        table[a]={};
        ids.forEach(b => {
            table[a][b] = a===b ? 0 : clusterDist(a,b,clusters,baseD,linkage);
        });
    });
    return {ids, table};
}

/* ══════════════════════════════════════════════════════════════════
   STEP TRACE
   ══════════════════════════════════════════════════════════════════ */
let DATA=[], STEPS=[], cursor=-1;

function buildSteps() {
    STEPS=[];
    const nC      = parseInt(document.getElementById('nClustersInput').value)||3;
    const linkage = document.getElementById('linkageInput').value;
    const metric  = distFn();
    const metricName = document.getElementById('distInput').value;
    const n       = DATA.length;

    /* ── step 0: init ── */
    STEPS.push({
        kind:'init', lines:LINE.init,
        title:'Import libraries — initialise',
        clusters: Object.fromEntries(DATA.map((_,i)=>[i,[i]])),
        baseD: null, clusterD: null, merges: [],
        highlightPair: null, minD: null,
        detail:
            `import numpy as np\n\n` +
            `Points    : ${n}\n` +
            `n_clusters: ${nC}\n` +
            `Linkage   : ${linkage}\n` +
            `Distance  : ${metricName}\n` +
            `Merges needed: ${n-nC}`
    });

    /* ── step 1: compute initial distance matrix ── */
    const baseD = buildDistMatrix(DATA, metric);
    let clusters = Object.fromEntries(DATA.map((_,i)=>[i,[i]]));
    const {ids:ids0, table:cdt0} = buildClusterDistTable(clusters, baseD, linkage);

    STEPS.push({
        kind:'dist_mx', lines:LINE.dist_mx,
        title:`Compute ${n}×${n} distance matrix`,
        clusters: JSON.parse(JSON.stringify(clusters)),
        baseD, clusterDist:{ids:ids0,table:cdt0},
        merges:[], highlightPair:null, minD:null,
        detail:
            `D[i][j] = ${metricName} distance between point i and j\n\n` +
            `Matrix size: ${n}×${n}\n\n` +
            `Sample distances:\n` +
            DATA.slice(0,Math.min(4,n)).map((pt,i)=>
                DATA.slice(i+1,Math.min(i+3,n)).map(pt2=>
                    `  d(P${i},P${DATA.indexOf(pt2)}) = ${fmt(metric(pt,pt2),4)}`
                ).join('\n')
            ).filter(Boolean).join('\n')
    });

    /* ── main merge loop ── */
    const merges = [];
    let nextId = n;

    while (Object.keys(clusters).length > nC) {
        const {pair, minD} = findMinPair(clusters, baseD, linkage);
        const [a,b] = pair;
        const {ids:idsB, table:cdtB} = buildClusterDistTable(clusters, baseD, linkage);

        /* step: find min ── show full distance table, highlight min cell */
        STEPS.push({
            kind:'find_min', lines:LINE.find_min,
            title:`Find closest pair → C${a} & C${b}  (d=${fmt(minD,4)})`,
            clusters: JSON.parse(JSON.stringify(clusters)),
            baseD, clusterDist:{ids:idsB, table:cdtB},
            merges:[...merges], highlightPair:[a,b], minD,
            detail:
                `Scanning all ${Object.keys(clusters).length} cluster pairs…\n\n` +
                `Minimum distance: ${fmt(minD,4)}\n` +
                `Closest pair    : Cluster ${a} ↔ Cluster ${b}\n\n` +
                `Cluster ${a} members: [${clusters[a].map(i=>`P${i}`).join(', ')}]\n` +
                `Cluster ${b} members: [${clusters[b].map(i=>`P${i}`).join(', ')}]`
        });

        /* merge */
        const newMembers = [...clusters[a],...clusters[b]];
        delete clusters[a]; delete clusters[b];
        clusters[nextId] = newMembers;
        merges.push({a, b, dist:minD, newId:nextId});
        const {ids:idsM, table:cdtM} = buildClusterDistTable(clusters, baseD, linkage);

        /* step: merge ── */
        STEPS.push({
            kind:'merge', lines:LINE.merge,
            title:`Merge Cluster ${a} + Cluster ${b} → Cluster ${nextId}`,
            clusters: JSON.parse(JSON.stringify(clusters)),
            baseD, clusterDist:{ids:idsM, table:cdtM},
            merges:[...merges], highlightPair:null, minD,
            detail:
                `Merging:\n` +
                `  Cluster ${a}: [${newMembers.slice(0,newMembers.length-clusters[nextId]?.length||0).map(i=>`P${i}`).join(', ')}]\n` +
                `  Cluster ${b}: […]\n` +
                `  → Cluster ${nextId}: [${newMembers.map(i=>`P${i}`).join(', ')}]\n\n` +
                `Clusters remaining: ${Object.keys(clusters).length}`
        });

        /* step: update distances ── */
        const {ids:idsU, table:cdtU} = buildClusterDistTable(clusters, baseD, linkage);
        STEPS.push({
            kind:'update', lines:LINE.update,
            title:`Update distances for Cluster ${nextId} (${linkage} linkage)`,
            clusters: JSON.parse(JSON.stringify(clusters)),
            baseD, clusterDist:{ids:idsU, table:cdtU},
            merges:[...merges], highlightPair:null, minD,
            detail:
                `Linkage = ${linkage}\n\n` +
                Object.keys(clusters).filter(c=>Number(c)!==nextId).map(c=>{
                    const d = clusterDist(nextId, Number(c), clusters, baseD, linkage);
                    return `  dist(C${nextId}, C${c}) = ${fmt(d,4)}  [${linkage}]`;
                }).join('\n') +
                `\n\nMatrix updated. Clusters remaining: ${Object.keys(clusters).length}`
        });

        nextId++;
    }

    /* ── done ── */
    const {ids:idsFin, table:cdtFin} = buildClusterDistTable(clusters, baseD, linkage);
    const clusterList = Object.values(clusters);
    STEPS.push({
        kind:'done', lines:LINE.done,
        title:'Agglomerative clustering complete',
        clusters: JSON.parse(JSON.stringify(clusters)),
        baseD, clusterDist:{ids:idsFin, table:cdtFin},
        merges:[...merges], highlightPair:null, minD:null,
        detail:
            `Clustering complete!\n\n` +
            `Total merges     : ${merges.length}\n` +
            `Final n_clusters : ${Object.keys(clusters).length}\n\n` +
            `Final clusters:\n` +
            clusterList.map((members,i)=>
                `  Cluster ${i+1}: [${members.map(j=>`P${j}`).join(', ')}]`
            ).join('\n')
    });
}

/* ══════════════════════════════════════════════════════════════════
   COLOUR ASSIGNMENT — give each final cluster a colour
   ══════════════════════════════════════════════════════════════════ */
function getPointColors(step) {
    if (!step || !step.clusters) return {};
    const colorMap = {}; // point index → colour
    Object.values(step.clusters).forEach((members, ci) => {
        members.forEach(pi => { colorMap[pi] = PALETTE[ci % PALETTE.length]; });
    });
    return colorMap;
}

/* ══════════════════════════════════════════════════════════════════
   SCATTER CANVAS
   ══════════════════════════════════════════════════════════════════ */
function drawScatter(step) {
    const canvas = document.getElementById('scatter-canvas');
    const W = canvas.offsetWidth||480, H = canvas.offsetHeight||260;
    canvas.width=W; canvas.height=H;
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    if (!DATA.length) return;

    const pad=32;
    const x1s=DATA.map(d=>d[0]), x2s=DATA.map(d=>d[1]);
    const x1Min=Math.min(...x1s)-1, x1Max=Math.max(...x1s)+1;
    const x2Min=Math.min(...x2s)-1, x2Max=Math.max(...x2s)+1;
    const px=x=>pad+(x-x1Min)/(x1Max-x1Min)*(W-2*pad);
    const py=y=>H-pad-(y-x2Min)/(x2Max-x2Min)*(H-2*pad);

    /* axes */
    ctx.strokeStyle='#94a3b8'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad,6); ctx.lineTo(pad,H-pad); ctx.lineTo(W-6,H-pad); ctx.stroke();
    ctx.fillStyle='#64748b'; ctx.font='9px sans-serif';
    ctx.fillText('x1',W-14,H-pad+2);
    ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2); ctx.fillText('x2',0,0); ctx.restore();

    /* grid */
    ctx.setLineDash([2,3]); ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=0.6;
    for(let t=0;t<=4;t++){
        const gx=pad+t*(W-2*pad)/4, gy=(H-pad)-t*(H-2*pad)/4;
        ctx.beginPath(); ctx.moveTo(gx,6); ctx.lineTo(gx,H-pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-6,gy); ctx.stroke();
        ctx.fillStyle='#94a3b8'; ctx.font='7px sans-serif';
        ctx.fillText((x1Min+t*(x1Max-x1Min)/4).toFixed(1),gx-8,H-pad+10);
        ctx.fillText((x2Min+t*(x2Max-x2Min)/4).toFixed(1),2,gy+3);
    }
    ctx.setLineDash([]);

    const colorMap = getPointColors(step);

    /* draw merge connection lines */
    if (step && step.merges && step.merges.length) {
        step.merges.forEach(m => {
            const membersA = findOriginalMembers(m.a, step.merges, DATA.length);
            const membersB = findOriginalMembers(m.b, step.merges, DATA.length);
            const centA = centroid(membersA);
            const centB = centroid(membersB);
            if (!centA || !centB) return;
            ctx.beginPath();
            ctx.moveTo(px(centA[0]),py(centA[1]));
            ctx.lineTo(px(centB[0]),py(centB[1]));
            ctx.strokeStyle='#94a3b8'; ctx.lineWidth=0.8;
            ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]);
        });
    }

    /* highlight the pair being merged */
    if (step && step.highlightPair) {
        const [a,b] = step.highlightPair;
        const mA = findOriginalMembers(a, step.merges||[], DATA.length);
        const mB = findOriginalMembers(b, step.merges||[], DATA.length);
        /* draw connecting line with distance label */
        const cA=centroid(mA), cB=centroid(mB);
        if(cA && cB){
            ctx.beginPath(); ctx.moveTo(px(cA[0]),py(cA[1])); ctx.lineTo(px(cB[0]),py(cB[1]));
            ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2; ctx.setLineDash([6,3]);
            ctx.stroke(); ctx.setLineDash([]);
            /* distance label on the edge */
            const mx2=(px(cA[0])+px(cB[0]))/2;
            const my2=(py(cA[1])+py(cB[1]))/2;
            const dx=px(cB[0])-px(cA[0]), dy=py(cB[1])-py(cA[1]);
            const len=Math.sqrt(dx*dx+dy*dy)||1;
            const ox=(-dy/len)*12, oy=(dx/len)*12;
            const lbl=`d=${fmt(step.minD,3)}`;
            ctx.font='bold 8.5px Consolas,monospace';
            const tw=ctx.measureText(lbl).width;
            ctx.fillStyle='rgba(255,255,255,0.9)';
            ctx.fillRect(mx2+ox-tw/2-2,my2+oy-9,tw+4,12);
            ctx.fillStyle='#d97706'; ctx.textAlign='center';
            ctx.fillText(lbl,mx2+ox,my2+oy); ctx.textAlign='left';
        }
        /* highlight halos */
        [...mA,...mB].forEach(pi=>{
            ctx.beginPath(); ctx.arc(px(DATA[pi][0]),py(DATA[pi][1]),10,0,2*Math.PI);
            ctx.fillStyle='rgba(245,158,11,0.2)'; ctx.fill();
        });
    }

    /* draw points */
    DATA.forEach((pt,i)=>{
        const col = colorMap[i] || '#94a3b8';
        ctx.beginPath(); ctx.arc(px(pt[0]),py(pt[1]),5,0,2*Math.PI);
        ctx.fillStyle=col; ctx.strokeStyle='#fff'; ctx.lineWidth=1.2;
        ctx.fill(); ctx.stroke();
        /* point label */
        ctx.fillStyle='#334155'; ctx.font='7px sans-serif';
        ctx.fillText(`P${i}`,px(pt[0])+6,py(pt[1])-5);
    });

    /* cluster convex hulls (approximate: draw circle around centroid) */
    if (step && step.clusters) {
        Object.values(step.clusters).forEach((members,ci)=>{
            if (members.length<2) return;
            const cen=centroid(members);
            if(!cen) return;
            const r=Math.max(...members.map(pi=>
                Math.sqrt((px(DATA[pi][0])-px(cen[0]))**2+(py(DATA[pi][1])-py(cen[1]))**2)
            ))+10;
            ctx.beginPath(); ctx.arc(px(cen[0]),py(cen[1]),r,0,2*Math.PI);
            ctx.strokeStyle=PALETTE[ci%PALETTE.length]+'66';
            ctx.lineWidth=1.2; ctx.setLineDash([4,3]);
            ctx.stroke(); ctx.setLineDash([]);
        });
    }
}

function findOriginalMembers(clusterId, merges, n) {
    // if id < n it is an original point
    if (clusterId < n) return [clusterId];
    // find the merge that created this cluster
    const m = merges.find(m=>m.newId===clusterId);
    if (!m) return [];
    return [
        ...findOriginalMembers(m.a, merges, n),
        ...findOriginalMembers(m.b, merges, n)
    ];
}

function centroid(ptIndices) {
    if (!ptIndices.length) return null;
    return [
        ptIndices.reduce((s,i)=>s+DATA[i][0],0)/ptIndices.length,
        ptIndices.reduce((s,i)=>s+DATA[i][1],0)/ptIndices.length,
    ];
}

/* ══════════════════════════════════════════════════════════════════
   DENDROGRAM CANVAS
   ══════════════════════════════════════════════════════════════════ */
function drawDendrogram(step) {
    const canvas = document.getElementById('dendro-canvas');
    const W = canvas.offsetWidth||480, H = canvas.offsetHeight||180;
    canvas.width=W; canvas.height=H;
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    if (!DATA.length || !step || !step.merges || !step.merges.length) {
        ctx.fillStyle='#94a3b8'; ctx.font='9px sans-serif';
        ctx.fillText('Dendrogram builds as merges happen…',10,H/2);
        return;
    }

    const n = DATA.length;
    const merges = step.merges;
    const padL=30, padR=10, padT=10, padB=24;
    const maxDist = Math.max(...merges.map(m=>m.dist)) * 1.1 || 1;

    /* leaf x-positions — evenly spaced */
    const leafX = {};
    DATA.forEach((_,i)=>{ leafX[i]=(padL + (i/(n-1||1))*(W-padL-padR)); });

    /* compute x-position for merged clusters (average of children) */
    const clusterX={...leafX};
    merges.forEach(m=>{
        const xA=clusterX[m.a]??0, xB=clusterX[m.b]??0;
        clusterX[m.newId]=(xA+xB)/2;
    });

    /* y = proportional to merge distance */
    /* y-axis: distance 0 at BOTTOM (leaves), maxDist at TOP (root)
       Higher merge distance → higher position on canvas (smaller y value) */
    const yScale = d => (H - padB) - (d / maxDist) * (H - padT - padB);

    /* axes */
    ctx.strokeStyle='#94a3b8'; ctx.lineWidth=0.8;
    ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,H-padB); ctx.stroke();
    ctx.fillStyle='#64748b'; ctx.font='7px sans-serif';
    ctx.fillText('0',         2, H-padB+3);          /* distance 0 at bottom */
    ctx.fillText(fmt(maxDist/1.1,2), 2, padT+6);    /* maxDist at top */

    /* draw leaf labels at bottom, stubs going upward */
    DATA.forEach((_,i)=>{
        ctx.fillStyle='#334155'; ctx.font='7px sans-serif'; ctx.textAlign='center';
        ctx.fillText(`P${i}`,leafX[i],H-padB+10);
        ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=0.6;
        ctx.beginPath(); ctx.moveTo(leafX[i],H-padB); ctx.lineTo(leafX[i],H-padB-4); ctx.stroke();
    });
    ctx.textAlign='left';

    /* draw each merge as an upside-down U */
    merges.forEach((m,mi)=>{
        const xA=clusterX[m.a], xB=clusterX[m.b];
        const yM=yScale(m.dist);
        const col=PALETTE[mi%PALETTE.length];
        const isLast = mi===merges.length-1 && step.kind==='find_min';
        ctx.strokeStyle = isLast ? '#f59e0b' : col;
        ctx.lineWidth   = isLast ? 2 : 1.2;

        /* vertical lines from each child's merge height UP to yM */
        const yA = m.a < n ? H-padB : yScale(merges.find(mm=>mm.newId===m.a)?.dist || 0);
        const yB = m.b < n ? H-padB : yScale(merges.find(mm=>mm.newId===m.b)?.dist || 0);
        ctx.beginPath(); ctx.moveTo(xA,yA); ctx.lineTo(xA,yM); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xB,yB); ctx.lineTo(xB,yM); ctx.stroke();
        /* horizontal bar */
        ctx.beginPath(); ctx.moveTo(xA,yM); ctx.lineTo(xB,yM); ctx.stroke();
        /* distance label */
        ctx.fillStyle=col; ctx.font='6.5px Consolas,monospace'; ctx.textAlign='center';
        ctx.fillText(fmt(m.dist,3),(xA+xB)/2,yM-2);
        ctx.textAlign='left';
    });
}

/* ══════════════════════════════════════════════════════════════════
   DISTANCE MATRIX TABLE
   ══════════════════════════════════════════════════════════════════ */
function renderDistMatrix(step) {
    const box = document.getElementById('dist-matrix-wrap');
    if (!step || !step.clusterDist) {
        box.innerHTML='<span style="color:#475569;font-size:9px;font-family:var(--mono);">—</span>'; return;
    }
    const {ids, table} = step.clusterDist;
    const pair = step.highlightPair || [];

    /* find global min (ignoring diagonal) */
    let minVal=Infinity;
    ids.forEach(a=>ids.forEach(b=>{ if(a!==b && table[a][b]<minVal) minVal=table[a][b]; }));

    let html=`<table><thead><tr><th>C\\C</th>${ids.map(i=>`<th>C${i}</th>`).join('')}</tr></thead><tbody>`;
    ids.forEach(a=>{
        html+=`<tr><th>C${a}</th>`;
        ids.forEach(b=>{
            if(a===b){ html+=`<td>—</td>`; return; }
            const v=table[a][b];
            const isMin = Math.abs(v-minVal)<1e-9;
            const isPair= pair.includes(a)&&pair.includes(b);
            const cls   = isPair?'cell-min':(isMin?'cell-min':'');
            html+=`<td class="${cls}">${fmt(v,3)}</td>`;
        });
        html+='</tr>';
    });
    html+='</tbody></table>';
    box.innerHTML=html;
}

/* ── merge log ── */
function renderMergeLog(step) {
    const box=document.getElementById('merge-log');
    if(!step||!step.merges||!step.merges.length){box.innerHTML='<span style="color:#475569;font-size:9px;font-family:var(--mono);">No merges yet</span>';return;}
    box.innerHTML=step.merges.map((m,i)=>
        `<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;font-size:9px;font-family:var(--mono);">
           <span style="background:${PALETTE[i%PALETTE.length]};color:#fff;border-radius:3px;padding:1px 5px;font-size:8px;font-weight:700;flex-shrink:0;">M${i+1}</span>
           <span style="color:var(--ink-2);">C${m.a}+C${m.b}→C${m.newId}</span>
           <span style="color:var(--amber);margin-left:auto;">d=${fmt(m.dist,3)}</span>
         </div>`
    ).join('');
}

/* ── metrics ── */
function renderMetrics(step) {
    if(!step||!step.clusters){document.getElementById('metrics').innerHTML='';return;}
    const nC=Object.keys(step.clusters).length;
    const nPts=DATA.length;
    const nMerges=step.merges.length;
    const linkage=document.getElementById('linkageInput').value;

    /* silhouette approximation */
    const colorMap=getPointColors(step);
    const clustersArr=Object.values(step.clusters);
    let silSum=0;
    DATA.forEach((pt,i)=>{
        const ci=clustersArr.findIndex(m=>m.includes(i));
        const myMembers=clustersArr[ci].filter(j=>j!==i);
        const a=myMembers.length?myMembers.reduce((s,j)=>s+euclidean(pt,DATA[j]),0)/myMembers.length:0;
        let b=Infinity;
        clustersArr.forEach((m,cj)=>{
            if(cj===ci||!m.length) return;
            const avgD=m.reduce((s,j)=>s+euclidean(pt,DATA[j]),0)/m.length;
            if(avgD<b) b=avgD;
        });
        silSum+=b===Infinity?0:(b-a)/Math.max(a,b);
    });
    const sil=silSum/nPts;

    const card=(l,v,c)=>
        `<div class="metric-card" style="border-color:${c};background:${c}15;">
           <span class="mc-label" style="color:${c};">${l}</span>
           <span class="mc-value" style="color:${c};">${v}</span>
         </div>`;
    document.getElementById('metrics').innerHTML=
        card('Clusters',  nC,           '#3b82f6')+
        card('Points',    nPts,          '#0891b2')+
        card('Merges',    nMerges,       '#7c3aed')+
        card('Linkage',   linkage,       '#d97706')+
        card('Silhouette',fmt(sil,3),    '#059669');
}

/* ══════════════════════════════════════════════════════════════════
   SHOW STEP
   ══════════════════════════════════════════════════════════════════ */
function show(i) {
    if(!STEPS.length) return;
    cursor=Math.max(0,Math.min(i,STEPS.length-1));
    const s=STEPS[cursor];

    highlight(s.lines);
    document.getElementById('step-counter').textContent=`Step ${cursor+1} / ${STEPS.length}`;
    document.getElementById('calc').innerHTML=
        `<span class="calc-title">${s.title}</span>`+
        `<span class="calc-body">${s.detail.replace(/</g,'&lt;')}</span>`;
    document.getElementById('mergeDisplay').value=
        s.merges?s.merges.length||'0':'—';
    document.getElementById('clusterCountDisplay').value=
        s.clusters?Object.keys(s.clusters).length:'—';
    document.getElementById('minDistDisplay').value=
        s.minD!=null?fmt(s.minD,4):'—';

    drawScatter(s);
    drawDendrogram(s);
    renderDistMatrix(s);
    renderMergeLog(s);
    renderMetrics(s);
}

/* ══════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════ */
function runAgg() {
    if(!DATA.length){alert('Load a dataset first.');return;}
    buildSteps();
    show(0);
}

document.getElementById('startBtn').onclick=runAgg;
document.getElementById('nextBtn').onclick =()=>{if(STEPS.length)show(cursor+1);};
document.getElementById('prevBtn').onclick =()=>{if(STEPS.length)show(cursor-1);};
document.getElementById('endBtn').onclick  =()=>{if(!STEPS.length)runAgg();else show(STEPS.length-1);};

/* ══════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════ */
function loadData(arr) {
    if(!arr.length){alert('Empty dataset');return;}
    const cols=Object.keys(arr[0]);
    DATA=arr.map(r=>[parseFloat(r[cols[0]]),parseFloat(r[cols[1]])]).filter(d=>!isNaN(d[0])&&!isNaN(d[1]));
    if(!DATA.length){alert('Could not parse numeric columns.');return;}

    const thead=document.getElementById('thead');
    const tbody=document.getElementById('tbody');
    thead.innerHTML=`<tr><th>${cols[0]}</th><th>${cols[1]}</th></tr>`;
    tbody.innerHTML=DATA.map(d=>`<tr><td>${fmt(d[0],2)}</td><td>${fmt(d[1],2)}</td></tr>`).join('');
    document.getElementById('table-box').classList.remove('hidden');

    STEPS=[]; cursor=-1;
    document.getElementById('calc').innerHTML=
        `<span style="color:#475569;">${DATA.length} pts loaded. Press <span style="color:#38bdf8;">Start</span>.</span>`;
    document.getElementById('step-counter').textContent='Step 0 / 0';
    document.getElementById('metrics').innerHTML='';
    drawScatter(null); drawDendrogram(null);
    document.getElementById('dist-matrix-wrap').innerHTML='<span style="color:#475569;font-size:9px;font-family:var(--mono);">—</span>';
    document.getElementById('merge-log').innerHTML='';
}

/* ── preset — small dataset so distance matrix is readable ── */
const PRESET=[
    {x1:1.0,x2:1.5},{x1:1.5,x2:2.0},{x1:1.2,x2:0.8},   /* group A */
    {x1:5.0,x2:5.5},{x1:5.5,x2:4.8},{x1:4.8,x2:5.2},   /* group B */
    {x1:8.5,x2:1.0},{x1:9.0,x2:1.8},{x1:8.2,x2:0.5},   /* group C */
    {x1:3.5,x2:8.0},{x1:4.0,x2:7.5},                    /* group D (boundary) */
];

const setActive=id=>['presetBtn','randomBtn','customBtn']
    .forEach(b=>document.getElementById(b).classList.toggle('btn-active',b===id));

document.getElementById('presetBtn').onclick=()=>{
    setActive('presetBtn');
    document.getElementById('upload-box').classList.add('hidden');
    loadData(JSON.parse(JSON.stringify(PRESET)));
};

document.getElementById('randomBtn').onclick=()=>{
    setActive('randomBtn');
    document.getElementById('upload-box').classList.add('hidden');
    const k=3;
    const centres=[[2,2],[7,7],[8,2]];
    const data=[];
    centres.forEach(c=>{
        for(let i=0;i<4;i++)
            data.push({x1:+(c[0]+(Math.random()-0.5)*2.5).toFixed(2),
                       x2:+(c[1]+(Math.random()-0.5)*2.5).toFixed(2)});
    });
    loadData(data);
};

document.getElementById('customBtn').onclick=()=>{
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

document.getElementById('presetBtn').click();