/* ══════════════════════════════════════════════════════════════════
   Ridge Regression Visualization — ridge_script.js
   Step-by-step: predict → MSE → L2 penalty → total loss →
                 gradients → update w,b → draw line
   ══════════════════════════════════════════════════════════════════ */

const CODE = `import numpy as np

# ── 1. Prediction ────────────────────────────────────
def predict(X, w, b):
    return w * X + b

# ── 2. MSE Loss ──────────────────────────────────────
def mse_loss(y_true, y_pred):
    n = len(y_true)
    return (1 / n) * np.sum((y_pred - y_true) ** 2)

# ── 3. L2 Penalty ────────────────────────────────────
def l2_penalty(w, lam):
    return lam * (w ** 2)

# ── 4. Total Ridge Loss ──────────────────────────────
def ridge_loss(y_true, y_pred, w, lam):
    return mse_loss(y_true, y_pred) + l2_penalty(w, lam)

# ── 5. Gradients ─────────────────────────────────────
def gradients(X, y_true, y_pred, w, lam):
    n = len(y_true)
    dw = (2/n) * np.sum((y_pred - y_true) * X) + 2 * lam * w
    db = (2/n) * np.sum(y_pred - y_true)
    return dw, db

# ── 6. Update parameters ─────────────────────────────
def update(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b

# ── 7. Draw regression line ──────────────────────────
def draw_regression_line(X, w, b):
    x_plot = np.linspace(X.min()-1, X.max()+1, 300)
    y_plot = w * x_plot + b
    plt.plot(x_plot, y_plot, color='purple', lw=2,
             label=f'Ridge: w={w:.4f}, b={b:.4f}')
    plt.show()

# ── 8. Training loop ─────────────────────────────────
def train(X, y, lr, iterations, lam):
    w, b = 0.0, 0.0
    for iteration in range(iterations):

        # Step 4: predict
        y_pred = predict(X, w, b)

        # Step 5: MSE loss
        mse = mse_loss(y, y_pred)

        # Step 6: L2 penalty
        penalty = l2_penalty(w, lam)

        # Step 7: total ridge loss
        loss = mse + penalty

        # Step 8: gradients
        dw, db = gradients(X, y, y_pred, w, lam)

        # Step 9: update
        w, b = update(w, b, dw, db, lr)

        # Step 10: draw regression line
        draw_regression_line(X, w, b)

    return w, b

# ── 9. Evaluate ──────────────────────────────────────
w, b = train(X_train, y_train, lr, iterations, lam)
y_pred_test = predict(X_test, w, b)
mse_test  = mse_loss(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)`;

const LINE = {
    init:    [42, 43, 70],
    mse:     [7, 8, 9, 10],
    penalty: [12, 13, 14],
    total:   [16, 17, 18],
    grad:    [20, 21, 22, 23],
    update:  [25, 26, 27, 28],
    draw:    [30, 31, 32, 33, 34, 35, 36],
    loop:    [38, 39, 40, 41],
    step4:   [ 46,47, 4, 5],
    step5:   [49, 50, 7,8,9,10],
    step6:   [ 52, 53, 12, 13, 14,],
    step7:   [55, 56],
    step8:   [58,59, 27,28,29,30,31],
    step9:   [61, 62],
    step10:  [64, 65],
    done:    [67],
};

const MATH_MAP = {
    init:    'math-init',
    loop:    'math-init',
    step4:   'math-predict',
    step5:   'math-mse',
    step6:   'math-penalty',
    step7:   'math-total',
    step8:   'math-grad',
    step9:   'math-update',
    step10:  'math-draw',
    done:    'math-done',
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
    document.querySelectorAll('.highlighted-row').forEach(e=>e.classList.remove('highlighted-row'));
    document.querySelectorAll('.dimmed-row').forEach(e=>e.classList.remove('dimmed-row'));
    if (!nums) return;
    const hl = new Set(nums.map(n=>`row-${n-1}`));
    document.querySelectorAll('#code-table tr').forEach(tr=>{
        tr.classList.add(hl.has(tr.id)?'highlighted-row':'dimmed-row');
    });
    document.getElementById(`row-${nums[0]-1}`)?.scrollIntoView({block:'center',behavior:'smooth'});
}

function highlightMath(kind) {
    document.querySelectorAll('.math-section').forEach(e=>e.classList.remove('math-active'));
    const id = MATH_MAP[kind];
    if (!id) return;
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.add('math-active');
    const mv = document.getElementById('view-math');
    if (mv && !mv.classList.contains('hidden'))
        el.scrollIntoView({block:'nearest',behavior:'smooth'});
}

const fmt = (v,d=4) => Number(v).toFixed(d);

/* ══════════════════════════════════════════════════════════════════
   CORE ALGORITHM
   ══════════════════════════════════════════════════════════════════ */
let X_all=[], y_all=[], X_train=[], y_train=[], X_test=[], y_test=[];
let STEPS=[], cursor=-1, olsW=0, olsB=0;

function mseLoss(y, yp) {
    const n=y.length;
    return y.reduce((s,yi,i)=>s+(yp[i]-yi)**2,0)/n;
}
function predict(X,w,b){ return X.map(x=>w*x+b); }

function trainAll(X,y,lr,iters,lam,nSteps) {
    /* always iter 0 and 1, then spread */
    const checkAt=[0,1];
    const rem=Math.max(0,nSteps-2);
    for(let s=0;s<rem;s++){
        const idx=Math.round(2+(s/Math.max(rem-1,1))*(iters-3));
        checkAt.push(Math.min(idx,iters-1));
    }
    const checkSet=new Set(checkAt);

    let w=0,b=0;
    const snaps=[];
    for(let iter=0;iter<iters;iter++){
        const yp=predict(X,w,b);
        const mse=mseLoss(y,yp);
        const penalty=lam*w*w;
        const total=mse+penalty;
        const n=X.length;
        let dw=0,db=0;
        X.forEach((xi,i)=>{ dw+=(yp[i]-y[i])*xi; db+=yp[i]-y[i]; });
        dw=2/n*dw + 2*lam*w;
        db=2/n*db;
        if(checkSet.has(iter)){
            snaps.push({iter,w,b,mse,penalty,total,dw,db,
                wNext:w-lr*dw, bNext:b-lr*db, yp:[...yp]});
        }
        w-=lr*dw; b-=lr*db;
    }
    return snaps;
}

function trainOLS(X,y,lr,iters){
    let w=0,b=0;
    for(let i=0;i<iters;i++){
        const yp=predict(X,w,b), n=X.length;
        let dw=0,db=0;
        X.forEach((xi,j)=>{ dw+=(yp[j]-y[j])*xi; db+=yp[j]-y[j]; });
        w-=lr*(2/n*dw); b-=lr*(2/n*db);
    }
    return {w,b};
}

/* ══════════════════════════════════════════════════════════════════
   BUILD STEPS
   ══════════════════════════════════════════════════════════════════ */
function buildSteps(snaps) {
    STEPS=[];
    const lr     = parseFloat(document.getElementById('lrInput').value)||0.01;
    const lam    = parseFloat(document.getElementById('lambdaInput').value)||0.1;
    const iters  = document.getElementById('iterInput').value;
    const n      = X_train.length;

    /* ── step 0: init ── */
    STEPS.push({
        kind:'init', lines:LINE.init,
        title:'Initialise — Ridge Regression',
        w:0, b:0, mse:null, penalty:null, total:null, curves:[],
        detail:
            `Train samples  : ${n}\n` +
            `Test samples   : ${X_test.length}\n` +
            `Learning rate α: ${lr}\n` +
            `Iterations     : ${iters}\n` +
            `λ (lambda)     : ${lam}\n\n` +
            `w = 0.0,  b = 0.0\n\n` +
            `Ridge adds L2 penalty = λ·w²\n` +
            `to the MSE loss to shrink w toward zero.`
    });

    const curvesAccum=[];

    snaps.forEach((s,si)=>{
        const iterLabel = si===0 ? 'Iteration 0'
                        : si===snaps.length-1 ? `Iteration ${s.iter} (final)`
                        : `Iteration ${s.iter}`;
        const showFull = si<=1;

        /* push current curve at start of iteration */
        curvesAccum.push({w:s.w, b:s.b, label:iterLabel});

        /* Step 4 — predict */
        STEPS.push({
            kind:'step4', lines:LINE.step4,
            title:`Step 4 — Predict  [${iterLabel}]`,
            w:s.w, b:s.b, mse:null, penalty:null, total:null,
            curves:[...curvesAccum],
            detail:
                `y_pred = predict(X, w, b) = w·X + b\n\n` +
                `w = ${fmt(s.w,6)}   b = ${fmt(s.b,6)}\n\n` +
                X_train.map((x,i)=>
                    `  ŷ[${i}] = ${fmt(s.w,4)}×${fmt(x,2)} + (${fmt(s.b,4)}) = ${fmt(s.yp[i],6)}`
                ).join('\n')
        });

        /* Step 5 — MSE */
        STEPS.push({
            kind:'step5', lines:LINE.step5,
            title:`Step 5 — MSE Loss  [${iterLabel}]`,
            w:s.w, b:s.b, mse:s.mse, penalty:null, total:null,
            curves:[...curvesAccum],
            detail:
                `mse = (1/n) Σ (ŷᵢ − yᵢ)²\n\n` +
                X_train.map((x,i)=>{
                    const diff=s.yp[i]-y_train[i];
                    return `  [${i}] ŷ=${fmt(s.yp[i],4)}  y=${fmt(y_train[i],4)}` +
                           `  diff²=${fmt(diff*diff,6)}`;
                }).join('\n') +
                `\n\nMSE = ${fmt(s.mse,8)}`
        });

        /* Step 6 — L2 penalty */
        STEPS.push({
            kind:'step6', lines:LINE.step6,
            title:`Step 6 — L2 Penalty  [${iterLabel}]`,
            w:s.w, b:s.b, mse:s.mse, penalty:s.penalty, total:null,
            curves:[...curvesAccum],
            detail:
                `penalty = λ · w²\n\n` +
                `λ      = ${lam}\n` +
                `w      = ${fmt(s.w,6)}\n` +
                `w²     = ${fmt(s.w*s.w,8)}\n\n` +
                `Penalty = ${lam} × ${fmt(s.w*s.w,8)}\n` +
                `        = ${fmt(s.penalty,8)}\n\n` +
                `Note: penalty grows with w² — larger w gets\n` +
                `penalised more heavily, shrinking it toward zero.`
        });

        /* Step 7 — total loss */
        STEPS.push({
            kind:'step7', lines:LINE.step7,
            title:`Step 7 — Total Ridge Loss  [${iterLabel}]`,
            w:s.w, b:s.b, mse:s.mse, penalty:s.penalty, total:s.total,
            curves:[...curvesAccum],
            detail:
                `loss = MSE + λ·w²\n\n` +
                `MSE     = ${fmt(s.mse,8)}\n` +
                `Penalty = ${fmt(s.penalty,8)}\n\n` +
                `Total L = ${fmt(s.mse,8)}\n` +
                `        + ${fmt(s.penalty,8)}\n` +
                `        = ${fmt(s.total,8)}\n\n` +
                `Gradient descent will minimise this combined loss.`
        });

        /* Step 8 — gradients */
        STEPS.push({
            kind:'step8', lines:LINE.step8,
            title:`Step 8 — Gradients  [${iterLabel}]`,
            w:s.w, b:s.b, mse:s.mse, penalty:s.penalty, total:s.total,
            curves:[...curvesAccum],
            detail:
                `dw = (2/n) Σ(ŷ−y)·x  +  2λw\n\n` +
                X_train.map((x,i)=>{
                    const diff=s.yp[i]-y_train[i];
                    return `  [${i}] (${fmt(diff,4)})×${fmt(x,4)} = ${fmt(diff*x,6)}`;
                }).join('\n') +
                `\n\n  MSE grad term = ${fmt(2/n*X_train.reduce((a,x,i)=>a+(s.yp[i]-y_train[i])*x,0),8)}\n` +
                `  L2  grad term = 2×${lam}×${fmt(s.w,6)} = ${fmt(2*lam*s.w,8)}\n\n` +
                `dw = ${fmt(s.dw,8)}\ndb = ${fmt(s.db,8)}`
        });

        /* Step 9 — update */
        curvesAccum[curvesAccum.length-1]={w:s.wNext, b:s.bNext, label:iterLabel};
        STEPS.push({
            kind:'step9', lines:LINE.step9,
            title:`Step 9 — Update w, b  [${iterLabel}]`,
            w:s.wNext, b:s.bNext, mse:s.mse, penalty:s.penalty, total:s.total,
            curves:[...curvesAccum],
            detail:
                `w ← w − α·dw\n` +
                `  = ${fmt(s.w,6)} − ${lr}×(${fmt(s.dw,8)})\n` +
                `  = ${fmt(s.wNext,6)}\n\n` +
                `b ← b − α·db\n` +
                `  = ${fmt(s.b,6)} − ${lr}×(${fmt(s.db,8)})\n` +
                `  = ${fmt(s.bNext,6)}\n\n` +
                `Weight shrank from ${fmt(s.w,6)} → ${fmt(s.wNext,6)}\n` +
                `(L2 penalty pulled it toward zero)`
        });

        /* Step 10 — draw line */
        STEPS.push({
            kind:'step10', lines:LINE.step10,
            title:`Step 10 — draw_regression_line()  [${iterLabel}]`,
            w:s.wNext, b:s.bNext, mse:s.mse, penalty:s.penalty, total:s.total,
            curves:[...curvesAccum],
            detail:
                `draw_regression_line(X, w, b)\n\n` +
                `w = ${fmt(s.wNext,6)}\n` +
                `b = ${fmt(s.bNext,6)}\n\n` +
                `x_plot = np.linspace(${fmt(Math.min(...X_all)-1,2)}, ${fmt(Math.max(...X_all)+1,2)}, 300)\n` +
                `y_plot = ${fmt(s.wNext,4)} × x_plot + (${fmt(s.bNext,4)})\n\n` +
                `← Regression line redrawn on canvas`
        });
    });

    /* final step */
    const finalSnap = snaps[snaps.length-1];
    const yTestPred = predict(X_test, finalSnap.wNext, finalSnap.bNext);
    const mseFinal  = mseLoss(y_test, yTestPred);
    const ssTot = y_test.reduce((s,y)=>s+(y-y_test.reduce((a,b)=>a+b,0)/y_test.length)**2,0);
    const ssRes = y_test.reduce((s,y,i)=>s+(y-yTestPred[i])**2,0);
    const r2    = 1 - ssRes/ssTot;

    STEPS.push({
        kind:'done', lines:LINE.done,
        title:'Step 9 — Training complete → Evaluate',
        w:finalSnap.wNext, b:finalSnap.bNext,
        mse:mseFinal, penalty:null, total:null,
        curves:[...curvesAccum,{w:finalSnap.wNext,b:finalSnap.bNext,label:'Final'}],
        detail:
            `Training complete!\n\n` +
            `Final w = ${fmt(finalSnap.wNext,6)}\n` +
            `Final b = ${fmt(finalSnap.bNext,6)}\n\n` +
            `Test MSE  = ${fmt(mseFinal,6)}\n` +
            `Test RMSE = ${fmt(Math.sqrt(mseFinal),6)}\n` +
            `R² score  = ${fmt(r2,6)}\n\n` +
            `Compare to OLS (λ=0):\n` +
            `  OLS  w = ${fmt(olsW,6)}\n` +
            `  Ridge w = ${fmt(finalSnap.wNext,6)}\n` +
            `  |Ridge w| < |OLS w| → shrinkage applied ✓`
    });
}

/* ══════════════════════════════════════════════════════════════════
   CANVAS DRAWING
   ══════════════════════════════════════════════════════════════════ */
const PALETTE=['#1d4ed8','#7c3aed','#047857','#b45309','#be185d','#0369a1','#b91c1c','#15803d'];

function drawCanvas(curves, currentW, currentB) {
    const canvas=document.getElementById('reg-canvas');
    const W=canvas.offsetWidth||480, H=canvas.offsetHeight||270;
    canvas.width=W; canvas.height=H;
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);

    if(!X_all.length) return;

    const pad=36;
    const xMin=Math.min(...X_all)-1, xMax=Math.max(...X_all)+1;
    const yMin=Math.min(...y_all)-2, yMax=Math.max(...y_all)+2;
    const px=x=>pad+(x-xMin)/(xMax-xMin)*(W-2*pad);
    const py=y=>H-pad-(y-yMin)/(yMax-yMin)*(H-2*pad);

    /* grid */
    ctx.setLineDash([2,3]); ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=0.6;
    for(let i=0;i<=4;i++){
        const gx=pad+i*(W-2*pad)/4, gy=(H-pad)-i*(H-2*pad)/4;
        ctx.beginPath(); ctx.moveTo(gx,pad); ctx.lineTo(gx,H-pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
    }
    ctx.setLineDash([]);

    /* axes */
    ctx.strokeStyle='#94a3b8'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
    ctx.fillStyle='#64748b'; ctx.font='9px sans-serif';
    ctx.fillText('x',W-pad+4,H-pad+3); ctx.fillText('y',pad-10,pad+3);

    /* OLS baseline */
    if(olsW!==0||olsB!==0){
        ctx.beginPath(); ctx.setLineDash([5,4]);
        ctx.strokeStyle='#94a3b8'; ctx.lineWidth=1.2;
        const x0=xMin, x1=xMax;
        ctx.moveTo(px(x0),py(olsW*x0+olsB)); ctx.lineTo(px(x1),py(olsW*x1+olsB));
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle='#94a3b8'; ctx.font='8px sans-serif';
        ctx.fillText('OLS',px(xMin)+4,py(olsW*xMin+olsB)-5);
    }

    /* past curves */
    const pastCurves=curves.filter(c=>
        !(Math.abs(c.w-currentW)<1e-9&&Math.abs(c.b-currentB)<1e-9)
    );
    pastCurves.forEach((c,ci)=>{
        ctx.beginPath(); ctx.lineWidth=1.4; ctx.globalAlpha=0.65;
        ctx.strokeStyle=PALETTE[ci%PALETTE.length];
        for(let i=0;i<=200;i++){
            const x=xMin+i*(xMax-xMin)/200;
            const y=c.w*x+c.b;
            i===0?ctx.moveTo(px(x),py(y)):ctx.lineTo(px(x),py(y));
        }
        ctx.stroke(); ctx.globalAlpha=1;
    });

    /* current bold ridge line */
    ctx.beginPath(); ctx.lineWidth=2.2;
    ctx.strokeStyle='#7c3aed';
    for(let i=0;i<=200;i++){
        const x=xMin+i*(xMax-xMin)/200;
        const y=currentW*x+currentB;
        i===0?ctx.moveTo(px(x),py(y)):ctx.lineTo(px(x),py(y));
    }
    ctx.stroke();

    /* data points */
    const splitIdx=Math.floor(X_all.length*parseInt(document.getElementById('splitInput').value)/100);
    X_all.forEach((x,i)=>{
        ctx.beginPath(); ctx.arc(px(x),py(y_all[i]),4,0,2*Math.PI);
        ctx.fillStyle = i<splitIdx ? '#0f172a' : '#7c3aed';
        ctx.strokeStyle='#fff'; ctx.lineWidth=1;
        ctx.fill(); ctx.stroke();
    });

    /* legend */
    ctx.font='8.5px Consolas,monospace';
    ctx.fillStyle='#0f172a'; ctx.fillRect(W-130,8,122,28);
    ctx.fillStyle='#0f172a'; ctx.strokeStyle='#1e293b'; ctx.strokeRect(W-130,8,122,28);
    ctx.fillStyle='#fff'; ctx.fillText(`w=${fmt(currentW,4)}  b=${fmt(currentB,4)}`,W-125,22);
    ctx.fillStyle='#c4b5fd'; ctx.fillText('Ridge (current)',W-125,34);
}

/* loss chart */
let lossHistory=[];

function drawLossChart() {
    const canvas=document.getElementById('loss-canvas');
    const W=canvas.offsetWidth||480, H=canvas.offsetHeight||110;
    canvas.width=W; canvas.height=H;
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);

    if(!lossHistory.length) return;
    const pad=28;
    const n=lossHistory.length;
    const totals=lossHistory.map(l=>l.total);
    const mses   =lossHistory.map(l=>l.mse);
    const pens   =lossHistory.map(l=>l.penalty);
    const maxV=Math.max(...totals,0.01);

    const px=i=>pad+(i/(n-1||1))*(W-2*pad);
    const py=v=>H-pad-(v/maxV)*(H-2*pad);

    /* axes */
    ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=0.6;
    ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
    ctx.fillStyle='#94a3b8'; ctx.font='7px sans-serif';
    ctx.fillText(fmt(maxV,2),2,pad+3); ctx.fillText('0',2,H-pad+3);
    ctx.fillText('Iter',W-pad+2,H-pad+3);

    const drawLine=(data,col,lw=1.2)=>{
        ctx.beginPath(); ctx.strokeStyle=col; ctx.lineWidth=lw;
        data.forEach((v,i)=>{ i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v)); });
        ctx.stroke();
    };

    drawLine(mses,   '#f59e0b', 1.2);   /* amber = MSE    */
    drawLine(pens,   '#dc2626', 1.2);   /* red = penalty  */
    drawLine(totals, '#1e3a5f', 1.8);   /* navy = total   */

    /* mark current step */
    if(cursor>=0&&cursor<STEPS.length){
        const s=STEPS[cursor];
        if(s.total!=null){
            const idx=lossHistory.findIndex(l=>Math.abs(l.total-s.total)<1e-10);
            if(idx>=0){
                ctx.beginPath(); ctx.arc(px(idx),py(s.total),4,0,2*Math.PI);
                ctx.fillStyle='#1e3a5f'; ctx.fill();
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
   SHOW STEP
   ══════════════════════════════════════════════════════════════════ */
function show(i) {
    if(!STEPS.length) return;
    cursor=Math.max(0,Math.min(i,STEPS.length-1));
    const s=STEPS[cursor];

    highlight(s.lines);
    highlightMath(s.kind);
    document.getElementById('step-counter').textContent=`Step ${cursor+1} / ${STEPS.length}`;
    document.getElementById('calc').innerHTML=
        `<span class="calc-title">${s.title}</span>`+
        `<span class="calc-body">${s.detail.replace(/</g,'&lt;')}</span>`;

    document.getElementById('weightDisplay').value  = fmt(s.w,6);
    document.getElementById('biasDisplay').value    = fmt(s.b,6);
    document.getElementById('mseDisplay').value     = s.mse!=null   ? fmt(s.mse,6)    : '—';
    document.getElementById('penaltyDisplay').value = s.penalty!=null? fmt(s.penalty,6): '—';
    document.getElementById('totalDisplay').value   = s.total!=null  ? fmt(s.total,6)  : '—';

    drawCanvas(s.curves||[], s.w, s.b);
    drawLossChart();
}

/* ══════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════ */
function runRidge() {
    if(!X_all.length){alert('Load a dataset first.');return;}
    const lr      = parseFloat(document.getElementById('lrInput').value)||0.01;
    const iters   = parseInt(document.getElementById('iterInput').value)||1000;
    const lam     = parseFloat(document.getElementById('lambdaInput').value)||0.1;
    const nSteps  = parseInt(document.getElementById('stepsInput').value)||8;
    const splitPct= parseInt(document.getElementById('splitInput').value)||80;
    const splitIdx= Math.floor(X_all.length*splitPct/100);

    X_train=X_all.slice(0,splitIdx); y_train=y_all.slice(0,splitIdx);
    X_test =X_all.slice(splitIdx);   y_test =y_all.slice(splitIdx);

    /* OLS baseline */
    const ols=trainOLS(X_train,y_train,lr,iters);
    olsW=ols.w; olsB=ols.b;

    /* build loss history for chart (all iterations) */
    lossHistory=[];
    let w=0,b=0;
    for(let iter=0;iter<iters;iter++){
        const yp=predict(X_train,w,b);
        const mse=mseLoss(y_train,yp);
        const penalty=lam*w*w;
        const total=mse+penalty;
        const n=X_train.length;
        let dw=0,db=0;
        X_train.forEach((xi,i)=>{dw+=(yp[i]-y_train[i])*xi; db+=yp[i]-y_train[i];});
        dw=2/n*dw+2*lam*w; db=2/n*db;
        lossHistory.push({mse,penalty,total});
        w-=lr*dw; b-=lr*db;
    }

    /* build step trace */
    const snaps=trainAll(X_train,y_train,lr,iters,lam,nSteps);
    buildSteps(snaps);

    /* render metrics */
    const finalSnap=snaps[snaps.length-1];
    const yTp=predict(X_test,finalSnap.wNext,finalSnap.bNext);
    const mseFinal=mseLoss(y_test,yTp);
    const ssTot=y_test.reduce((s,y)=>s+(y-y_test.reduce((a,b)=>a+b,0)/y_test.length)**2,0);
    const ssRes=y_test.reduce((s,y,i)=>s+(y-yTp[i])**2,0);
    const r2=1-ssRes/ssTot;
    const card=(l,v,c)=>
        `<div class="metric-card" style="border-color:${c};background:${c}15;display:flex;align-items:center;justify-content:space-between;gap:6px;padding:4px 10px;min-width:0;">
           <span class="mc-label" style="color:${c};margin-bottom:0;">${l}</span>
           <span class="mc-value" style="color:${c};font-size:13px;">${v}</span>
         </div>`;
    document.getElementById('metrics').innerHTML=
        card('MSE',          fmt(mseFinal,4),        '#f59e0b')+
        card('RMSE',         fmt(Math.sqrt(mseFinal),4),'#d97706')+
        card('R² Score',     fmt(r2,4),              '#059669')+
        card('λ (Lambda)',   document.getElementById('lambdaInput').value, '#7c3aed')+
        card('Train N',      X_train.length,         '#0891b2')+
        card('Test N',       X_test.length,          '#64748b')+
        card('OLS w',        fmt(olsW,4),            '#94a3b8')+
        card('Ridge w',      fmt(finalSnap.wNext,4), '#7c3aed');

    show(0);
}

document.getElementById('startBtn').onclick=runRidge;
document.getElementById('nextBtn').onclick =()=>{if(STEPS.length)show(cursor+1);};
document.getElementById('prevBtn').onclick =()=>{if(STEPS.length)show(cursor-1);};
document.getElementById('endBtn').onclick  =()=>{if(!STEPS.length)runRidge();else show(STEPS.length-1);};

/* ══════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════ */
function loadData(rows) {
    if(!rows.length){alert('Empty dataset');return;}
    const keys=Object.keys(rows[0]);
    X_all=rows.map(r=>parseFloat(r[keys[0]])).filter(v=>!isNaN(v));
    y_all=rows.map(r=>parseFloat(r[keys[1]])).filter(v=>!isNaN(v));
    if(X_all.length<4){alert('Need at least 4 rows.');return;}

    document.getElementById('thead').innerHTML=`<tr><th>${keys[0]}</th><th>${keys[1]}</th></tr>`;
    document.getElementById('tbody').innerHTML=
        rows.map(r=>`<tr><td>${parseFloat(r[keys[0]]).toFixed(2)}</td>
                         <td>${parseFloat(r[keys[1]]).toFixed(2)}</td></tr>`).join('');
    document.getElementById('table-box').classList.remove('hidden');

    STEPS=[]; cursor=-1; lossHistory=[];
    document.getElementById('calc').innerHTML=
        `<span style="color:#475569;">${X_all.length} pts loaded. Press <span style="color:#c4b5fd;">Start</span>.</span>`;
    document.getElementById('step-counter').textContent='Step 0 / 0';
    document.getElementById('metrics').innerHTML='';
    drawCanvas([],0,0); drawLossChart();
}

/* preset */
const PRESET=[
    {x:1,y:2.1},{x:2,y:3.9},{x:3,y:5.8},{x:4,y:8.2},{x:5,y:10.1},
    {x:6,y:12.5},{x:7,y:13.8},{x:8,y:16.3},{x:9,y:18.7},{x:10,y:20.5},
    {x:1.5,y:3.2},{x:2.5,y:5.1},{x:3.5,y:6.9},{x:4.5,y:9.4},{x:5.5,y:11.2},
    {x:6.5,y:13.1},{x:7.5,y:15.4},{x:8.5,y:17.5},{x:9.5,y:19.8},{x:11,y:22.4},
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
    const data=Array.from({length:20},(_,i)=>{
        const x=+(Math.random()*10+1).toFixed(2);
        return {x, y:+(2.1*x+1.5+(Math.random()-0.5)*3).toFixed(2)};
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
        if (STEPS.length) show(cursor + 1);
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault(); flashBtn('prevBtn');
        if (STEPS.length) show(cursor - 1);
    } else if (e.key === 'Home') {
        e.preventDefault(); flashBtn('startBtn');
        if (STEPS.length) show(0);
    } else if (e.key === 'End') {
        e.preventDefault(); flashBtn('endBtn');
        if (STEPS.length) show(STEPS.length - 1);
    }
});