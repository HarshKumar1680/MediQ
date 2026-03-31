/* MedIR app.js - Cache busted, no parallel requests */
(function(){
'use strict';

const ALGO = {
  bm25:  {label:'BM25',           color:'#3b82f6', year:1994},
  bm25p: {label:'BM25+',          color:'#8b5cf6', year:2011},
  vsm:   {label:'VSM Cosine',     color:'#10b981', year:2008},
  lm:    {label:'LM JM',          color:'#f59e0b', year:1998},
  tfidf: {label:'TF-IDF',         color:'#ef4444', year:1972},
};

let activeMethod  = 'bm25';
let activeAlgoTab = 'bm25';

const $  = id => document.getElementById(id);
const $$ = s  => document.querySelectorAll(s);

/* ── Init ─────────────────────────────────────────────────────── */
fetch('/stats')
  .then(r => r.json())
  .then(s => {
    $('chip-docs').textContent  = s.total_docs + ' docs';
    $('chip-terms').textContent = Number(s.unique_terms).toLocaleString() + ' terms';
  })
  .catch(() => {
    $('chip-docs').textContent  = '40 docs';
    $('chip-terms').textContent = 'Ready';
  });

/* ── Search ───────────────────────────────────────────────────── */
async function doSearch(){
  const q = $('q').value.trim();
  if(!q){ $('q').focus(); return; }
  const topK = parseInt($('topk').value) || 10;

  setLoader(true, 'Step 1/2 — Running ' + (ALGO[activeMethod]?.label||activeMethod) + ' search…');

  try {
    // ── Step 1: single fast search ──────────────────────────────
    const r1 = await fetchWithTimeout('/search?q=' + encodeURIComponent(q) +
                           '&method=' + activeMethod + '&top_k=' + topK, 15000);
    if(!r1.ok) throw new Error('Server error ' + r1.status);
    const d1 = await r1.json();

    renderResults(d1);
    renderPipeline(d1);

    $('results-area').hidden = false;
    activateTab('results');
    $('results-area').scrollIntoView({behavior:'smooth', block:'start'});

    // Do not block UX on compare; run it in background.
    setLoader(false);
    runCompareInBackground(q, topK);

  } catch(err) {
    showErr(err.message);
    $('results-area').hidden = false;
  } finally {
    setLoader(false);
  }
}

/* ── Render Results ──────────────────────────────────────────── */
function renderResults(data){
  const m = ALGO[data.method] || {label:data.method, color:'#fff'};
  $('res-meta').innerHTML =
    'Found <strong>' + data.total_results + '</strong> docs for <strong>"' + esc(data.query) + '"</strong>' +
    ' <span style="background:' + m.color + '22;color:' + m.color + ';border:1px solid ' + m.color + '44;' +
    'padding:2px 8px;border-radius:4px;font-size:.7rem;margin-left:6px;font-family:var(--mono)">' + m.label + '</span>' +
    ' <span class="t-badge">&#9201; ' + data.time_ms + ' ms</span>';

  if(!data.results || !data.results.length){
    $('res-list').innerHTML = '<div class="no-res"><h3>No results</h3><p>Try different keywords.</p></div>';
    return;
  }
  const maxS = data.results[0].score || 1;
  $('res-list').innerHTML = data.results.map(function(r,i){
    var pct  = Math.max(4, Math.round((r.score / maxS) * 100));
    var toks = (r.tokens_matched||[]).map(function(t){
      return '<span class="rt">' + esc(t) + '</span>';
    }).join('');
    return '<div class="rc" style="animation-delay:' + (i*30) + 'ms">' +
      '<div class="rc-rank">' + r.rank + '</div>' +
      '<div>' +
        '<div class="rc-title">' + esc(r.title) + '</div>' +
        '<div class="rc-disease">' + esc(r.disease) + '</div>' +
        '<div class="rc-snippet">' + esc(r.snippet) + '</div>' +
        '<div class="rc-tokens">' + toks + '</div>' +
      '</div>' +
      '<div class="rc-score">' +
        '<div class="sc-label">Score</div>' +
        '<div class="sc-val">' + Number(r.score).toFixed(4) + '</div>' +
        '<div class="sc-bar"><div class="sc-fill" style="width:' + pct + '%"></div></div>' +
      '</div>' +
    '</div>';
  }).join('');
}

/* ── Render Compare ──────────────────────────────────────────── */
function renderCompare(data){
  var comp = data.comparison;
  if(!comp) return;

  var arr   = Object.values(comp);
  var times = arr.map(function(c){ return c.time_ms; });
  var minT  = Math.min.apply(null, times);
  var maxT  = Math.max.apply(null, times);
  var sorted = arr.slice().sort(function(a,b){ return a.time_ms - b.time_ms; });
  var medals = ['&#127945;','&#127946;','&#127947;'];

  $('timing-tbody').innerHTML = sorted.map(function(c, i){
    var isMin  = c.time_ms === minT;
    var isMax  = c.time_ms === maxT;
    var tvCls  = isMin ? 'time-val time-fastest' : isMax ? 'time-val time-slowest' : 'time-val';
    var barPct = maxT > 0 ? Math.round((c.time_ms / maxT) * 100) : 100;
    var medal  = medals[i] || (i+1);
    return '<tr>' +
      '<td style="font-family:var(--mono);color:var(--text3)">' + medal + '</td>' +
      '<td><span class="algo-dot" style="background:' + c.color + '"></span>' +
          '<span class="algo-name">' + esc(c.label) + '</span></td>' +
      '<td><span class="algo-year">' + c.year + '</span></td>' +
      '<td><span class="' + tvCls + '">' + Number(c.time_ms).toFixed(4) + ' ms' +
          (isMin?' &#9889;':isMax?' &#128034;':'') + '</span></td>' +
      '<td><div class="speed-outer"><div class="speed-inner" style="width:' +
          barPct + '%;background:' + c.color + '"></div></div></td>' +
      '<td><div class="top1-cell">' + esc(c.top1_title) + '</div></td>' +
      '<td><div class="algo-desc-cell">' + esc(c.desc) + '</div></td>' +
    '</tr>';
  }).join('');

  /* algo tab buttons */
  $('algo-tabs').innerHTML = arr.map(function(c){
    var active = c.method === activeAlgoTab;
    return '<button class="at' + (active?' active':'') + '" data-am="' + c.method + '" style="' +
      (active ? 'background:'+c.color+';border-color:'+c.color+';color:#fff'
              : 'border-color:'+c.color+'44;color:'+c.color) + '">' +
      esc(c.label) + '</button>';
  }).join('');

  $$('.at').forEach(function(btn){
    btn.addEventListener('click', function(){
      activeAlgoTab = btn.dataset.am;
      renderAlgoResults(comp);
      $$('.at').forEach(function(b){
        var mc = ALGO[b.dataset.am] || {color:'#fff'};
        b.classList.remove('active');
        b.style.background=''; b.style.color=mc.color; b.style.borderColor=mc.color+'44';
      });
      var ac = ALGO[activeAlgoTab] || {color:'#fff'};
      btn.classList.add('active');
      btn.style.background=ac.color; btn.style.color='#fff'; btn.style.borderColor=ac.color;
    });
  });
  renderAlgoResults(comp);
}

function renderAlgoResults(comp){
  var c = comp[activeAlgoTab];
  if(!c) return;
  $('algo-results').innerHTML = c.results.slice(0,5).map(function(r){
    return '<div class="mrc">' +
      '<div class="mrc-rank">' + r.rank + '</div>' +
      '<div style="flex:1;min-width:0">' +
        '<div class="mrc-title">' + esc(r.title) + '</div>' +
        '<div class="mrc-disease">' + esc(r.disease) + '</div>' +
        '<div class="mrc-snippet">' + esc(r.snippet) + '</div>' +
      '</div>' +
      '<div class="mrc-score">' + Number(r.score).toFixed(4) + '</div>' +
    '</div>';
  }).join('');
}

/* ── Render Pipeline ─────────────────────────────────────────── */
function renderPipeline(data){
  var chips    = (data.tokens||[]).map(function(t){ return '<span class="tok-chip hl">'+esc(t)+'</span>'; }).join('');
  var rawToks  = data.query.toLowerCase().replace(/[^a-z\s]/g,' ').trim().split(/\s+/);
  var rawChips = rawToks.map(function(t){ return '<span class="tok-chip">'+esc(t)+'</span>'; }).join('');
  var m        = ALGO[data.method] || {label: data.method};

  $('pipeline-wrap').innerHTML =
    '<div class="pipe-step"><div class="pipe-lbl">Step 1 &middot; Raw Query</div>' +
      '<div class="pipe-val">"' + esc(data.query) + '"</div></div>' +
    '<div class="pipe-step"><div class="pipe-lbl">Step 2 &middot; Lowercase + Strip noise</div>' +
      '<div class="pipe-val">"' + esc(data.query.toLowerCase().replace(/[^a-z\s]/g,' ').replace(/\s+/g,' ').trim()) + '"</div></div>' +
    '<div class="pipe-step"><div class="pipe-lbl">Step 3 &middot; Tokenize</div>' +
      '<div class="pipe-val">' + rawChips + '</div></div>' +
    '<div class="pipe-step"><div class="pipe-lbl">Step 4 &middot; Remove stopwords + Lemmatize</div>' +
      '<div class="pipe-val">' + chips + '</div></div>' +
    '<div class="pipe-step"><div class="pipe-lbl">Step 5 &middot; Selected Algorithm</div>' +
      '<div class="pipe-val"><span class="tok-chip hl">' + esc(m.label) + '</span></div></div>' +
    '<div class="pipe-step"><div class="pipe-lbl">Step 6 &middot; Documents retrieved</div>' +
      '<div class="pipe-val">' + data.total_results + ' documents matched</div></div>' +
    '<div class="pipe-step"><div class="pipe-lbl">Step 7 &middot; Query time</div>' +
      '<div class="pipe-val" style="color:var(--amber)">&#9201; ' + data.time_ms + ' ms</div></div>';
}

/* ── Evaluation ──────────────────────────────────────────────── */
async function runEval(method){
  setLoader(true, 'Evaluating ' + (ALGO[method]?.label||method) + ' on ground truth…');
  try{
    var r = await fetch('/evaluate?method=' + method);
    if(!r.ok) throw new Error('HTTP ' + r.status);
    var data = await r.json();
    renderEval(data, method);
  }catch(e){
    $('eval-output').innerHTML = '<p style="color:var(--red);font-family:var(--mono)">Error: ' + esc(e.message) + '</p>';
  }finally{
    setLoader(false);
  }
}

function renderEval(data, method){
  var m   = ALGO[method] || {label:method, color:'#fff'};
  var pct = (data.MAP * 100).toFixed(1);
  var rows = data.queries.map(function(q){
    return '<tr>' +
      '<td style="font-family:var(--serif)">' + esc(q.query) + '</td>' +
      '<td>' + mp(q['P@10']) + '</td>' +
      '<td>' + mp(q['R@10']) + '</td>' +
      '<td>' + mp(q['F1@10']) + '</td>' +
      '<td>' + mp(q['AP']) + '</td>' +
    '</tr>';
  }).join('');

  $('eval-output').innerHTML =
    '<div class="map-banner">' +
      '<div>' +
        '<div style="font-family:var(--mono);font-size:.65rem;color:var(--text3);margin-bottom:4px">MAP &mdash; ' + esc(m.label) + '</div>' +
        '<div class="map-big" style="color:' + m.color + '">' + data.MAP + '</div>' +
        '<div class="map-sub">' + pct + '% avg relevant docs retrieved</div>' +
      '</div>' +
      '<div class="map-desc">MAP is the gold-standard IR evaluation metric.<br>Score 1.0 = perfect ranking.</div>' +
    '</div>' +
    '<table class="eval-table">' +
      '<thead><tr><th>Query</th><th>P@10</th><th>R@10</th><th>F1@10</th><th>AP</th></tr></thead>' +
      '<tbody>' + rows + '</tbody>' +
    '</table>';
}

function mp(v){
  var c = v>=0.7?'mp-h':v>=0.4?'mp-m':'mp-l';
  return '<span class="mp ' + c + '">' + v + '</span>';
}

async function runCompareInBackground(q, topK){
  setLoader(true, 'Comparing all 5 algorithms timing…');
  try{
    const r = await fetchWithTimeout('/compare?q=' + encodeURIComponent(q) + '&top_k=' + topK, 10000);
    if(!r.ok) throw new Error('Compare error ' + r.status);
    const data = await r.json();
    renderCompare(data);
  } catch(e){
    // Keep primary search results visible even if compare fails/times out.
    console.warn('Compare request failed:', e);
  } finally{
    setLoader(false);
  }
}

/* ── Helpers ─────────────────────────────────────────────────── */
async function fetchWithTimeout(url, timeoutMs){
  var ctrl = new AbortController();
  var timer = setTimeout(function(){ ctrl.abort(); }, timeoutMs);
  try{
    return await fetch(url, {signal: ctrl.signal});
  } catch(err){
    if(err && err.name === 'AbortError'){
      throw new Error('Request timed out. Please try again.');
    }
    throw err;
  } finally{
    clearTimeout(timer);
  }
}

function activateTab(name){
  $$('.tb').forEach(function(b){ b.classList.toggle('active', b.dataset.tab===name); });
  $$('.tab-pane').forEach(function(p){ p.classList.toggle('active', p.id==='tab-'+name); });
}
function setLoader(on, msg){
  $('loader').hidden = !on;
  if(msg){ var sp = $('loader').querySelector('span'); if(sp) sp.textContent = msg; }
}
function showErr(msg){
  $('res-list').innerHTML =
    '<div class="no-res">' +
      '<h3>&#9888; Error</h3>' +
      '<p style="font-family:var(--mono);font-size:.78rem;color:var(--text3);margin-bottom:.8rem">' + esc(msg) + '</p>' +
      '<p style="font-size:.82rem">Check VS Code terminal for errors.<br>' +
      'Try: <code style="font-family:var(--mono)">python run.py</code></p>' +
    '</div>';
}
function esc(s){
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── Wire up events ──────────────────────────────────────────── */
$('search-btn').addEventListener('click', doSearch);
$('q').addEventListener('keydown', function(e){ if(e.key==='Enter') doSearch(); });

$$('.ap').forEach(function(btn){
  btn.addEventListener('click', function(){
    $$('.ap').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    activeMethod = btn.dataset.m;
  });
});

$$('.dq').forEach(function(pill){
  pill.addEventListener('click', function(){
    $('q').value = pill.dataset.q;
    doSearch();
  });
});

$$('.tb').forEach(function(btn){
  btn.addEventListener('click', function(){ activateTab(btn.dataset.tab); });
});

$$('.eval-run-btn').forEach(function(btn){
  btn.addEventListener('click', function(){ runEval(btn.dataset.m); });
});

})();
