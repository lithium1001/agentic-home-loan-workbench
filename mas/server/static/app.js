/* Agentic Home Loan Workbench — frontend controller.
   Loads live case data from /api/*, drives the chat over SSE, renders per-agent
   thinking blocks, the HITL Approve/Revise gate, misroute prompts, and the
   activity log. Replaces the mockup's hardcoded CUSTOMERS/DOCS/BASE_CONVOS
   fixtures with real API calls. */
'use strict';

const STAGE_DOT = { done: '#22863A', active: '#D97706', pending: '#AEABA4' };

const state = {
  cases: [],
  caseData: null,       // /api/case/{id} payload
  selectedId: null,
  selectedStage: 'IPA',
  convos: {},           // "APP|STAGE" -> [message objects] for re-render on stage switch
  streaming: false,
  // While an RM is actively revising a draft, the normal composer is hidden so
  // revisions can only go through the revise input (which re-triggers the draft/
  // PDF flow). Set on Revise, cleared on Cancel or Approve; it PERSISTS across the
  // reject→redraft round-trip so the composer stays hidden until the RM approves.
  reviseMode: false,
};

const $ = (id) => document.getElementById(id);

// ── helpers ───────────────────────────────────────────────────────────────
function fmtMoney(n) {
  if (n == null) return '—';
  return 'S$' + Math.round(n).toLocaleString('en-SG');
}
function convoKey() { return state.selectedId + '|' + state.selectedStage; }
function getConvo() {
  const k = convoKey();
  if (!state.convos[k]) state.convos[k] = [];
  return state.convos[k];
}

async function getJSON(url) { const r = await fetch(url); return r.json(); }

// ── case switcher ───────────────────────────────────────────────────────────
async function loadCases() {
  const data = await getJSON('/api/cases');
  state.cases = data.cases || [];
  renderPicker();
  if (state.cases.length && !state.selectedId) selectCase(state.cases[0].id);
}

function renderPicker() {
  const list = $('pickerList');
  list.innerHTML = '';
  state.cases.forEach((c) => {
    const active = Object.keys(c.stageStatuses || {}).find((k) => c.stageStatuses[k] === 'active') || 'IPA';
    const dot = STAGE_DOT[c.stageStatuses[active]] || STAGE_DOT.pending;
    const row = document.createElement('div');
    row.className = 'picker-row' + (c.id === state.selectedId ? ' current' : '');
    row.innerHTML =
      `<span class="picker-dot" style="background:${dot}"></span>
       <span class="picker-info"><span class="nm">${c.name}</span><br>
       <span class="mt">${c.id} · ${c.stageShort}</span></span>`;
    row.onclick = () => { selectCase(c.id); $('casePicker').hidden = true; };
    list.appendChild(row);
  });
}

async function selectCase(id) {
  state.selectedId = id;
  state.caseData = await getJSON('/api/case/' + id);
  if (state.caseData.error) { alert(state.caseData.error); return; }
  state.selectedStage = state.caseData.activeStage || 'IPA';
  renderTopbar();
  renderWorkflowSwitch();
  renderStages();
  renderWorkspace();
  renderChips();
  renderChat();
}

// Repaint the whole center workspace (borrower strip + calculator card +
// assessment result card + income/docs/flags grid) — everything derived from
// /api/case plus the current lane's assessed state. The right pane is the chat;
// all structured information lives here in the center.
function renderWorkspace() {
  // The comparison tool replaces the case workspace rather than sitting inside
  // it: it is about a hypothetical loan, so showing this borrower's case cards
  // beside it would imply the figures came from their file.
  const compare = inCompareTrack();
  $('workspace').hidden = compare;
  $('compareWorkspace').hidden = !compare;
  if (compare) {
    renderContextBar();
    renderComparePage();
    return;
  }
  renderContextBar();
  renderBorrowerStrip();
  renderPlanner();
  renderIncomeGrid();
}

// ── assessed-state helpers (drive the v3 empty→filled behaviour) ────────────
// The Loan Scenario card fills ONLY for a turn that actually ran the calculator —
// an assessment flow OR a drafting flow (which assesses, then drafts). Signals:
// the server-tracked assessedStages, or (same turn, before /api/case is re-pulled)
// a message the stream tagged. An ad-hoc question that merely got an answer must
// NOT fill it: those numbers would come from the seed loanSummary, not from
// anything the RM asked to compute. Until then the card shows dashes.
function isAssessed() {
  const srv = (state.caseData && state.caseData.assessedStages) || {};
  if (srv[state.selectedStage]) return true;
  return getConvo().some((m) => m.kind === 'agent' && (m.assessment || m.computed));
}
// The body for the card's "Assistant's reasoning" footer = the most recent
// *assessment* answer in this lane. Deliberately ignores `computed` (drafting)
// messages: their body is the letter, not the reasoning behind the numbers. So a
// draft turn fills the card but leaves the earlier assessment's reasoning quoted.
function lastAgentAnswer() {
  const convo = getConvo();
  for (let i = convo.length - 1; i >= 0; i--) {
    if (convo[i].kind === 'agent' && convo[i].assessment && convo[i].text) return convo[i].text;
  }
  return '';
}

// Re-pull case data after a server-side progress change (e.g. an approved draft
// advanced the deal-progress milestones) and repaint ONLY the progress-bearing
// panels. Deliberately keeps the RM's current stage and chat lane untouched — the
// dots move on their own (guide, don't silently switch); whether to follow the
// newly-active stage is the RM's click. Chat is owned by the in-flight stream, so
// we never repaint it here.
async function refreshCaseData() {
  if (!state.selectedId) return;
  const [cd, cases] = await Promise.all([
    getJSON('/api/case/' + state.selectedId),
    getJSON('/api/cases'),
  ]);
  if (cd && !cd.error) state.caseData = cd;
  if (cases && cases.cases) state.cases = cases.cases;
  renderPicker();
  renderTopbar();
  renderWorkflowSwitch();
  renderStages();
  renderWorkspace();
  renderChips();
}

function renderTopbar() {
  const c = state.cases.find((x) => x.id === state.selectedId);
  const cd = state.caseData;
  $('currentName').textContent = cd ? cd.name : '—';
  const stageLabel = (cd.stages.find((s) => s.key === state.selectedStage) || {}).label || '';
  $('currentMeta').textContent = `${cd.id} · ${stageLabel}`;
}

// ── workflow switch (v3 segmented toggle) ──────────────────────────────────────
// v3 groups the new-purchase journey (IPA → Letter of Offer) into one workflow
// track and Reprice/Retention into the other. This segmented control at the top of
// the left rail is the top-level lane switch; the Deal Progress list below lets the
// RM pick the exact stage (IPA vs LO) within the IPA → LO track.
// COMPARE is a standalone what-if tool rather than a case stage: it has no
// server-side stage, no HITL gate and no letter, so it carries a synthetic stage
// key and swaps the workspace body instead of driving the case flow.
const WF_TRACKS = [
  { id: 'PURCHASE', label: 'IPA → LO', stages: ['IPA', 'LO'] },
  { id: 'REPRICE', label: 'Reprice', stages: ['REPRICE'] },
  { id: 'COMPARE', label: 'Package Comparison', stages: ['COMPARE'] },
];
function currentTrack() {
  return WF_TRACKS.find((t) => t.stages.includes(state.selectedStage)) || WF_TRACKS[0];
}
function inCompareTrack() { return state.selectedStage === 'COMPARE'; }
function renderWorkflowSwitch() {
  const el = $('wfSwitch');
  el.innerHTML = '';
  const active = currentTrack().id;
  WF_TRACKS.forEach((t) => {
    const b = document.createElement('button');
    b.className = 'wf-seg' + (t.id === active ? ' active' : '');
    b.textContent = t.label;
    b.onclick = () => {
      // Land on the track's active stage if it has one, else its first stage.
      const stages = state.caseData.stages;
      const activeStage = t.stages.find((k) =>
        (stages.find((s) => s.key === k) || {}).status === 'active');
      switchStage(activeStage || t.stages[0]);
    };
    el.appendChild(b);
  });
}

// ── stage path (top nav row) ─────────────────────────────────────────────────
// The server sends all three stages (IPA/LO/REPRICE) in one list; showing them
// all confuses the two journeys. Only render the CURRENT track's stages: the
// purchase track shows IPA + LO (renumbered 01/02), Reprice shows its single
// stage with no step number or connector — it is an advisory lane, not a journey.
function trackStages() {
  const keys = currentTrack().stages;
  // COMPARE has no server-side stage record — synthesise its single chip so the
  // path bar still renders (and stays consistent with the other tracks).
  if (inCompareTrack()) {
    return [{ key: 'COMPARE', label: 'Interest Savings', status: 'active',
              subtitle: 'Compare packages and early conversion' }];
  }
  return state.caseData.stages.filter((s) => keys.includes(s.key));
}
// Horizontal Salesforce-path-style stage chips; click one to switch. The stage
// subtitle lives in the chip's tooltip — the context bar carries the guidance.
function renderStages() {
  const ol = $('stages');
  ol.innerHTML = '';
  const stages = trackStages();
  stages.forEach((s, i) => {
    if (i) {
      const link = document.createElement('li');
      link.className = 'sb-connector';
      ol.appendChild(link);
    }
    const li = document.createElement('li');
    let cls = 'sb-item';
    if (s.status === 'done') cls += ' done';
    if (s.key === state.selectedStage) cls += ' active';
    li.className = cls;
    li.title = s.subtitle || '';
    li.innerHTML =
      `<span class="sb-dot">${s.status === 'done' ? '✓' : ''}</span>` +
      (stages.length > 1 ? `<span class="sb-num">${String(i + 1).padStart(2, '0')}</span>` : '') +
      `<span class="sb-label">${s.label}</span>`;
    li.onclick = () => switchStage(s.key);
    ol.appendChild(li);
  });
}

// Case header — one slim identity row (avatar + name + meta + CBS chip). All
// the case FACTS moved down into the categorised info cards; the header only
// answers "whose case am I looking at".
function renderBorrowerStrip() {
  const el = $('borrowerStrip');
  const b = state.caseData.borrower || {};
  el.innerHTML =
    `<div class="avatar">${escapeHtml(b.initials || '?')}</div>
     <div class="ch-main">
       <div class="ch-name">${escapeHtml(b.name || '—')}</div>
       <div class="ch-meta">${[b.age ? 'Age ' + b.age : '', b.citizenship, b.nric].filter(Boolean).map(escapeHtml).join(' · ')}</div>
     </div>
     ${b.cbs_score ? `<span class="bs-cbs">CBS ${b.cbs_score}${b.cbs_grade ? ' · ' + escapeHtml(b.cbs_grade) : ''}</span>` : ''}`;
}

// Case information — one slim full-width band per category (按行分开):
// Borrower Profile / Property / Income & Affordability / Documents / Case
// Flags. Inside a band the fields flow horizontally as label-over-value pairs
// (auto-fit grid), so nothing ever fights for one line or overflows — on a
// narrow window the fields simply wrap.
function renderIncomeGrid() {
  const el = $('wsGrid');
  const b = state.caseData.borrower || {};
  const p = state.caseData.property || {};
  const inc = state.caseData.income || {};
  const docs = (state.caseData.documents || {})[state.selectedStage] || [];
  const fv = (k, v, cls) => (v == null || v === '') ? '' :
    `<div class="fv ${cls || ''}"><div class="fv-k">${k}</div><div class="fv-v">${v}</div></div>`;
  const band = (title, body) =>
    `<div class="info-row"><div class="ic-title">${title}</div>${body}</div>`;
  const empty = (msg) => `<div class="chat-empty">${msg}</div>`;

  const borrowerBand = band('Borrower Profile', `<div class="fv-grid">
    ${fv('Citizenship', escapeHtml(b.citizenship || ''))}
    ${fv('Age', b.age)}
    ${fv('CBS score', b.cbs_score ? `${b.cbs_score}${b.cbs_grade ? ' · ' + escapeHtml(b.cbs_grade) : ''}` : null)}
    ${fv('Properties owned', b.n_props_owned)}
    ${fv('Outstanding loans', b.n_outstanding_loans === 0 ? 'None' : b.n_outstanding_loans, b.n_outstanding_loans === 0 ? 'ok' : '')}
    ${fv('CPF OA balance', b.cpf_oa != null ? fmtMoney(b.cpf_oa) : null)}
  </div>`);

  const propertyBand = band('Property', (p.price != null || p.type || p.detail)
    ? `<div class="fv-grid">
        ${fv('Purchase price', p.price != null ? fmtMoney(p.price) : null, 'em')}
        ${fv('Type', escapeHtml(p.type || ''))}
        ${fv('Address & detail', escapeHtml(p.detail || ''), 'wide')}
      </div>`
    : empty('No property on file.'));

  const pct = inc.tdsr_pct;
  const fillW = pct != null ? Math.min(100, (pct / 55) * 100) : 0;
  const incomeBand = band('Income &amp; Affordability · TDSR basis', Object.keys(inc).length
    ? `<div class="fv-grid">
        ${fv('Verified monthly income', fmtMoney(inc.qualifying_income), 'em')}
        ${fv('Fixed (declared)', fmtMoney(inc.fixed))}
        ${fv('Variable (declared)', fmtMoney(inc.variable))}
        ${inc.payslip_avg != null ? fv('Payslip avg gross (3 mo)', fmtMoney(inc.payslip_avg)) : ''}
        ${inc.noa_annual != null ? fv('NOA annual', fmtMoney(inc.noa_annual)) : ''}
        ${fv('Monthly commitments', fmtMoney(inc.monthly_commitments), inc.monthly_commitments ? '' : 'ok')}
        ${fv('TDSR ceiling (55%)', fmtMoney(inc.tdsr_ceiling) + ' / mo')}
      </div>
      <div class="tdsr-wrap">
        <div class="tdsr-bar"><div class="tdsr-fill ${pct != null && pct > 55 ? 'warn' : ''}" style="width:${fillW}%"></div></div>
        <div class="tdsr-scale"><span>0%</span><span>${pct != null ? 'Est. TDSR ' + pct + '%' : ''}</span><span>55% cap</span></div>
      </div>`
    : empty('No income data for this case.'));

  const done = docs.filter((d) => d.done).length;
  const docsBand = band(`Documents · ${docs.length ? `${done}/${docs.length}` : escapeHtml(currentStageMeta().label || '')}`,
    docs.length
      ? `<div class="doc-inline">` + docs.map((d) =>
          `<div class="doc-item ${d.done ? '' : 'miss'}">
             <span class="doc-tick ${d.done ? 'done' : 'miss'}">${d.done ? '✓' : ''}</span>
             <span class="nm">${escapeHtml(d.name)}</span></div>`).join('') + `</div>`
      : empty('No documents required at this stage.'));

  const flagsBand = band('Case Flags', `<div class="case-flags">${caseFlagsHtml()}</div>`);

  el.innerHTML = borrowerBand + propertyBand + incomeBand + docsBand + flagsBand;
}

// Auto-derived case flags (no agent call) — the third column of the ws grid.
function caseFlagsHtml() {
  const flags = state.caseData.caseFlags || [];
  if (!flags.length) return '<div class="chat-empty">No flags.</div>';
  return flags.map((f) => {
    const mark = f.status === 'ok' ? '✓' : (f.status === 'bad' ? '✕' : '!');
    return `<div class="flag ${f.status}">
      <div class="flag-left"><span class="flag-dot ${f.status}">${mark}</span>
        <span class="flag-label">${escapeHtml(f.label)}</span></div>
      <span class="flag-note">${escapeHtml(f.note || '')}</span></div>`;
  }).join('');
}

// The Loan Scenario calculator card — the workspace's focal point (v3). Before the
// assessment runs it shows dashes and a "Pending assessment" note; once the RM has
// run the assessment it fills with the numbers computed by the real calculate_loan
// tool (surfaced via /api/case's loanSummary) and appends the Monthly / TDSR /
// Eligibility summary strip, matching v3's empty→filled behaviour.
function renderPlanner() {
  const el = $('planner');
  const s = state.caseData.loanSummary;
  if (!s || s.error) {
    el.innerHTML = '<div class="planner-empty">No loan scenario for this case yet.</div>';
    return;
  }
  const assessed = isAssessed();
  const dash = (v) => (assessed ? v : '—');
  const kpi = (k, v, sub, muted) =>
    `<div class="kpi"><div class="kpi-k">${k}</div>` +
    `<div class="kpi-v ${muted ? 'muted' : ''}">${v}</div>` +
    (sub && assessed ? `<div class="kpi-sub">${sub}</div>` : '') + `</div>`;

  const rate = s.interest_rate_pct ? `${s.interest_rate_pct}% p.a.` : '3.85% p.a.';
  // After an RM adjustment (e.g. repriced to 1.2%) the card re-prices; flag it so
  // the RM knows the numbers moved because of their change, not the seed data.
  const headRight = s.adjusted
    ? `<span class="calc-rate adj">Adjusted · ${rate}</span>`
    : assessed
      ? `<span class="calc-rate">Indicative · ${rate}</span>`
      : `<span class="calc-pending">Pending assessment</span>`;

  // Summary strip (Monthly / TDSR / Eligibility) only after assessment, like v3.
  const inc = state.caseData.income || {};
  const tdsr = inc.tdsr_pct;
  const conditions = tdsr != null && tdsr > 55;
  const summaryStrip = assessed
    ? `<div class="calc-summary">
         <div class="cs-cell">
           <div class="cs-k">Monthly Repayment</div>
           <div class="cs-v navy">${fmtMoney(s.monthly_repayment)}</div>
           <div class="cs-sub">@ ${rate}, ${s.tenure_years ?? '—'}yr</div>
         </div>
         <div class="cs-cell">
           <div class="cs-k">TDSR</div>
           <div class="cs-v ${conditions ? 'amber' : 'green'}">${tdsr != null ? tdsr + '%' : '—'}</div>
           <div class="cs-sub ${conditions ? 'amber' : 'green'}">${tdsr != null ? (conditions ? 'Over 55% cap' : 'Under 55% cap ✓') : ''}</div>
         </div>
         <div class="cs-cell wide">
           <div class="cs-k">Eligibility</div>
           <div class="elig ${conditions ? 'warn' : 'ok'}">
             <span class="elig-mark">${conditions ? '⚠' : '✓'}</span>
             <span class="elig-txt">${conditions ? 'Eligible with conditions' : 'Eligible'}</span>
           </div>
         </div>
       </div>`
    : `<div class="calc-hint">Results will appear here once the assessment runs.</div>`;

  // Post-assessment footer: the assistant's own reasoning, collapsed by default.
  // The Approve / Adjust buttons were removed — Approve only re-fired the draft
  // action chip that already sits above the composer, and Adjust just focused the
  // chat input, so both were duplicate/dead-end controls. The card states the
  // verdict; the actions live in the Assistant pane.
  const reason = assessed ? lastAgentAnswer() : '';
  const footer = reason
    ? `<div class="calc-foot">
         <div class="ar-reason" id="arReason">
           <div class="ar-reason-head"><span>Assistant's reasoning</span><span class="ar-caret">▾</span></div>
           <div class="ar-reason-body md">${renderMarkdown(reason)}</div>
         </div>
       </div>`
    : '';

  el.innerHTML =
    `<div class="calc-card">
       <div class="calc-head">
         <span class="calc-title">Loan Scenario</span>
         ${headRight}
       </div>
       <div class="calc-kpis">
         ${kpi('LTV Cap', dash((s.ltv_pct ?? '—') + '%'), null, !assessed)}
         ${kpi('Loan Amount', dash(fmtMoney(s.loan_amount)), null, !assessed)}
         ${kpi('Property', fmtMoney(s.property_price))}
         ${kpi('Tenure', dash((s.tenure_years ?? '—') + ' yrs'), 'age-capped', !assessed)}
         ${kpi('Rate (p.a.)', rate.replace(' p.a.', ''))}
       </div>
       ${summaryStrip}
       ${footer}
     </div>`;

  // Wire the folded-in reasoning toggle (collapsed by default).
  const reasonEl = el.querySelector('#arReason');
  if (reasonEl) reasonEl.querySelector('.ar-reason-head').onclick = () => reasonEl.classList.toggle('open');
}

// Slim context bar under the workspace header (v3): "Step 2 of 3 · run the
// assessment…" before assessed, "✓ Assessment complete · …" after.
function renderContextBar() {
  const el = $('contextBar');
  if (!el) return;
  // The comparison tool is not part of a case's progress, so the usual
  // assessment guidance would be misleading here.
  if (inCompareTrack()) {
    el.className = 'context-bar';
    el.innerHTML =
      `<span class="cb-strong">Package Comparison</span><span class="cb-dot">·</span>` +
      `<span>Model what an existing loan saves by converting to a cheaper package, now or later. ` +
      `Figures are computed directly from the loan calculator — nothing here is written to the case.</span>`;
    return;
  }
  const assessed = isAssessed();
  const m = currentStageMeta();
  if (assessed) {
    const s = state.caseData.loanSummary || {};
    el.className = 'context-bar done';
    el.innerHTML =
      `<span class="cb-strong">✓ Assessment complete</span><span class="cb-dot">·</span>` +
      `<span>${escapeHtml(state.caseData.name || '')} — ${s.ltv_pct != null ? s.ltv_pct + '% LTV, ' + fmtMoney(s.loan_amount) : 'see the Loan Scenario card'}. ` +
      `Review the Loan Scenario card, then run the next action in the Assistant panel.</span>`;
  } else {
    el.className = 'context-bar';
    el.innerHTML =
      `<span class="cb-strong">${escapeHtml(m.label || '')}</span><span class="cb-dot">·</span>` +
      `<span>Pick an action in the Assistant panel — it will compute the LTV cap, stress-test TDSR, and flag any conditions.</span>`;
  }
}

// (The standalone Assessment Result card was removed — it duplicated the Loan
// Scenario KPIs. Its one unique part, the assistant's reasoning, is now folded
// into the bottom of the Loan Scenario card in renderPlanner.)

// ── workspace header / actions ─────────────────────────────────────────────────
// (The left-rail Documents panel is gone — the stage's checklist lives in the
// Documents band of the case-information rows; docs are shown exactly once.)
function currentStageMeta() { return state.caseData.stages.find((s) => s.key === state.selectedStage) || {}; }

// (The nav-row primary action button was removed — it duplicated the assistant's
// action chips and the Approve→draft flow. The single next-best action lives as
// the highlighted chip above the composer; the assessment verdict's Approve
// button lives in the Loan Scenario footer.)

// The stage's action chips, rendered above the composer in the Assistant pane.
// The next-best-action chip is highlighted (amber). Clicking one sends its
// label as an RM message — the guided entry point into the agent graph, so a
// new RM never has to invent a prompt.
function renderChips() {
  const el = $('chips');
  if (!el) return;
  // On the COMPARE track the chips are interpolated from the panel's live inputs
  // rather than served as fixed text — see compareChips(). Everywhere else they
  // come from the case payload.
  const chips = inCompareTrack()
    ? compareChips()
    : ((state.caseData.actionChips || {})[state.selectedStage] || []);
  el.innerHTML = '';
  chips.forEach((c) => {
    const label = typeof c === 'string' ? c : c.label;
    const primary = typeof c === 'object' && c.primary;
    // A chip may SEND something longer than it shows. The COMPARE chips read as a
    // short instruction ("What do I save converting now vs waiting?") but send every
    // figure the panel holds, so the agent never has to ask for an input that is
    // already on screen. What gets sent is what the RM sees in their chat bubble —
    // the prompt is not hidden from them, only from the button face.
    const prompt = (typeof c === 'object' && c.prompt) || label;
    const b = document.createElement('button');
    b.className = 'chip' + (primary ? ' primary' : '');
    b.textContent = label;
    b.title = prompt;
    b.onclick = () => sendMessage(prompt);
    el.appendChild(b);
  });
}

// ── Package Comparison (interest savings on an existing loan) ───────────────
// Deterministic end to end: inputs → /api/compare/savings → utils/calculator.
// The prose under the figures is TEMPLATED from the same response, not written
// by an LLM, so the narrative can never quote a number the calculation did not
// produce (and it is instant and free).
const CMP_FIELDS = [
  { id: 'cmpLoan',    label: 'Outstanding Loan (SGD)',            value: 1200000, step: 10000, min: 0 },
  { id: 'cmpRate',    label: 'Current Interest Rate (p.a., %)',   value: 2.00,    step: 0.05,  min: 0 },
  { id: 'cmpTenure',  label: 'Remaining Tenure (months)',         value: 360,     step: 12,    min: 1 },
  { id: 'cmpAfter',   label: 'No. of months for early conversion', value: 3,      step: 1,     min: 0 },
  { id: 'cmpRateA',   label: 'Scenario 1: Convert now to Rate A (p.a., %)',    value: 1.55, step: 0.05, min: 0 },
  { id: 'cmpRateB',   label: 'Scenario 2: Convert to Rate B (p.a., %)',        value: 1.50, step: 0.05, min: 0 },
];

function renderComparePage() {
  const host = $('compareWorkspace');
  if (host.dataset.built) { runCompare(); return; }   // keep the RM's inputs on re-render

  host.innerHTML =
    `<section class="calc-card cmp-card">
       <div class="calc-head">
         <span class="calc-title">Interest Savings — Package Comparison</span>
         <span class="calc-rate" id="cmpHorizon">—</span>
       </div>
       <div class="cmp-inputs">
         ${CMP_FIELDS.map((f) => `
           <div class="cmp-f">
             <label for="${f.id}">${f.label}</label>
             <div class="cmp-step">
               <button class="cmp-b" data-for="${f.id}" data-dir="-1">−</button>
               <input id="${f.id}" type="number" value="${f.value}" step="${f.step}" min="${f.min}" />
               <button class="cmp-b" data-for="${f.id}" data-dir="1">+</button>
             </div>
           </div>`).join('')}
       </div>
       <div class="cmp-heads" id="cmpHeads"></div>
       <div class="cmp-body" id="cmpBody"></div>
     </section>`;

  host.querySelectorAll('input').forEach((i) => i.addEventListener('input', runCompare));
  host.querySelectorAll('.cmp-b').forEach((b) => {
    b.onclick = () => {
      const el = $(b.dataset.for);
      const step = parseFloat(el.step || '1');
      const next = (parseFloat(el.value || '0') || 0) + step * parseInt(b.dataset.dir, 10);
      el.value = Math.max(parseFloat(el.min || '0'), parseFloat(next.toFixed(4)));
      runCompare();
    };
  });
  host.dataset.built = '1';
  runCompare();
}

function readCompare() {
  const num = (id) => parseFloat(($(id) || {}).value || '0') || 0;
  return {
    outstanding_loan: num('cmpLoan'),
    current_rate_pct: num('cmpRate'),
    remaining_months: Math.round(num('cmpTenure')),
    convert_after_months: Math.round(num('cmpAfter')),
    rate_a_pct: num('cmpRateA'),
    rate_b_pct: num('cmpRateB'),
  };
}

// The Assistant's action chips for this track: a SHORT label on the button, and a
// full prompt carrying every figure in the panel.
//
// The assistant cannot see this panel — it posts to /api/compare/savings, not into
// the chat's message history — so a chip saying "my two scenarios" gets back "I
// don't have any scenarios in our current conversation to compare" (verified
// live). The prompt therefore restates the terms, read from readCompare() so the
// sentence stays true after the RM changes a number.
//
// Label and prompt are split because those two jobs conflict: the button has room
// for a few words, while the prompt must state all six inputs. It states all six
// even where a question does not obviously need them (the rate-gap probe needs the
// rates to sweep from; the note needs the tenure to describe the horizon), because
// a missing figure costs a whole round-trip — the agent stops and asks the RM for a
// number that is already on screen, which is exactly what this panel exists to
// avoid. The RM still sees the full prompt: it is what appears in their chat bubble.
//
// Only the INPUTS are interpolated, never the computed savings. The assistant
// re-derives the figures by calling `interest_savings` — the same function this
// panel's endpoint calls — so its answer matches what is on screen and carries the
// usual tool-call audit trail. Pasting this panel's results into the prompt would
// reduce the agent to restating arithmetic it never checked.
//
// That tool is what makes these chips safe to ask. Before it existed (2026-08-04)
// the agent had no way to answer the panel's own question and did the sums in its
// head, quoting a monthly-instalment delta as the saving — off by ~S$11k. If the
// COMPARE track is ever routed to an agent without `interest_savings` in its
// allowlist, these chips go back to inviting that. See tests/test_interest_savings_tool.py.
function compareChips() {
  const i = readCompare();
  // Before the panel is built, and while the RM has a field cleared, readCompare()
  // reports zeros — "a S$0 loan at 0.00%" is not a question worth sending, so fall
  // back to the served chips until the terms describe a real loan.
  const served = (state.caseData.actionChips || {}).COMPARE || [];
  if (!(i.outstanding_loan > 0 && i.current_rate_pct > 0 && i.remaining_months > 0)) {
    return served;
  }
  const pct = (v) => `${Number(v).toFixed(2)}%`;
  // Every input the panel holds, phrased once and reused by each prompt below.
  const terms =
    `an existing loan with an outstanding balance of ${fmtMoney(i.outstanding_loan)}, `
    + `a current rate of ${pct(i.current_rate_pct)} p.a. and ${i.remaining_months} months `
    + `of tenure remaining. Scenario 1 is converting now to ${pct(i.rate_a_pct)} p.a.; `
    + `scenario 2 is staying on the current rate for ${i.convert_after_months} months `
    + `and then converting to ${pct(i.rate_b_pct)} p.a.`;
  return [
    { label: 'What do I save — convert now vs wait?',
      prompt: `I have ${terms} Which scenario saves more interest, and by how much?` },
    { label: 'At what rate gap does waiting win?',
      prompt: `I have ${terms} At what rate would scenario 2 have to be for waiting to `
            + `beat converting now? Show me where the comparison flips.` },
    { label: 'Why does converting early save more?',
      prompt: `I have ${terms} Explain in plain English why converting earlier saves `
            + `more interest, using these figures.` },
    { label: 'Draft a note for the customer',
      prompt: `I have ${terms} Write a short note I can send the customer explaining `
            + `both scenarios and which one you recommend.` },
  ];
}

let _cmpSeq = 0;
async function runCompare() {
  const seq = ++_cmpSeq;
  // Every input edit funnels through here, so this is where the chips are kept in
  // step with the panel. Done before the await: the chips depend only on the
  // inputs, so they must not wait on (or be skipped by) the calculator round-trip.
  renderChips();
  try {
    const res = await (await fetch('/api/compare/savings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readCompare()),
    })).json();
    if (seq !== _cmpSeq) return;          // a newer keystroke already superseded this
    renderCompareResult(res);
  } catch (e) {
    if (seq === _cmpSeq) $('cmpBody').innerHTML =
      `<div class="planner-empty">Could not compute: ${escapeHtml(e.message)}</div>`;
  }
}

function renderCompareResult(res) {
  const scenarios = (res && res.scenarios) || [];
  if (!scenarios.length) {
    $('cmpHeads').innerHTML = '';
    $('cmpBody').innerHTML =
      `<div class="planner-empty">${escapeHtml((res && res.error) || 'Enter a loan amount, rate and remaining tenure.')}</div>`;
    $('cmpHorizon').textContent = '—';
    return;
  }
  const inp = res.inputs || {};
  const H = inp.horizon_months;
  $('cmpHorizon').textContent = `Compared over ${H} months`;

  // Headline figures, one per scenario.
  $('cmpHeads').innerHTML = scenarios.map((s) => {
    const loss = s.savings < 0;
    const sub = s.id === 1
      ? `next ${s.first_phase_months}m + subsequent ${s.second_phase_months}m`
      : `next ${H} months`;
    return `<div class="cmp-head-cell">
        <div class="cmp-h-k">Scenario ${s.id}: ${loss ? 'Extra cost' : 'Savings'} (${sub})</div>
        <div class="cmp-h-v ${loss ? 'amber' : 'green'}">${fmtMoney(Math.abs(s.savings))}</div>
      </div>`;
  }).join('');

  $('cmpBody').innerHTML =
    `<h3 class="cmp-sec">Savings for Each Scenario</h3>` +
    scenarios.map((s) => compareNarrative(s, res)).join('') +
    compareTakeaway(scenarios, res);
}

// Templated prose — every number below comes from the response above.
function compareNarrative(s, res) {
  const inp = res.inputs || {};
  const base = (res.baseline || {}).rate_pct;
  const pct = (v) => `${Number(v).toFixed(2)}%`;
  const H = inp.horizon_months;
  const loss = s.savings < 0;

  if (s.id === 1) {
    return `<div class="cmp-scen">
      <h4>Scenario 1 — Convert now to Rate A</h4>
      <ul>
        <li><strong>Next ${s.first_phase_months} months ${loss ? 'extra cost' : 'savings'}: ${fmtMoney(Math.abs(s.savings_first_phase))}</strong>
          <ul>
            <li>You immediately replace ${pct(base)} with ${pct(s.rate_pct)} for months 1–${s.first_phase_months}.</li>
            <li>${loss ? 'Rate A is higher than what you pay today, so converting now costs more.'
                       : 'Savings come from paying a lower interest rate on a higher outstanding balance early on.'}</li>
            <li><em>Rule of thumb (month 1 interest delta)</em>: about ${fmtMoney(Math.abs(s.month_1_interest_delta))}
                <em>(illustrative; the totals use an amortisation simulation)</em>.</li>
          </ul>
        </li>
        <li><strong>Subsequent ${s.second_phase_months} months (months ${s.first_phase_months + 1}–${H}): ${fmtMoney(Math.abs(s.savings_second_phase))}</strong>
          <ul>
            <li>Rate remains ${pct(s.rate_pct)} while the baseline stays at ${pct(base)}.</li>
            <li>Even though the outstanding balance reduces over time, the rate gap continues to generate ${loss ? 'the difference' : 'savings'}.</li>
          </ul>
        </li>
      </ul>
    </div>`;
  }
  return `<div class="cmp-scen">
      <h4>Scenario 2 — Convert in ${s.convert_after_months} months to Rate B</h4>
      <ul>
        <li><strong>${loss ? 'Extra cost' : 'Savings'} over next ${H} months (months 1–${H}): ${fmtMoney(Math.abs(s.savings))}</strong>
          <ul>
            <li>Months 1–${s.convert_after_months}: you remain on ${pct(base)} → no change versus the baseline during this period.</li>
            <li>Months ${s.convert_after_months + 1}–${H}: you switch to ${pct(s.rate_pct)} → the difference begins after month ${s.convert_after_months}.</li>
            <li><em>Rule of thumb</em>: the driver is the rate gap between ${pct(base)} and ${pct(s.rate_pct)}, applied to the balance still outstanding after ${s.convert_after_months} months.</li>
          </ul>
        </li>
      </ul>
    </div>`;
}

function compareTakeaway(scenarios, res) {
  const s1 = scenarios.find((s) => s.id === 1);
  const s2 = scenarios.find((s) => s.id === 2);
  const lines = [];
  if (s1 && s2) {
    const better = s1.savings >= s2.savings ? s1 : s2;
    const gap = Math.abs(s1.savings - s2.savings);
    lines.push(`On these inputs, <strong>Scenario ${better.id}</strong> is ahead by ${fmtMoney(gap)} over the comparison window.`);
    if (s1.savings >= s2.savings) {
      lines.push('Converting earlier usually wins, because the balance interest is charged on is highest early on.');
    } else {
      lines.push('Waiting wins here because Rate B is enough below Rate A to outweigh the months spent on the current rate.');
    }
  }
  lines.push('Scenario 2 can still make sense if timing constrains you — a lock-in period, or a package that only becomes available later.');
  if (s1 && s1.savings < 0 && s2 && s2.savings < 0) {
    lines.push('Both scenarios cost more than staying put: neither quoted rate beats the current one.');
  }
  return `<div class="cmp-take"><h4>Quick takeaway</h4><ul>${
    lines.map((l) => `<li>${l}</li>`).join('')}</ul></div>`;
}

function switchStage(key) {
  state.selectedStage = key;
  renderTopbar();
  renderWorkflowSwitch();
  renderStages();
  renderWorkspace();
  renderChips();
  renderLog();
  renderChat();
}

/* ── Clear this lane's conversation ───────────────────────────────────────
   Server-side history only ever grows (each turn appends to the session), so a
   long case can outgrow the model's context window and reject every further
   message. This clears BOTH sides — the server transcript and this lane's local
   chat + activity log — so the next turn starts from nothing.

   Confirmed first: the chat is visible work an RM may still be reading, and the
   wipe cannot be undone. Refused mid-stream rather than racing the in-flight
   turn, whose SSE events would otherwise land in the lane we just emptied.
   Case data (KPIs, overrides, letters) lives in other stores and is untouched. */
async function clearConversation() {
  if (state.streaming) return;
  const convo = getConvo();
  const turns = getLogTurns();
  if (!convo.length && !turns.length) return;   // already empty — nothing to do
  if (!confirm('Clear this conversation? The case data and any letters are kept.')) return;

  const k = convoKey();
  try {
    await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ applicant_id: state.selectedId, stage: state.selectedStage }),
    });
  } catch (e) {
    // Local-only clear would desync the panes: the RM would see an empty chat
    // while the server still replays the old transcript into every next turn.
    console.error('reset failed', e);
    alert('Could not clear the conversation — the server did not respond.');
    return;
  }
  state.convos[k] = [];
  if (state.logTurnsByKey) state.logTurnsByKey[k] = [];
  // A pending gate belonged to the discarded turn; leaving it up would offer an
  // Approve that resumes a thread the server has just dropped.
  showHitl(false);
  state.reviseMode = false;
  syncComposer();
  renderChat();
  renderLog();
}

// ── chat rendering ────────────────────────────────────────────────────────────
function renderChat(keepScroll) {
  const chat = $('chat');
  const prevTop = chat.scrollTop;
  chat.innerHTML = '';
  const convo = getConvo();
  if (!convo.length) {
    chat.innerHTML =
      `<div class="chat-empty">Pick an action below, or ask your own question. The assistant's reasoning and answers appear here.</div>`;
    $('hitlRow').hidden = true;
    return;
  }
  convo.forEach((msg) => {
    try {
      chat.appendChild(renderMsg(msg));
    } catch (e) {
      // A single bad message must never blank the whole chat area.
      console.error('renderMsg failed for', msg, e);
      const d = document.createElement('div');
      d.className = 'msg note';
      d.innerHTML = `<div class="bubble">render error: ${escapeHtml(e.message)}</div>`;
      chat.appendChild(d);
    }
  });
  // Toggling a thinking block re-renders the whole list; without this the chat
  // would jump to the bottom every time the RM expands one mid-history.
  if (keepScroll) chat.scrollTop = prevTop;
  else scrollChat();
}

function renderMsg(msg) {
  if (msg.kind === 'think') {
    const d = document.createElement('div');
    d.className = 'think' + (msg.done ? ' done' : '') + (msg.open ? ' open' : '');
    const hasSummary = !!(msg.summary && msg.summary.trim());
    d.innerHTML =
      `<div class="think-head">
         <span class="spinner"></span><span class="check">✓</span>
         <span class="ttl">${escapeHtml(msg.title || '')}</span>
         ${msg.duration != null ? `<span class="dur">${msg.duration}s</span>` : ''}
         ${hasSummary ? '<span class="think-caret">▾</span>' : ''}
       </div>
       <div class="think-body md">${hasSummary ? renderMarkdown(msg.summary) : ''}</div>`;
    // Persist open/closed in the data model so it survives the next re-render.
    d.querySelector('.think-head').onclick = () => { msg.open = !msg.open; renderChat(true); };
    return d;
  }
  if (msg.kind === 'misroute') {
    const d = document.createElement('div');
    d.className = 'misroute';
    d.innerHTML =
      `<span>This looks like a ${stageLabel(msg.suggested_stage)} question — switch stage?</span>
       <span class="mr-actions">
         <button class="mr-btn primary">Switch</button>
         <button class="mr-btn">Stay</button></span>`;
    const [sw, stay] = d.querySelectorAll('.mr-btn');
    sw.onclick = () => { d.remove(); switchStage(msg.suggested_stage); };
    stay.onclick = () => d.remove();
    return d;
  }
  const d = document.createElement('div');
  d.className = 'msg ' + (msg.kind || 'agent');
  // Agent replies render markdown (bold / lists / tables); the RM's own messages
  // and inline notes stay plain text.
  if (msg.kind === 'agent') {
    d.innerHTML = `<div class="bubble md">${renderMarkdown(msg.text || '')}${renderLetterCard(msg.letter)}</div>${renderSources(msg.sources)}`;
  } else {
    d.innerHTML = `<div class="bubble">${escapeHtml(msg.text || '')}</div>`;
  }
  return d;
}

// "Sources" strip under an agent reply — the policy clauses search_policy
// retrieved this turn (from the SSE answer event's `sources`). Each is a source
// document + clause number, so the RM can see the answer's cited basis at a
// glance without opening the activity log. Empty sources → nothing rendered.
function renderSources(sources) {
  if (!sources || !sources.length) return '';
  const tags = sources.map((s) => {
    const doc = prettySource(s.source);
    const label = s.clause ? `${doc} · Clause ${s.clause}` : doc;
    return `<span class="src-tag" title="${escapeHtml(s.source)}">${escapeHtml(label)}</span>`;
  }).join('');
  return `<div class="src-strip"><span class="src-label">Sources</span>${tags}</div>`;
}

// A Claude-style PDF attachment card under a draft/released agent bubble. `letter`
// is {url, name, draft}: a red PDF glyph tile on the left, the file name + a small
// subtitle on the right. The whole card is the link and opens the PDF in a new tab
// (the endpoint serves it inline for preview). A draft card carries an amber DRAFT
// tag; the released one is plain.
function renderLetterCard(letter) {
  if (!letter || !letter.url) return '';
  const name = letter.name || (letter.draft ? 'Draft letter.pdf' : 'Letter.pdf');
  const sub = letter.draft ? 'PDF · Draft — pending review' : 'PDF · Official';
  const tag = letter.draft ? '<span class="letter-tag">DRAFT</span>' : '';
  return `<a class="letter-card${letter.draft ? ' draft' : ''}" href="${escapeHtml(letter.url)}" `
    + `target="_blank" rel="noopener" title="Open ${escapeHtml(name)}">`
    + `<span class="letter-glyph">PDF</span>`
    + `<span class="letter-meta"><span class="letter-name">${escapeHtml(name)}</span>`
    + `<span class="letter-sub">${sub}</span></span>${tag}</a>`;
}

// Turn a PDF file name into a readable document name for the Sources strip:
// "credit-facilities-tncs.pdf" → "Credit Facilities Tncs".
function prettySource(name) {
  return (name || '')
    .replace(/\.pdf$/i, '')
    .replace(/[-_]+/g, ' ')
    .replace(/\btncs?\b/i, 'T&C')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

function stageLabel(key) {
  const s = (state.caseData.stages || []).find((x) => x.key === key);
  return s ? s.label : key;
}
function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
// Small, self-contained Markdown → HTML renderer. We do NOT use marked.js: it
// relies on `eval`/`Function`, which a strict Content-Security-Policy (some
// browser extensions / hosting injects one) blocks — that throws inside
// rendering and collapses the agent bubbles / thinking summaries to blank lines.
// This covers what the agents actually emit: headings, bold, inline code,
// bullet/numbered lists, GFM tables, and paragraphs. Everything is escaped first
// so it is XSS-safe.
function renderMarkdown(src) {
  const esc = (t) => escapeHtml(t);
  const inline = (t) => esc(t)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  const lines = (src || '').replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let i = 0;
  let para = [];
  const flushPara = () => {
    if (para.length) { out.push('<p>' + para.map(inline).join('<br>') + '</p>'); para = []; }
  };

  while (i < lines.length) {
    const line = lines[i];

    // GFM table: header row, a separator row of ---/:--:, then body rows.
    if (/\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && /-/.test(lines[i + 1])) {
      flushPara();
      const splitRow = (r) => r.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());
      const head = splitRow(line);
      i += 2;
      const body = [];
      while (i < lines.length && /\|/.test(lines[i])) { body.push(splitRow(lines[i])); i++; }
      out.push('<table><thead><tr>' + head.map((c) => '<th>' + inline(c) + '</th>').join('') +
        '</tr></thead><tbody>' +
        body.map((r) => '<tr>' + r.map((c) => '<td>' + inline(c) + '</td>').join('') + '</tr>').join('') +
        '</tbody></table>');
      continue;
    }

    // Headings (#, ##, ###).
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) { flushPara(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }

    // Unordered / ordered list blocks.
    if (/^\s*[-*]\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {
      flushPara();
      const ordered = /^\s*\d+\.\s+/.test(line);
      const items = [];
      while (i < lines.length && (/^\s*[-*]\s+/.test(lines[i]) || /^\s*\d+\.\s+/.test(lines[i]))) {
        items.push(lines[i].replace(/^\s*(?:[-*]|\d+\.)\s+/, ''));
        i++;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>` + items.map((it) => '<li>' + inline(it) + '</li>').join('') + `</${tag}>`);
      continue;
    }

    // Blank line ends a paragraph; otherwise accumulate.
    if (line.trim() === '') { flushPara(); i++; continue; }
    para.push(line);
    i++;
  }
  flushPara();
  return out.join('');
}
function scrollChat() { const c = $('chat'); c.scrollTop = c.scrollHeight; }

// ── streaming chat ────────────────────────────────────────────────────────────
async function sendMessage(text) {
  text = (text || $('input').value).trim();
  if (!text) return;
  if (state.streaming) {
    // A turn is still in flight. Don't silently swallow the message (the old
    // behaviour that made the composer look dead when `streaming` got stuck) —
    // tell the RM what's happening so the input never feels unresponsive.
    const convo = getConvo();
    convo.push({ kind: 'note', text: '⏳ Still working on the previous request — please wait a moment and resend.' });
    renderChat();
    return;
  }
  $('input').value = '';
  autosize();

  const convo = getConvo();
  convo.push({ kind: 'rm', text });
  renderChat();
  await streamPost('/api/chat', { applicant_id: state.selectedId, stage: state.selectedStage, message: text });
  // A chat turn may have advanced this case server-side (an assessment flow ran,
  // a stage milestone was recorded). Re-pull so the deal-progress dots and the
  // next-best-action chip highlight move to the next step without a manual reload.
  await refreshCaseData();
}

async function streamPost(url, body) {
  // Re-entrancy guard: never run two streams at once. Without this, an
  // Approve/Reject fired while a turn was still streaming would race the first
  // stream's finally block and could leave `state.streaming` stuck true — which
  // silently kills the composer (sendMessage bails on state.streaming). One
  // stream at a time keeps the flag honest.
  if (state.streaming) {
    console.warn('[streamPost] ignored — a stream is already in flight');
    return;
  }
  state.streaming = true;
  setSendEnabled(false);
  $('hitlRow').hidden = true;
  const convo = getConvo();
  // Pin the lane this stream belongs to. The RM may switch stage/case while it
  // is still running; events keep landing in `convo` (the right lane's array),
  // but we must only repaint the chat/log when that lane is the one on screen —
  // otherwise the in-flight thinking blocks get rendered into the wrong lane and
  // collapse to empty lines. When the RM switches back, switchStage/selectCase
  // repaints from `convo` and everything is intact.
  const streamKey = convoKey();
  // Repaint chat + log, and the workspace (so the KPI card fills and the structured
  // Assessment Result card appears the moment the agent's answer lands, before the
  // post-turn refreshCaseData re-pull).
  const repaint = () => {
    if (convoKey() === streamKey) { renderChat(); renderLog(); renderWorkspace(); }
  };
  let openThink = null;
  console.log('[streamPost] start streamKey=', streamKey, 'convoLen=', convo.length);

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Process every COMPLETE SSE frame (terminated by a blank line). A frame
      // split across network chunks stays in `buf` until its terminator arrives.
      // Starlette's StreamingResponse emits CRLF, so frames end in '\r\n\r\n'
      // (not the '\n\n' we naively assumed) — match either, or the buffer grows
      // forever and nothing past the local RM bubble ever renders.
      let m;
      const sepRe = /\r?\n\r?\n/;
      while ((m = sepRe.exec(buf)) !== null) {
        const frame = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        const line = frame.split(/\r?\n/).find((l) => l.startsWith('data:'));
        if (!line) continue;
        let ev;
        try {
          ev = JSON.parse(line.slice(line.indexOf(':') + 1).trim());
        } catch (err) {
          console.error('SSE parse failed for frame:', frame, err);
          continue;  // skip a bad frame; never abort the whole stream
        }
        openThink = handleEvent(ev, convo, openThink, streamKey);
        if (convoKey() !== streamKey) console.warn('[repaint SKIP] live=', convoKey(), 'stream=', streamKey, 'ev=', ev.event);
        repaint();
      }
    }
  } catch (e) {
    console.error('[streamPost] threw:', e);
    convo.push({ kind: 'note', text: 'Connection error: ' + e.message });
    repaint();
  } finally {
    console.log('[streamPost] end convoLen=', convo.length, 'liveKey=', convoKey(), 'streamKey=', streamKey);
    state.streaming = false;
    setSendEnabled(true);
  }
}

function handleEvent(ev, convo, openThink, logKey) {
  switch (ev.event) {
    case 'user':
      // RM message already rendered locally; ignore the echo.
      return openThink;
    case 'turn':
      logStartTurn(ev.num, ev.user_msg, logKey);
      return openThink;
    case 'routed':
      logChain(ev, logKey);             // colored routing-chain badge
      return openThink;
    case 'a2a':
      logCard(ev, logKey);             // system / user / assistant message card
      return openThink;
    case 'thinking_open': {
      // Every block starts collapsed; the RM clicks the head to expand any one.
      const block = { kind: 'think', agent: ev.agent, title: ev.title, done: false, open: false };
      convo.push(block);
      return block;
    }
    case 'thinking_close':
      if (openThink) {
        openThink.done = true;
        openThink.title = ev.title || openThink.title;
        if (ev.duration != null) openThink.duration = ev.duration;
        if (ev.summary) openThink.summary = ev.summary;
        // Stay collapsed by default — the RM opens a block only if they want the
        // agent's detailed conclusion.
      }
      return null;
    case 'tool_call':
      logTool(ev.call_id, ev.name, ev.args, null, logKey);
      return openThink;
    case 'tool_result':
      logTool(ev.call_id, ev.name, undefined, ev.result, logKey);
      return openThink;
    case 'misroute':
      convo.push({ kind: 'misroute', suggested_stage: ev.suggested_stage });
      return openThink;
    case 'answer':
      convo.push({ kind: 'agent', text: ev.text, sources: ev.sources || [],
                   assessment: !!ev.assessment });
      return openThink;
    case 'draft':
      if (ev.replan_msg) convo.push({ kind: 'note', text: 'ℹ️ ' + ev.replan_msg });
      // `computed`, not `assessment`: a drafting flow does run the calculator (so the
      // scenario card may fill), but this message's body is the LETTER — quoting it in
      // the card's reasoning footer would show the letter, not the assessment.
      convo.push({ kind: 'agent', text: '📝 Draft ready — review before sending:\n\n' + ev.draft,
                   computed: !!ev.assessment,
                   letter: ev.pdf_url ? { url: ev.pdf_url, name: ev.pdf_name, draft: true } : null });
      // Only reveal the Approve/Revise gate if this stream's lane is on screen;
      // otherwise it would dangle on whatever lane the RM switched to.
      if (!logKey || logKey === convoKey()) showHitl(true);
      return openThink;
    case 'letter_ready':
      // The RM approved the draft; the released (non-DRAFT) letter PDF is ready.
      convo.push({ kind: 'agent', text: '✅ Letter released.',
                   letter: ev.pdf_url ? { url: ev.pdf_url, name: ev.pdf_name, draft: false } : null });
      return openThink;
    case 'error':
      convo.push({ kind: 'note', text: '⚠️ ' + ev.text });
      return openThink;
    case 'done':
      logFinishTurn(logKey);
      // Safety net: if a revise round-trip ended without producing a new draft
      // (e.g. an error), the HITL gate won't be showing — don't leave the RM with
      // the composer hidden and no controls. Restore it.
      if (state.reviseMode && $('hitlRow').hidden) { state.reviseMode = false; syncComposer(); }
      return openThink;
    default:
      return openThink;
  }
}

// ── HITL Approve / Revise ─────────────────────────────────────────────────────
// The normal composer is hidden whenever reviseMode is on, so a pending revision
// can't be sent through the free-form box (which wouldn't re-trigger the draft/PDF
// flow). Cancel and Approve clear reviseMode and bring the composer back.
function syncComposer() { $('composer').hidden = state.reviseMode; }
function showHitl(show) { $('hitlRow').hidden = !show; resetReviseInput(); }
// Collapse the revise input back to the Approve / Revise choice. Clicking Revise
// SWAPS the two-button choice for the input + Send/Cancel, so there is never a
// duplicate "revise" control on screen at once. Note: this does NOT clear
// reviseMode — only Cancel/Approve do — so the composer stays hidden across a
// reject→redraft round-trip until the RM approves.
function resetReviseInput() {
  $('reviseInput').hidden = true; $('reviseInput').value = '';
  $('reviseSend').hidden = true; $('reviseCancel').hidden = true;
  $('approveBtn').hidden = false; $('reviseBtn').hidden = false;
  syncComposer();
}

async function approveDraft() {
  state.reviseMode = false;   // confirmed — bring the normal composer back
  showHitl(false);
  syncComposer();
  await streamPost('/api/approve', { applicant_id: state.selectedId, stage: state.selectedStage, feedback: '' });
  // The approval may have advanced this case's milestones server-side; re-pull so
  // the deal-progress dots move without a manual page reload.
  await refreshCaseData();
}
async function rejectDraft(feedback) {
  showHitl(false);
  await streamPost('/api/reject', { applicant_id: state.selectedId, stage: state.selectedStage, feedback: feedback || '' });
  // A revise may have RECOMPUTED the case (e.g. "reprice to 1.2%") — the override
  // is now live server-side, so re-pull to reflect it in the KPI card, exactly as
  // approveDraft does. A wording-only redraft simply re-pulls the same numbers.
  await refreshCaseData();
}

// ── activity log (full Gradio-style: per-turn groups, colored routing chain,
//    role/tool cards, collapsible) ────────────────────────────────────────────
const ROLE_COLORS = {
  system:    { border: '#6366f1', bg: '#eef2ff', tag: '#c7d2fe' },
  user:      { border: '#0ea5e9', bg: '#f0f9ff', tag: '#bae6fd' },
  assistant: { border: '#10b981', bg: '#f0fdf4', tag: '#a7f3d0' },
};

// Per-(applicant, stage) log turns, so switching lanes preserves each lane's log.
// `key` is normally the live lane; an in-flight stream passes its pinned lane so
// events keep landing in the lane that started the turn even if the RM switches.
function getLogTurns(key) {
  const k = key || convoKey();
  if (!state.logTurnsByKey) state.logTurnsByKey = {};
  if (!state.logTurnsByKey[k]) state.logTurnsByKey[k] = [];
  return state.logTurnsByKey[k];
}
function curLogTurn(key) {
  const turns = getLogTurns(key);
  return turns.length ? turns[turns.length - 1] : null;
}

function logStartTurn(num, userMsg, key) {
  getLogTurns(key).push({ num, userMsg, chain: [], cards: [], tools: {}, active: true });
}
function logFinishTurn(key) {
  const t = curLogTurn(key);
  if (t) { t.active = false; t.chain.push({ done: true }); }
}
function logChain(ev, key) {
  const t = curLogTurn(key);
  if (!t) return;
  t.chain.push({ label: ev.label || ev.agent, color: ev.color || '#6b7280' });
}
function logCard(ev, key) {
  const t = curLogTurn(key);
  if (!t) return;
  t.cards.push({ type: 'a2a', role: ev.role, from: ev.from, to: ev.to,
                 content: ev.content, is_final: ev.is_final });
}
function logTool(callId, name, args, result, key) {
  const t = curLogTurn(key);
  if (!t) return;
  const tkey = callId || ('t' + t.cards.length);
  if (!t.tools[tkey]) { t.tools[tkey] = { type: 'tool', name, args: {}, result: null };
    t.cards.push(t.tools[tkey]); }
  if (args !== undefined) { t.tools[tkey].args = args; t.tools[tkey].name = name; }
  if (result !== undefined) { t.tools[tkey].result = result; }
}

function renderLog() {
  const body = $('logBody');
  const turns = getLogTurns();
  if (!turns.length) { body.innerHTML = '<div class="log-empty">No activity yet.</div>'; return; }
  body.innerHTML = turns.map(renderLogTurn).join('');
  // wire up collapsibles
  body.querySelectorAll('.log-card-head').forEach((h) => {
    h.onclick = () => h.parentElement.classList.toggle('open');
  });
  body.scrollTop = body.scrollHeight;
}

function renderLogTurn(t) {
  const header = `<div class="log-turn-head">Turn ${t.num} — ${escapeHtml((t.userMsg || '').slice(0, 60))}</div>`;
  const chain = t.chain.length
    ? '<div class="log-chain">' + t.chain.map((c) =>
        c.done ? badge('✓ Done', '#6b7280') : badge(c.label, c.color)
      ).join('<span class="arrow">→</span>') + '</div>'
    : '';
  const cards = t.cards.map((c) => c.type === 'tool' ? toolCard(c) : a2aCard(c)).join('');
  return `<div class="log-turn">${header}${chain}${cards}</div>`;
}

function badge(label, color) {
  return `<span class="log-badge" style="color:${color};border-color:${color};background:${color}22">${escapeHtml(label)}</span>`;
}

function a2aCard(c) {
  const col = ROLE_COLORS[c.role] || { border: '#6b7280', bg: '#f9fafb', tag: '#e5e7eb' };
  const label = c.is_final ? 'final answer' : (c.role || 'msg');
  const preview = (c.content || '').slice(0, 80).replace(/\n/g, ' ') + ((c.content || '').length > 80 ? '…' : '');
  return `<div class="log-card" style="background:${col.bg};border-left:3px solid ${col.border}">
    <div class="log-card-head">
      <span class="log-tag" style="background:${col.tag};color:${col.border}">${escapeHtml(label.toUpperCase())}</span>
      <span class="log-route">${escapeHtml(c.from || '')} → ${escapeHtml(c.to || '')}</span>
      <span class="log-caret">▾</span>
    </div>
    <pre class="log-pre">${escapeHtml(c.content || '')}</pre>
    <div class="log-preview">${escapeHtml(preview)}</div>
  </div>`;
}

function toolCard(c) {
  const argsStr = JSON.stringify(c.args || {}, null, 2);
  let resultStr = c.result;
  if (resultStr != null) { try { resultStr = JSON.stringify(JSON.parse(resultStr), null, 2); } catch (e) {} }
  const resultBlock = c.result != null
    ? `<div class="log-sub">result (${(c.result || '').length} chars)</div><pre class="log-pre result">${escapeHtml(resultStr)}</pre>`
    : `<div class="log-exec">⏳ executing…</div>`;
  return `<div class="log-card tool" style="background:#fefce8;border-left:3px solid #f59e0b">
    <div class="log-card-head">
      <span class="log-tool-name">🔧 ${escapeHtml(c.name || '')}</span>
      <span class="log-caret">▾</span>
    </div>
    <div class="log-sub">args</div><pre class="log-pre">${escapeHtml(argsStr)}</pre>
    ${resultBlock}
  </div>`;
}

// ── input autosize ─────────────────────────────────────────────────────────────
function autosize() {
  const t = $('input');
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 140) + 'px';
}
function setSendEnabled(on) { $('sendBtn').disabled = !on; }

// ── wire up ─────────────────────────────────────────────────────────────────────
function init() {
  $('switchBtn').onclick = () => { $('casePicker').hidden = !$('casePicker').hidden; };
  $('caseCurrent').onclick = () => { $('casePicker').hidden = !$('casePicker').hidden; };
  $('pickerClose').onclick = () => { $('casePicker').hidden = true; };
  $('sendBtn').onclick = () => sendMessage();
  $('input').addEventListener('input', autosize);
  $('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  $('approveBtn').onclick = approveDraft;
  $('reviseBtn').onclick = () => {
    // Swap the Approve / Revise choice for the revise input + Send / Cancel, and
    // hide the normal composer so the revision can only go through this input.
    state.reviseMode = true;
    $('approveBtn').hidden = true; $('reviseBtn').hidden = true;
    $('reviseInput').hidden = false; $('reviseSend').hidden = false; $('reviseCancel').hidden = false;
    syncComposer();
    $('reviseInput').focus();
  };
  $('reviseCancel').onclick = () => {
    // Back out of revising: restore the Approve / Revise choice and the composer.
    state.reviseMode = false;
    resetReviseInput();
  };
  $('reviseSend').onclick = () => rejectDraft($('reviseInput').value);
  $('reviseInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); rejectDraft($('reviseInput').value); }
  });

  $('logLink').onclick = (e) => {
    e.preventDefault();
    $('logPane').hidden = !$('logPane').hidden;
    if (!$('logPane').hidden) renderLog();
  };
  $('logClose').onclick = () => { $('logPane').hidden = true; };
  $('clearBtn').onclick = () => clearConversation();

  initRatesPanel();
  loadCases();
}

/* ── Market rates panel ───────────────────────────────────────────────────
   Two-speed by design: opening reads the stored CSV (instant), while ⟳ runs a
   live collect (~4s, MAS API dominates). Never blocks on open, never re-hits
   the sources unless asked. */
const rates = { data: null, cat: 'fixed', loading: false };

function initRatesPanel() {
  $('ratesBtn').onclick = () => {
    const p = $('ratesPanel');
    p.hidden = !p.hidden;
    if (!p.hidden && !rates.data) loadRates();
  };
  $('ratesClose').onclick = () => { $('ratesPanel').hidden = true; };
  $('ratesRefresh').onclick = () => loadRates(true);

  document.querySelectorAll('.rp-tab').forEach((t) => {
    t.onclick = () => {
      rates.cat = t.dataset.cat;
      document.querySelectorAll('.rp-tab').forEach((x) => x.classList.toggle('is-on', x === t));
      renderRatesTable();
      renderBankSpark();   // the trend follows the same Fixed/Floating tab
    };
  });

  makeDraggable($('ratesPanel'), $('ratesDrag'));
}

/* Drag by the header. Switches the panel from right/top anchoring to explicit
   left/top so it can be parked anywhere; clamped to stay on-screen. */
function makeDraggable(panel, handle) {
  let dx = 0, dy = 0, dragging = false;

  handle.addEventListener('mousedown', (e) => {
    if (e.target.closest('button')) return;   // let ⟳ / ✕ work
    const r = panel.getBoundingClientRect();
    dx = e.clientX - r.left; dy = e.clientY - r.top;
    dragging = true;
    panel.style.left = r.left + 'px';
    panel.style.top = r.top + 'px';
    panel.style.right = 'auto';
    handle.classList.add('dragging');
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const w = panel.offsetWidth, h = panel.offsetHeight;
    const x = Math.min(Math.max(0, e.clientX - dx), window.innerWidth - w);
    const y = Math.min(Math.max(0, e.clientY - dy), window.innerHeight - Math.min(h, 120));
    panel.style.left = x + 'px';
    panel.style.top = y + 'px';
  });

  window.addEventListener('mouseup', () => {
    dragging = false;
    handle.classList.remove('dragging');
  });
}

async function loadRates(refresh = false) {
  if (rates.loading) return;
  rates.loading = true;
  const btn = $('ratesRefresh');
  btn.classList.add('spin');
  $('ratesStamp').textContent = refresh ? 'fetching…' : 'loading…';
  try {
    const r = await fetch('/api/rates' + (refresh ? '/refresh' : ''), {
      method: refresh ? 'POST' : 'GET',
    });
    rates.data = await r.json();
    renderRates();
  } catch (e) {
    $('ratesErr').hidden = false;
    $('ratesErr').textContent = 'Could not load rates: ' + e;
    $('ratesStamp').textContent = '';
  } finally {
    rates.loading = false;
    btn.classList.remove('spin');
  }
}

function renderRates() {
  const d = rates.data || {};

  const err = $('ratesErr');
  if (d.errors && d.errors.length) {
    err.hidden = false;
    err.textContent = d.errors.join(' · ');
  } else { err.hidden = true; }

  $('ratesStamp').textContent = d.scraped_at ? 'scraped ' + agoText(d.scraped_at) : '';

  // Benchmark strip
  const b = $('ratesBench');
  if (d.sora_latest != null) {
    b.innerHTML =
      '<span class="rp-bench-val">' + d.sora_latest.toFixed(4) + '%</span>' +
      '<span class="rp-bench-lbl">MAS 3M Compounded SORA</span>' +
      '<span class="rp-bench-meta">as of ' + escapeHtml(d.sora_as_of) + '</span>';
  } else {
    // No fallback number — say it's unavailable rather than show a stale one.
    b.innerHTML = '<span class="rp-bench-lbl">MAS SORA unavailable</span>' +
      '<span class="rp-bench-meta">try ⟳</span>';
  }

  renderRatesTable();
  renderSpark();
  renderRatesSources();
}

/* Source attribution. Competitor rates are a broker's republication, not a
   bank's own quote, so the panel says who published what rather than letting
   both figures read as equally official. */
function renderRatesSources() {
  const s = (rates.data && rates.data.sources) || {};
  // Each source is one unbreakable chunk, so a wrap never orphans a detail
  // label onto its own line away from the link it describes.
  const link = (k) => {
    const v = s[k];
    if (!v) return '';
    return '<span class="rp-src-item"><a href="' + escapeHtml(v.url) +
      '" target="_blank" rel="noopener noreferrer">' + escapeHtml(v.label) +
      '</a> <span class="rp-src-det">' + escapeHtml(v.detail) + '</span></span>';
  };
  const parts = [link('mas'), link('dollarback')].filter(Boolean);
  $('ratesSrc').innerHTML = parts.length
    ? '<span class="rp-src-lbl">Sources</span>' + parts.join('')
    : '';
}

function renderRatesTable() {
  const rows = ((rates.data && rates.data.competitors) || [])
    .filter((c) => c.rate_category === rates.cat);
  const el = $('ratesTable');

  if (!rows.length) {
    el.innerHTML = '<div class="rp-empty">No ' + escapeHtml(rates.cat) + ' packages stored yet.</div>';
    return;
  }
  const floating = rates.cat === 'floating';
  // Y1–Y3 only: Y4–Y6 are almost always the same reversion rate, so they cost
  // width without adding information.
  el.innerHTML =
    '<table><thead><tr><th>Bank</th><th>Yr 1</th><th>Yr 2</th><th>Yr 3</th></tr></thead><tbody>' +
    rows.map((c) =>
      '<tr><td><span class="rp-bank">' + escapeHtml(c.bank) + '</span>' +
      '<span class="rp-sub">' + escapeHtml(c.loan_type) +
      (c.lock_in_years ? ' · ' + escapeHtml(c.lock_in_years) + 'y lock-in' : '') + '</span></td>' +
      [0, 1, 2].map((i) => {
        const v = c.years[i];
        return '<td' + (i === 0 ? ' class="rp-y1"' : '') + '>' + (v ? escapeHtml(v) + (floating ? '' : '%') : '—') + '</td>';
      }).join('') + '</tr>'
    ).join('') + '</tbody></table>' +
    (floating ? '<div class="rp-sub" style="padding:6px 6px 0">Spreads over the benchmark — add SORA above for the all-in rate.</div>' : '');
}

/* Inline SVG sparklines — no chart library (CSP blocks eval-based bundles, and
   a dependency is not worth two collapsed strips).

   Both charts render whatever exists, down to a single observation drawn as a
   dot: bank rates have no historical feed (the source page only shows "now"),
   so that series grows one point per day the collector runs and must not sit
   behind a minimum-points gate. */
const SERIES_COLORS = ['#00237B', '#2E77C2', '#0F766E', '#B45309', '#9333EA', '#BE123C', '#4D7C0F'];

function renderSpark() {
  renderSoraSpark();
  renderBankSpark();
}

/* Pick ~n round gridline values covering [lo, hi]. Steps are 1/2/5×10^k so
   labels land on readable numbers (1.20, 1.25…) rather than raw data minima. */
function niceTicks(lo, hi, n) {
  if (hi === lo) {
    // Flat series: invent a small window so the value still gets a labelled line.
    const pad = Math.abs(lo) > 1 ? 0.05 : 0.01;
    lo -= pad; hi += pad;
  }
  const raw = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const first = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = first; v <= hi + step * 0.001; v += step) {
    ticks.push(Math.round(v / step) * step);   // kill float drift (1.2000000000002)
  }
  return { ticks, min: Math.min(lo, ticks[0]), max: Math.max(hi, ticks[ticks.length - 1]) };
}

/* Shared plotter. `series` = [{label, color, points:[{rate}], fill}].
   Draws a labelled y-axis with gridlines — without a scale the shape alone
   doesn't say what any level actually is.

   NOTE: no preserveAspectRatio="none" here. Stretching the viewBox would
   distort the axis text along with the plot, so the chart keeps its ratio and
   the CSS sizes it. */
function sparkSvg(series, aria, decimals) {
  const W = 440, H = 96;
  // Left gutter holds tick labels; the right one holds end-of-line names when
  // any series asks for one (otherwise the text would be clipped).
  const PL = 42, PT = 8, PB = 16;
  const PR = series.some((s) => s.endLabel) ? 46 : 6;
  const all = series.flatMap((s) => s.points.map((p) => p.rate));
  if (!all.length) return '';
  const dp = decimals == null ? 2 : decimals;

  const { ticks, min, max } = niceTicks(Math.min(...all), Math.max(...all), 3);
  const span = max - min || 1;
  const y = (v) => PT + (H - PT - PB) * (1 - (v - min) / span);
  const maxLen = Math.max(...series.map((s) => s.points.length));
  const x = (i) => (maxLen < 2 ? (PL + W - PR) / 2 : PL + (i * (W - PL - PR)) / (maxLen - 1));

  const grid = ticks.map((t) =>
    '<line x1="' + PL + '" y1="' + y(t).toFixed(1) + '" x2="' + (W - PR) +
    '" y2="' + y(t).toFixed(1) + '" stroke="#E2E8F0" stroke-width="1"/>' +
    '<text x="' + (PL - 5) + '" y="' + (y(t) + 3).toFixed(1) +
    '" text-anchor="end" font-size="9" fill="#94A3B8">' + t.toFixed(dp) + '</text>'
  ).join('');

  const axis = '<line x1="' + PL + '" y1="' + PT + '" x2="' + PL + '" y2="' + (H - PB) +
    '" stroke="#CBD5E1" stroke-width="1"/>';

  const body = series.map((s) => {
    const pts = s.points;
    const d = pts
      .map((p, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(p.rate).toFixed(1))
      .join(' ');
    // Area fill suits a lone series; with several lines it would muddy them.
    const area = (s.fill && pts.length > 1)
      ? '<path d="' + d + ' L' + x(pts.length - 1).toFixed(1) + ' ' + (H - PB) +
        ' L' + x(0).toFixed(1) + ' ' + (H - PB) + ' Z" fill="rgba(0,35,123,.09)"/>'
      : '';
    const line = pts.length > 1
      ? '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="' +
        (s.dash ? '1.2" stroke-dasharray="3 2' : '1.5') + '" ' +
        'stroke-linejoin="round" stroke-linecap="round"/>'
      : '';

    // Hover targets. Native <title> gives a tooltip with no JS and no library;
    // the marker is what the pointer actually has to hit, so on dense series
    // only its generous transparent halo is enlarged, not the visible dot.
    const dots = pts.map((p, i) => {
      const cx = x(i).toFixed(1), cy = y(p.rate).toFixed(1);
      const tip = (p.day ? p.day + ' · ' : '') + s.label + ': ' + p.rate.toFixed(dp) + '%';
      return '<g class="rp-pt"><title>' + escapeHtml(tip) + '</title>' +
        '<circle cx="' + cx + '" cy="' + cy + '" r="7" fill="transparent"/>' +
        '<circle class="rp-dot" cx="' + cx + '" cy="' + cy + '" r="' +
        (pts.length > 40 ? 1.6 : 2.4) + '" fill="' + s.color + '"/></g>';
    }).join('');

    return area + line + dots;
  }).join('');

  // Name each line at its right end so a colour never has to be matched back to
  // the legend by eye. Done after the lines so labels sit on top, and nudged
  // apart vertically — packages often sit within a few basis points of each
  // other, which would otherwise stack the text into an unreadable clump.
  const LH = 9;
  const placed = series
    .filter((s) => s.endLabel)
    .map((s) => ({ s, y: y(s.points[s.points.length - 1].rate) }))
    .sort((a, b) => a.y - b.y);
  placed.forEach((p, i) => {
    if (i && p.y - placed[i - 1].y < LH) p.y = placed[i - 1].y + LH;
  });
  // If the nudging pushed the stack off the bottom, shift it all back up.
  const overflow = placed.length ? placed[placed.length - 1].y - (H - 2) : 0;
  if (overflow > 0) placed.forEach((p) => { p.y -= overflow; });

  const endLabels = placed.map((p) =>
    '<text x="' + (W - PR + 3) + '" y="' + (p.y + 3).toFixed(1) +
    '" font-size="8.5" fill="' + p.s.color + '" font-weight="600">' +
    escapeHtml(p.s.endLabel) + '</text>'
  ).join('');

  return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
    'aria-label="' + escapeHtml(aria) + '">' +
    grid + axis + body + endLabels + '</svg>';
}

function renderSoraSpark() {
  const s = (rates.data && rates.data.sora) || [];
  const el = $('ratesSpark');
  const sub = $('ratesTrendSub');
  if (!s.length) {
    sub.textContent = '';
    el.innerHTML = '<div class="rp-empty">No SORA observations stored yet.</div>';
    return;
  }
  sub.textContent = '(' + s.length + (s.length === 1 ? ' day)' : ' days)');

  const vals = s.map((p) => p.rate);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const svg = sparkSvg(
    [{
      label: 'SORA',
      color: '#00237B',
      points: s.map((p) => ({ rate: p.rate, day: p.as_of })),
      fill: true,
    }],
    'SORA trend, ' + s.length + ' days',
    4
  );
  // The axis now carries the levels, so the footer only spans the date range.
  el.innerHTML = svg +
    '<div class="rp-sub" style="display:flex;justify-content:space-between;padding-top:2px">' +
    '<span>' + escapeHtml(s[0].as_of) + '</span>' +
    '<span>' + lo.toFixed(4) + '–' + hi.toFixed(4) + '%</span>' +
    '<span>' + escapeHtml(s[s.length - 1].as_of) + '</span></div>';
}

function renderBankSpark() {
  const h = (rates.data && rates.data.bank_history) || {};
  const el = $('ratesBankSpark');
  const sub = $('ratesBankSub');
  const labels = Object.keys(h);
  if (!labels.length) {
    sub.textContent = '';
    el.innerHTML = '<div class="rp-empty">No competitor rates stored yet.</div>';
    return;
  }

  // Follow the Fixed/Floating tab. Both tabs plot all-in rates (the server
  // resolves floating spreads against the SORA of the day), so the two charts
  // read on the same scale — but they stay separate because a fixed rate and a
  // floating one answer different questions.
  const shown = labels.filter((k) => h[k].category === rates.cat);
  if (!shown.length) {
    sub.textContent = '';
    el.innerHTML = '<div class="rp-empty">No ' + escapeHtml(rates.cat) +
      ' packages stored yet.</div>';
    return;
  }

  const days = new Set();
  shown.forEach((k) => h[k].points.forEach((p) => days.add(p.day)));
  const nDays = days.size;
  const unit = rates.cat === 'floating' ? 'Yr 1 all-in' : 'Yr 1';
  sub.textContent = '(' + unit + ' · ' + nDays + (nDays === 1 ? ' day)' : ' days)');

  // Short end-of-line tag: bank name, plus the lock-in when one bank has two
  // packages on the chart (e.g. BOC 2y and 3y) and the name alone is ambiguous.
  const banks = shown.map((k) => k.split(' · ')[0]);
  // Long names don't fit the right gutter; the legend below carries the full one.
  const SHORT = { 'Bank of China': 'BOC', 'Standard Chartered': 'StanChart',
                  'State Bank of India': 'SBI', 'Sing Investments & Finance': 'SIF' };
  const series = shown.map((k, i) => {
    const [bank, pkg] = k.split(' · ');
    const dup = banks.filter((b) => b === bank).length > 1;
    const yrs = (pkg.match(/(\d+)\s*Year/i) || [])[1];
    const short = SHORT[bank] || bank;
    return {
      label: k,
      endLabel: dup && yrs ? short + ' ' + yrs + 'y' : short,
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      points: h[k].points.map((p) => ({ rate: p.rate, day: p.day })),
    };
  });

  // On the floating chart, draw the benchmark the packages are priced off.
  // Dashed and grey so it reads as a reference level, not another package.
  const sortedDays = [...days].sort();
  if (rates.cat === 'floating') {
    const sora = (rates.data && rates.data.sora) || [];
    const onDay = (d) => {
      let v = null;
      for (const p of sora) { if (p.as_of <= d) v = p.rate; else break; }
      return v;
    };
    const pts = sortedDays.map((d) => ({ rate: onDay(d), day: d })).filter((p) => p.rate != null);
    if (pts.length) {
      series.push({
        label: 'MAS 3M SORA (benchmark)',
        endLabel: 'SORA',
        color: '#94A3B8',
        dash: true,
        points: pts,
      });
    }
  }

  // Legend still carries the full package name (the in-chart tag is abbreviated
  // to fit), so keep it — but it is no longer the only way to read the chart.
  const legend = series.map((s) =>
    '<span class="rp-leg"><i class="' + (s.dash ? 'rp-leg-dash' : '') +
    '" style="background:' + s.color + '"></i>' +
    escapeHtml(s.label) + '</span>'
  ).join('');

  // One day of history is the normal starting state, not a failure — say so
  // plainly instead of hiding the chart.
  const hint = nDays < 2
    ? '<div class="rp-sub" style="padding-top:4px">Tracking started today — the line builds as the collector runs on later days.</div>'
    : '';

  const range = nDays > 1
    ? '<div class="rp-sub" style="display:flex;justify-content:space-between;padding-top:2px">' +
      '<span>' + escapeHtml(sortedDays[0]) + '</span>' +
      '<span>' + escapeHtml(sortedDays[sortedDays.length - 1]) + '</span></div>'
    : '';

  el.innerHTML = sparkSvg(series, 'Competitor Year-1 rates, ' + nDays + ' days', 2) +
    range + '<div class="rp-legend">' + legend + '</div>' + hint;
}

function agoText(iso) {
  const t = Date.parse(iso);
  if (isNaN(t)) return '';
  const mins = Math.round((Date.now() - t) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return mins + 'm ago';
  const h = Math.round(mins / 60);
  return h < 24 ? h + 'h ago' : Math.round(h / 24) + 'd ago';
}

// Build marker — if you DON'T see this line in the browser console after a refresh,
// the browser is serving a STALE cached app.js (hard-reload / clear site data).
console.log('%cAgentic Home Loan Workbench UI build: top-nav planner layout (no rail)', 'color:#1A3D6B;font-weight:bold');

document.addEventListener('DOMContentLoaded', init);
