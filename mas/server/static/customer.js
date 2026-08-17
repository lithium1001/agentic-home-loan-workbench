/* Home Loan Explorer — customer self-service portal controller.
   A guided three-step wizard (DBS-marketplace style): pick an intent → tell us
   about yourself → deterministic results. All four intents resolve to modes of
   POST /api/customer/calc, which drives utils/calculator directly (no LLM: it
   is instant, free, and cannot hallucinate a number). The right pane is the
   restricted customer assistant chat (role="customer" → the graph hard-routes
   to customer_assistant; its only tools are the calculator, policy search, and
   the public package catalog). */
'use strict';

const $ = (id) => document.getElementById(id);

const MODES = {
  price:       { card: 'Property Price',     lead: 'You have a price in mind — let\'s see what it takes.' },
  instalment:  { card: 'Monthly Instalment', lead: 'You know your monthly budget — let\'s find the price it supports.' },
  downpayment: { card: 'Downpayment',        lead: 'You have a downpayment set aside — let\'s see your budget.' },
  explore:     { card: 'I\'m still exploring', lead: 'Let\'s see the maximum loan your income supports.' },
};
// Progress-line fill per wizard step (the house marker rides the fill edge).
const STEP_FILL = { 1: '13%', 2: '47%', 3: '83%' };

const state = {
  step: 1,
  mode: 'price',
  packages: [],
  lastCalc: null,       // last successful calc result (for the chat handoff)
  convo: [],
  streaming: false,
};

function fmtMoney(n) {
  if (n == null) return '—';
  return 'S$' + Math.round(n).toLocaleString('en-SG');
}
function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
// Clipboard fallback for contexts where navigator.clipboard is unavailable
// (e.g. non-secure origin during local demo).
function fallbackCopy(text, done) {
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); done && done(); } catch (e) { /* noop */ }
  document.body.removeChild(ta);
}

// ── wizard navigation ────────────────────────────────────────────────────────
function goStep(n) {
  state.step = n;
  $('panelIntent').hidden = n !== 1;
  $('panelForm').hidden = n !== 2;
  $('panelResults').hidden = n !== 3;
  document.querySelectorAll('.cx-step').forEach((b) => {
    const s = parseInt(b.dataset.step, 10);
    b.classList.toggle('active', s === n);
    b.classList.toggle('done', s < n);
  });
  $('progressFill').style.width = STEP_FILL[n];
  $('progressIcon').style.left = STEP_FILL[n];   // house marker rides the fill edge
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('.cx-intent').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  // Show only this mode's extra fields (fields may belong to several modes).
  document.querySelectorAll('.mode-f').forEach((el) => {
    el.hidden = !el.classList.contains('mode-' + mode);
  });
  $('formLead').textContent = MODES[mode].lead;
  $('formTitle').textContent = MODES[mode].card + ' · tell us more about yourself';
}

// ── rate packages (public catalog) ──────────────────────────────────────────
async function loadPackages() {
  try {
    const data = await (await fetch('/api/customer/packages')).json();
    state.packages = data.packages || [];
  } catch (e) { state.packages = []; }
  const sel = $('fRate');
  sel.innerHTML = '';
  state.packages.forEach((p) => {
    if (p.indicative_rate_pct == null) return;
    const o = document.createElement('option');
    o.value = p.indicative_rate_pct;
    const nm = p.package_name || p.package_id || p.rate_type || 'Package';
    o.textContent = `${nm} · ${p.indicative_rate_pct}%`;
    sel.appendChild(o);
  });
  if (!sel.options.length) {
    const o = document.createElement('option');
    o.value = '3.85';
    o.textContent = 'Indicative · 3.85%';
    sel.appendChild(o);
  }
}

// ── MAS 645 eligible financial assets ───────────────────────────────────────
// Mirrors utils/calculator._MAS645_ASSET_CLASSES. The live preview here is only
// a courtesy so the customer sees the effect as they type; the figure that
// actually prices the case is always the server's, never this one.
const ASSET_CLASSES = [
  { key: 'fixed_deposit',    label: 'Fixed Deposit',                  pledgedHaircut: 0.00 },
  { key: 'ssb_sgs',          label: 'Singapore Savings Bonds / SGS',  pledgedHaircut: 0.00 },
  { key: 'foreign_currency', label: 'Foreign Currency Deposit',       pledgedHaircut: 0.30 },
  { key: 'gold',             label: 'Gold Certificate',               pledgedHaircut: 0.30 },
  { key: 'shares',           label: 'Shares',                         pledgedHaircut: 0.70 },
  { key: 'unit_trust',       label: 'Unit Trust',                     pledgedHaircut: 0.70 },
];
const UNPLEDGED_HAIRCUT = 0.70;
const AMORTISATION_MONTHS = 48;

function renderAssetRows() {
  const tb = $('assetsRows');
  if (!tb) return;
  tb.innerHTML = ASSET_CLASSES.map((a) =>
    `<tr>
       <td class="cxat-label">${a.label}</td>
       <td><input class="cxat-amt" id="asset_${a.key}" type="number" min="0" step="1000" value="0" /></td>
       <td class="cxat-pledge"><input id="pledge_${a.key}" type="checkbox" checked /></td>
       <td class="cxat-out" id="out_${a.key}">—</td>
     </tr>`).join('');
  ASSET_CLASSES.forEach((a) => {
    $('asset_' + a.key).addEventListener('input', updateAssetPreview);
    $('pledge_' + a.key).addEventListener('change', updateAssetPreview);
  });
  updateAssetPreview();
}

// Reads the asset inputs into the request shape the API expects.
function readAssets() {
  const out = {};
  ASSET_CLASSES.forEach((a) => {
    const el = $('asset_' + a.key);
    if (!el) return;
    const amount = parseFloat(el.value || '0') || 0;
    if (amount > 0) out[a.key] = { amount, pledged: $('pledge_' + a.key).checked };
  });
  return out;
}

function updateAssetPreview() {
  let total = 0;
  ASSET_CLASSES.forEach((a) => {
    const amount = parseFloat(($('asset_' + a.key) || {}).value || '0') || 0;
    const pledged = ($('pledge_' + a.key) || {}).checked;
    const haircut = pledged ? a.pledgedHaircut : UNPLEDGED_HAIRCUT;
    const monthly = amount > 0 ? (amount * (1 - haircut)) / AMORTISATION_MONTHS : 0;
    total += monthly;
    const cell = $('out_' + a.key);
    if (cell) {
      cell.textContent = amount > 0
        ? `${fmtMoney(monthly)} /mo  ·  ${(haircut * 100).toFixed(0)}% deduction`
        : '—';
    }
  });
  const t = $('assetsTotal');
  if (t) t.textContent = fmtMoney(total) + (total ? ' /mo' : '');
  const sum = $('assetsSummary');
  if (sum) sum.textContent = total ? '+' + fmtMoney(total) + ' /mo' : '';
}

// ── calculate (deterministic; no LLM) ───────────────────────────────────────
function readForm() {
  const num = (id) => parseFloat($(id).value || '0') || 0;
  const body = {
    mode: state.mode,
    age: Math.round(num('fAge')),
    monthly_fixed_income: num('fFixed'),
    monthly_variable_income: num('fVar'),
    nationality: $('fNat').value,
    property_type: $('fProp').value,
    properties_owned_now: Math.round(num('fOwned')),
    outstanding_home_loans: Math.round(num('fLoans')),
    monthly_car_loan: num('fCar'),
    monthly_other: num('fOther'),
    interest_rate_pct: parseFloat($('fRate').value || '3.85'),
    financial_assets: readAssets(),
  };
  if (state.mode === 'price') body.target_property_price = num('fPrice');
  if (state.mode === 'instalment') { body.monthly_budget = num('fBudget'); body.cash_cpf_available = num('fCash'); }
  if (state.mode === 'downpayment') body.cash_cpf_available = num('fCash');
  return body;
}

async function runCalc() {
  const btn = $('calcBtn');
  btn.disabled = true; btn.textContent = 'Calculating…';
  try {
    const res = await (await fetch('/api/customer/calc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readForm()),
    })).json();
    renderResults(res);
    state.lastCalc = res.error ? null : res;
    goStep(3);
  } catch (e) {
    renderResults({ error: 'Could not reach the calculator: ' + e.message });
    goStep(3);
  } finally {
    btn.disabled = false; btn.textContent = 'Show my results ›';
  }
}

// Per-mode KPI framing: the tile order leads with the number the customer asked
// for (price → cash needed; instalment/downpayment → property budget; explore →
// max loan). All values come straight from the calculator response.
function renderResults(r) {
  const el = $('cxResults');
  if (!r || r.error) {
    el.innerHTML = `<div class="planner-empty">⚠️ ${escapeHtml(r ? r.error : 'No result')}</div>`;
    return;
  }
  const kpi = (k, v, sub) =>
    `<div class="kpi"><div class="kpi-k">${k}</div><div class="kpi-v">${v}</div>` +
    (sub ? `<div class="kpi-sub">${sub}</div>` : '') + `</div>`;

  const mode = r.mode || state.mode;
  const loanTile = kpi('Max loan', fmtMoney(r.eligible_loan),
    r.binding_constraint ? 'capped by ' + escapeHtml(r.binding_constraint) : '');
  const cashTile = kpi('Cash + CPF needed', fmtMoney(r.required_cash_cpf), 'downpayment + stamp duty');
  const priceTile = kpi(mode === 'price' ? 'Property price' : 'Property budget', fmtMoney(r.property_price));
  const tenureTile = kpi('Tenure', (r.loan_tenure_years ?? '—') + ' yrs', 'age-capped');
  const ltvTile = kpi('LTV limit', (r.ltv_limit_pct ?? '—') + '%');
  const dutyTile = kpi('Stamp duty', fmtMoney(r.total_stamp_duty),
    'BSD ' + fmtMoney(r.buyer_stamp_duty) + (r.additional_bsd ? ' + ABSD ' + fmtMoney(r.additional_bsd) : ''));
  const tiles = {
    price:       [loanTile, priceTile, cashTile, tenureTile, ltvTile, dutyTile],
    instalment:  [priceTile, loanTile, cashTile, tenureTile, ltvTile, dutyTile],
    downpayment: [priceTile, loanTile, cashTile, tenureTile, ltvTile, dutyTile],
    explore:     [loanTile, priceTile, cashTile, tenureTile, ltvTile, dutyTile],
  }[mode] || [loanTile, priceTile, cashTile, tenureTile, ltvTile, dutyTile];

  const tdsr = r.tdsr_pct;
  const over = tdsr != null && tdsr > 55;
  const steps = (r.calculation_steps || []).map((s) =>
    `<tr><td>${escapeHtml(s.step || s.label || '')}</td><td class="cxs-f">${escapeHtml(s.formula || '')}</td>` +
    `<td class="cxs-v">${typeof s.value === 'number' ? s.value.toLocaleString('en-SG') : escapeHtml(String(s.value ?? ''))}</td></tr>`).join('');

  el.innerHTML =
    `<div class="calc-card">
       <div class="calc-head">
         <span class="calc-title">Your indicative results — ${escapeHtml(MODES[mode] ? MODES[mode].card : '')}</span>
         <span class="calc-rate">Indicative · ${r.interest_rate_pct}% p.a.</span>
       </div>
       ${r.note ? `<div class="cx-note">ℹ️ ${escapeHtml(r.note)}</div>` : ''}
       ${r.amortized_monthly_income ? `<div class="cx-note cx-note-645">
          ✓ Includes <strong>${fmtMoney(r.amortized_monthly_income)}/month</strong> recognised from your
          financial assets under MAS Notice 645, giving total qualifying income of
          ${fmtMoney(r.total_monthly_income)}/month.
        </div>` : ''}
       <div class="calc-kpis">${tiles.join('')}</div>
       <div class="calc-summary">
         <div class="cs-cell">
           <div class="cs-k">Monthly repayment</div>
           <div class="cs-v navy">${fmtMoney(r.monthly_repayment)}</div>
           <div class="cs-sub">${fmtMoney(r.monthly_repayment_stress)} at the 4% stress rate</div>
         </div>
         <div class="cs-cell">
           <div class="cs-k">TDSR</div>
           <div class="cs-v ${over ? 'amber' : 'green'}">${tdsr != null ? tdsr + '%' : '—'}</div>
           <div class="cs-sub ${over ? 'amber' : 'green'}">${tdsr != null ? (over ? 'over the 55% cap' : 'under the 55% cap ✓') : ''}</div>
         </div>
         <div class="cs-cell wide">
           <div class="cs-k">Next step</div>
           <div class="cx-nextbtns">
             <button class="chip cx-explain" id="explainBtn">Ask the assistant to explain these numbers ›</button>
             <button class="chip cx-handoff" id="handoffBtn">Send my summary to a specialist ›</button>
           </div>
         </div>
       </div>
       <div class="calc-foot">
         <div class="ar-reason" id="cxSteps">
           <div class="ar-reason-head"><span>How we calculated this (full trace)</span><span class="ar-caret">▾</span></div>
           <div class="ar-reason-body"><table class="cx-steps">${steps}</table></div>
         </div>
       </div>
     </div>
     ${renderSchedulePanel(r)}
     ${renderBucPanel(r)}`;

  const stepsEl = $('cxSteps');
  stepsEl.querySelector('.ar-reason-head').onclick = () => stepsEl.classList.toggle('open');
  wireSchedulePanel(r);
  wireCardToggle($('cxBuc'));
  $('explainBtn').onclick = explainInChat;
  $('handoffBtn').onclick = prepareHandoff;
}

// ── repayment schedule ───────────────────────────────────────────────────────
// Its own card below the results card, expanded by default showing the first
// year. It is a headline deliverable in its own right, not a footnote to the
// KPIs, so it does not live inside the amber "how we calculated this" trace.
// The first 12 months ship with the result; "show all" fetches the remaining
// months on demand (a 35y loan is 420 rows — not worth the payload by default),
// and the whole card can be collapsed once the customer is done with it.
function fmtCents(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString('en-SG', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function scheduleRowsHtml(rows) {
  return rows.map((s) =>
    `<tr>
       <td class="cxa-m">${s.month}</td>
       <td>${fmtCents(s.beginning_balance)}</td>
       <td>${fmtCents(s.instalment)}</td>
       <td class="cxa-int">${fmtCents(s.interest_paid)}</td>
       <td class="cxa-prin">${fmtCents(s.principal_paid)}</td>
       <td>${fmtCents(s.ending_balance)}</td>
     </tr>`).join('');
}

function renderSchedulePanel(r) {
  const rows = r.schedule || [];
  if (!rows.length) return '';
  const total = r.schedule_total_months || rows.length;
  const years = Math.round(total / 12);
  // Collapsed by default: with two schedule cards below the results, leading with
  // both expanded buries the KPIs the customer actually asked for.
  return (
    `<div class="calc-card sched-card" id="cxSched">
       <div class="calc-head sched-head">
         <span class="calc-title">Your payment schedule — month by month</span>
         <span class="sched-toggle"><span class="sched-toggle-txt">Show</span><span class="ar-caret">▾</span></span>
       </div>
       <div class="sched-body">
         <div class="cxa-intro">
           Every instalment is split between interest and repaying what you borrowed.
           Early on most of it is interest; over time the balance falls and more of each
           payment goes to principal.
         </div>
         <div class="cxa-scroll">
           <table class="cx-amort">
             <thead>
               <tr>
                 <th>Month</th><th>Beginning balance</th><th>Monthly instalment</th>
                 <th>Interest paid</th><th>Principal paid</th><th>Ending balance</th>
               </tr>
             </thead>
             <tbody id="cxSchedBody">${scheduleRowsHtml(rows)}</tbody>
           </table>
         </div>
         <div class="cxa-foot">
           <span id="cxSchedCount">Showing months 1–${rows.length} of ${total} (${years} years)</span>
           ${rows.length < total ? `<button class="chip cxa-more" id="cxSchedMore">Show all ${total} months</button>` : ''}
         </div>
       </div>
     </div>`);
}

// Shared Show/Hide behaviour for the collapsible cards below the results.
function wireCardToggle(el) {
  if (!el) return;
  const label = el.querySelector('.sched-toggle-txt');
  el.querySelector('.sched-head').onclick = () => {
    label.textContent = el.classList.toggle('open') ? 'Hide' : 'Show';
  };
}

function wireSchedulePanel(r) {
  const el = $('cxSched');
  if (!el) return;
  wireCardToggle(el);

  const more = $('cxSchedMore');
  if (!more) return;
  more.onclick = async () => {
    more.disabled = true; more.textContent = 'Loading…';
    try {
      const res = await (await fetch('/api/customer/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          eligible_loan: r.eligible_loan,
          interest_rate_pct: r.interest_rate_pct,
          loan_tenure_years: r.loan_tenure_years,
        }),
      })).json();
      const rows = res.schedule || [];
      if (!rows.length) throw new Error(res.error || 'No schedule returned');
      $('cxSchedBody').innerHTML = scheduleRowsHtml(rows);
      $('cxSchedCount').textContent =
        `Showing all ${rows.length} months (${Math.round(rows.length / 12)} years)`;
      more.remove();
    } catch (e) {
      more.disabled = false;
      more.textContent = 'Could not load — try again';
    }
  };
}

// ── BUC progressive payment schedule ─────────────────────────────────────────
// For a property still being built the developer is paid in statutory stages, so
// the bank releases the loan piecemeal and the instalment climbs with each
// drawdown. Collapsed by default (it only applies to new launches) and it cites
// the statute the stage percentages come from, because they are law, not a bank
// product — the customer can check them.
function renderBucPanel(r) {
  const buc = r.buc || {};
  const rows = buc.rows || [];
  if (!rows.length) return '';
  const t = buc.totals || {};
  const src = buc.source || {};

  const body = rows.map((s) => {
    const pct = s.pct_of_price != null ? s.pct_of_price.toFixed(2) + '%' : '—';
    const isLoan = (s.loan || 0) > 0;
    return `<tr class="${isLoan ? 'buc-loan' : 'buc-cash'}">
        <td class="buc-stage">${escapeHtml(s.stage)}</td>
        <td class="buc-time">${escapeHtml(s.timeframe || '')}</td>
        <td>${pct}</td>
        <td>${fmtMoney(s.amount_payable)}</td>
        <td class="buc-fund">${escapeHtml(s.funding)}</td>
        <td class="buc-mth">${s.monthly_repayment ? fmtMoney(s.monthly_repayment) : '—'}</td>
      </tr>`;
  }).join('');

  return (
    `<div class="calc-card sched-card" id="cxBuc">
       <div class="calc-head sched-head">
         <span class="calc-title">Buying a property under construction (BUC) — progressive payments</span>
         <span class="sched-toggle"><span class="sched-toggle-txt">Show</span><span class="ar-caret">▾</span></span>
       </div>
       <div class="sched-body">
         <div class="cxa-intro">
           If the home is still being built, you pay the developer in stages as construction
           progresses. Your own cash and CPF go first; the bank only starts releasing the loan
           once that is used up, so your monthly repayment <strong>starts at nothing and rises</strong>
           with each release, reaching ${fmtMoney(t.final_monthly_repayment)} once the full
           ${fmtMoney(t.loan_total)} is drawn.
         </div>
         <div class="cxa-scroll buc-scroll">
           <table class="cx-amort cx-buc">
             <thead>
               <tr>
                 <th>Stage of work</th><th>Timeframe</th><th>% of price</th>
                 <th>Amount payable</th><th>Paid from</th><th>Monthly repayment</th>
               </tr>
             </thead>
             <tbody>${body}</tbody>
           </table>
         </div>
         <div class="cxa-foot">
           <span>
             Cash + CPF ${fmtMoney(t.cash_cpf_total)} · Loan ${fmtMoney(t.loan_total)}
             ${t.loan_capped_by ? ' · capped by ' + escapeHtml(t.loan_capped_by.toLowerCase()) : ''}
           </span>
         </div>
         <div class="buc-note">
           Stage percentages are set by law, not by the bank — see
           <a href="${escapeHtml(src.url || '')}" target="_blank" rel="noopener">${escapeHtml(src.label || 'the statutory payment schedule')}</a>.
           The <strong>Paid from</strong> split is not prescribed by that schedule: it is estimated
           on the usual disbursement order, where your own cash and CPF are drawn down first and the
           loan starts only once they run out. Where the switch falls therefore depends on your own
           downpayment, so treat the split as indicative — your bank's letter of offer governs.
           Banks commonly charge interest only until the property is completed, so the figures
           above are the full instalment for each level of drawdown.
         </div>
       </div>
     </div>`);
}

function explainInChat() {
  const r = state.lastCalc;
  if (!r) return;
  const f = readForm();
  const ask = {
    price: `Target price S$${f.target_property_price}.`,
    instalment: `Monthly budget S$${f.monthly_budget}, cash+CPF set aside S$${f.cash_cpf_available}.`,
    downpayment: `Cash+CPF set aside S$${f.cash_cpf_available}.`,
    explore: `I wanted my maximum loan amount.`,
  }[state.mode] || '';
  const msg =
    `I used the calculator on this page (${MODES[state.mode].card}): age ${f.age}, ` +
    `fixed income S$${f.monthly_fixed_income}/mo, variable S$${f.monthly_variable_income}/mo, ` +
    `${f.nationality}, ${f.property_type} property, ${f.properties_owned_now} property(ies) owned now, ` +
    `rate ${f.interest_rate_pct}%. ${ask} ` +
    `It says max loan ${fmtMoney(r.eligible_loan)}, property ${fmtMoney(r.property_price)}, ` +
    `monthly ${fmtMoney(r.monthly_repayment)}, TDSR ${r.tdsr_pct}%, cash+CPF needed ${fmtMoney(r.required_cash_cpf)}. ` +
    `Can you explain in plain language what's limiting my loan and what these numbers mean?`;
  sendMessage(msg);
}

// ── handoff to a human specialist (honest closed loop) ───────────────────────
// We do NOT pretend to send anything. This assembles a structured, copy-ready
// summary for the customer to hand to a mortgage specialist — the last hop
// (actually contacting a human) stays explicitly with the person. The summary is
// built locally from the SAME calculator result shown on the card (no LLM), so
// the figures can never drift from what the customer just saw, and it costs
// nothing. It matches the system's core stance: human-in-the-loop, no claims
// that aren't true (see the honest-referral / no-hallucination design line).
function buildHandoffSummary() {
  const r = state.lastCalc;
  if (!r) return null;
  const f = readForm();
  const mode = r.mode || state.mode;
  const money = (n) => 'S$' + Math.round(n || 0).toLocaleString('en-SG');

  // Written in the customer's own voice — like introducing their situation to a
  // specialist. Figures come from the SAME calculator result on the card.
  const intent = {
    price:       `I have a home in mind at around ${money(f.target_property_price)} and I'd like to know what it would take to buy it.`,
    instalment:  `I'm comfortable paying about ${money(f.monthly_budget)} a month, and I'd like to know what price that supports.`,
    downpayment: `I've set aside about ${money(f.cash_cpf_available)} in cash and CPF, and I'd like to know what budget that gives me.`,
    explore:     `I'm still exploring, and I wanted to get a sense of the most I could borrow.`,
  }[mode] || `I've been looking into what I could afford for a home loan.`;

  const ask = {
    price:       `I'd like to speak with a specialist to go over the details and my next steps.`,
    instalment:  `I'd like to speak with a specialist to go over the details and my options.`,
    downpayment: `I'd like to speak with a specialist to go over the details and my options.`,
    explore:     `I'd like to speak with a specialist to talk through my options from here.`,
  }[mode] || `I'd like to speak with a specialist to talk through my options from here.`;

  // Plain-text body so it copies cleanly into email or a message.
  const L = [];
  L.push(`Hi, I'm exploring a home loan — here's where I've got to.`);
  L.push('');
  L.push(intent);
  L.push('');
  L.push('A bit about me:');
  L.push(`  • I'm ${f.age}, ${f.nationality.toLowerCase()}, looking at a ${f.property_type.toLowerCase()} property.`);
  L.push(`  • My income is ${money(f.monthly_fixed_income)}/month` +
         (f.monthly_variable_income ? `, plus about ${money(f.monthly_variable_income)}/month variable.` : '.'));
  L.push(`  • I currently own ${f.properties_owned_now} propert${f.properties_owned_now === 1 ? 'y' : 'ies'}` +
         (f.outstanding_home_loans ? ` with ${f.outstanding_home_loans} home loan(s) outstanding.` : ' with no home loans outstanding.'));
  const commit = (f.monthly_car_loan || 0) + (f.monthly_other || 0);
  if (commit) L.push(`  • I have about ${money(commit)}/month in other repayments.`);
  L.push('');
  L.push(`Here's what the calculator on your website showed me (indicative, at ${r.interest_rate_pct}% p.a.):`);
  L.push(`  • I could borrow up to ${money(r.eligible_loan)}` + (r.binding_constraint ? `, limited by ${String(r.binding_constraint).toLowerCase()}.` : '.'));
  L.push(`  • That works out to a ${mode === 'price' ? 'price' : 'budget'} of ${money(r.property_price)}, needing ${money(r.required_cash_cpf)} in cash and CPF up front.`);
  L.push(`  • My monthly repayment would be about ${money(r.monthly_repayment)}` +
         (r.tdsr_pct != null ? `, and my TDSR came to ${r.tdsr_pct}%${r.tdsr_pct > 55 ? ' — which I think is over the limit.' : '.'}` : '.'));
  L.push('');
  L.push(ask);
  L.push('');
  L.push('Thank you!');
  return L.join('\n');
}

function prepareHandoff() {
  const summary = buildHandoffSummary();
  if (!summary) return;
  state.convo.push({ kind: 'handoff', text: summary, sent: false });
  renderChat();
  $('chat').scrollTop = $('chat').scrollHeight;
}

// Pseudo-send: flips the card to a handed-off state and shows a toast. This is a
// PREVIEW stub — there is no specialist-routing backend yet, so both the toast
// and the card carry an explicit "in development" label. It demos the intended
// closed loop without claiming a message was actually delivered.
function sendHandoff(msg) {
  msg.sent = true;
  renderChat();
  showToast('Summary sent — a specialist will follow up', 'A mortgage specialist will be in touch · preview — routing in development');
}

function showToast(title, sub) {
  let host = $('toastHost');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toastHost';
    host.className = 'toast-host';
    document.body.appendChild(host);
  }
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML =
    `<span class="toast-check">✓</span>` +
    `<div class="toast-txt"><div class="toast-t">${escapeHtml(title.replace(/^✓\s*/, ''))}</div>` +
    (sub ? `<div class="toast-s">${escapeHtml(sub)}</div>` : '') + `</div>`;
  host.appendChild(t);
  // fade in, hold, fade out, remove
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 300); }, 3600);
}

// ── chat (restricted customer assistant) ────────────────────────────────────
const SUGGESTED = [
  'What fixed-rate packages do you have?',
  'me and my wife apply together, two gifts or one',
  'is refinancing an existing home loan from another bank eligible for this promotion',
  'How much can I borrow on a S$9,000 salary?',
];

function renderChips() {
  const el = $('chips');
  el.innerHTML = '';
  SUGGESTED.forEach((q) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = q;
    b.onclick = () => sendMessage(q);
    el.appendChild(b);
  });
}

function renderChat() {
  const chat = $('chat');
  chat.innerHTML = '';
  if (!state.convo.length) {
    chat.innerHTML = `<div class="chat-empty">Ask about our loan terms &amp; conditions, packages, or what you could afford. I can't access any account data — and my answers are guidance, not offers.</div>`;
    return;
  }
  state.convo.forEach((msg) => {
    const d = document.createElement('div');
    if (msg.kind === 'think') {
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
      d.querySelector('.think-head').onclick = () => { msg.open = !msg.open; renderChat(); };
    } else if (msg.kind === 'agent') {
      d.className = 'msg agent';
      d.innerHTML = `<div class="bubble md">${renderMarkdown(msg.text || '')}</div>${renderSources(msg.sources)}`;
    } else if (msg.kind === 'note') {
      d.className = 'msg note';
      d.innerHTML = `<div class="bubble">${escapeHtml(msg.text || '')}</div>`;
    } else if (msg.kind === 'handoff') {
      d.className = 'msg handoff';
      const subject = 'Home loan enquiry — indicative summary';
      const mailto = 'mailto:?subject=' + encodeURIComponent(subject) +
        '&body=' + encodeURIComponent(msg.text || '');
      // Compact card for the narrow assistant pane: a one-line lead + the full
      // message tucked into a collapsible preview + actions. Two states:
      // (1) review — send/copy/email; (2) handed-off — a slim confirmation.
      // The "Send" hop is a preview stub (no real routing backend), labelled
      // "in development" wherever it appears.
      const preview =
        `<details class="ho-details"${msg.open ? ' open' : ''}>
           <summary>Preview message</summary>
           <pre class="ho-body">${escapeHtml(msg.text || '')}</pre>
         </details>`;
      const foot = msg.sent
        ? `<div class="ho-confirm"><span class="ho-check">✓</span>
             <span>A mortgage specialist will follow up <em>· preview, routing in development</em></span></div>
           ${preview}
           <div class="ho-actions">
             <button class="chip ho-copy">Copy</button>
             <a class="chip ho-mail" href="${mailto}">Email ›</a>
           </div>`
        : `<div class="ho-lead">Your summary is ready — a mortgage specialist can take it from here.</div>
           ${preview}
           <div class="ho-actions">
             <button class="chip primary ho-send">Send to a specialist ›</button>
             <button class="chip ho-copy">Copy</button>
             <a class="chip ho-mail" href="${mailto}">Email ›</a>
           </div>
           <div class="ho-dev">Sending is a preview — specialist routing is in development.</div>`;
      d.innerHTML =
        `<div class="ho-card${msg.sent ? ' sent' : ''}">
           <div class="ho-head">${msg.sent ? '✓ A specialist will follow up' : '📋 Summary for a specialist'}</div>
           ${foot}
         </div>`;
      const det = d.querySelector('.ho-details');
      if (det) det.addEventListener('toggle', () => { msg.open = det.open; });
      const sendBtn = d.querySelector('.ho-send');
      if (sendBtn) sendBtn.onclick = () => sendHandoff(msg);
      const copyBtn = d.querySelector('.ho-copy');
      copyBtn.onclick = () => {
        const done = () => { copyBtn.textContent = 'Copied ✓'; setTimeout(() => { copyBtn.textContent = 'Copy summary'; }, 1800); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(msg.text || '').then(done, () => fallbackCopy(msg.text || '', done));
        } else { fallbackCopy(msg.text || '', done); }
      };
    } else {
      d.className = 'msg rm';
      d.innerHTML = `<div class="bubble">${escapeHtml(msg.text || '')}</div>`;
    }
    chat.appendChild(d);
  });
  chat.scrollTop = chat.scrollHeight;
}

function renderSources(sources) {
  if (!sources || !sources.length) return '';
  const tags = sources.map((s) => {
    const doc = (s.source || '').replace(/\.pdf$/i, '').replace(/[-_]+/g, ' ')
      .replace(/\btncs?\b/i, 'T&C').replace(/\b\w/g, (c) => c.toUpperCase()).trim();
    return `<span class="src-tag">${escapeHtml(s.clause ? `${doc} · Clause ${s.clause}` : doc)}</span>`;
  }).join('');
  return `<div class="src-strip"><span class="src-label">Sources</span>${tags}</div>`;
}

async function sendMessage(text) {
  text = (text || $('input').value).trim();
  if (!text || state.streaming) return;
  $('input').value = '';
  state.convo.push({ kind: 'rm', text });
  renderChat();

  state.streaming = true;
  $('sendBtn').disabled = true;
  let openThink = null;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ applicant_id: 'GUEST', stage: 'CUSTOMER', message: text, role: 'customer' }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Starlette emits CRLF frame terminators — match either (see app.js).
      let m;
      const sepRe = /\r?\n\r?\n/;
      while ((m = sepRe.exec(buf)) !== null) {
        const frame = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        const line = frame.split(/\r?\n/).find((l) => l.startsWith('data:'));
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line.slice(line.indexOf(':') + 1).trim()); }
        catch (err) { continue; }
        openThink = handleEvent(ev, openThink);
        renderChat();
      }
    }
  } catch (e) {
    state.convo.push({ kind: 'note', text: 'Connection error: ' + e.message });
    renderChat();
  } finally {
    state.streaming = false;
    $('sendBtn').disabled = false;
  }
}

function handleEvent(ev, openThink) {
  switch (ev.event) {
    case 'thinking_open': {
      const block = { kind: 'think', title: ev.title, done: false, open: false };
      state.convo.push(block);
      return block;
    }
    case 'thinking_close':
      if (openThink) {
        openThink.done = true;
        openThink.title = ev.title || openThink.title;
        if (ev.duration != null) openThink.duration = ev.duration;
        if (ev.summary) openThink.summary = ev.summary;
      }
      return null;
    case 'answer':
      state.convo.push({ kind: 'agent', text: ev.text, sources: ev.sources || [] });
      return openThink;
    case 'error':
      state.convo.push({ kind: 'note', text: '⚠️ ' + ev.text });
      return openThink;
    default:
      return openThink;   // tool_call / tool_result / routed etc. — no log pane here
  }
}

// ── minimal markdown (same no-eval renderer as app.js; CSP-safe) ─────────────
function renderMarkdown(src) {
  const inline = (t) => escapeHtml(t)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
  const lines = (src || '').replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let i = 0, para = [];
  const flushPara = () => {
    if (para.length) { out.push('<p>' + para.map(inline).join('<br>') + '</p>'); para = []; }
  };
  while (i < lines.length) {
    const line = lines[i];
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
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) { flushPara(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
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
    if (line.trim() === '') { flushPara(); i++; continue; }
    para.push(line);
    i++;
  }
  flushPara();
  return out.join('');
}

// ── wire up ──────────────────────────────────────────────────────────────────
function init() {
  document.querySelectorAll('.cx-intent').forEach((b) => {
    b.onclick = () => { setMode(b.dataset.mode); goStep(2); };
  });
  // Step labels are clickable when already reached (back-navigation only).
  document.querySelectorAll('.cx-step').forEach((b) => {
    b.onclick = () => {
      const s = parseInt(b.dataset.step, 10);
      if (s < state.step) goStep(s);
    };
  });
  $('backBtn').onclick = () => goStep(1);
  $('adjustBtn').onclick = () => goStep(2);
  $('restartBtn').onclick = () => goStep(1);
  $('calcBtn').onclick = runCalc;
  $('sendBtn').onclick = () => sendMessage();
  $('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  setMode('price');
  goStep(1);
  renderAssetRows();
  renderChips();
  renderChat();
  loadPackages();
}

document.addEventListener('DOMContentLoaded', init);
