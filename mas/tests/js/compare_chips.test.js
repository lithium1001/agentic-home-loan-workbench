// Behaviour test for the Package Comparison track's action chips, which are
// interpolated in the browser from the panel's live inputs (app.js compareChips).
//
// Driven by tests/test_compare_chips_js.py so it runs inside the normal `py -m
// pytest`; runnable on its own with `node tests/js/compare_chips.test.js`.
//
// app.js is a plain browser script with no module exports, so rather than
// duplicating the logic here (which would test a copy, not the shipped code) this
// extracts the functions under test from the real file and runs them against a
// stub $() standing in for the panel's DOM inputs. Node only, no jsdom.
'use strict';

const fs = require('fs');
const path = require('path');

const APP_JS = path.join(__dirname, '..', '..', 'server', 'static', 'app.js');
const src = fs.readFileSync(APP_JS, 'utf8');

// Slice out a top-level `function name(...) { ... }` by brace matching.
function grab(name) {
  const start = src.indexOf('function ' + name + '(');
  if (start < 0) throw new Error(`app.js no longer defines ${name}()`);
  let depth = 0;
  for (let k = src.indexOf('{', start); k < src.length; k++) {
    if (src[k] === '{') depth++;
    else if (src[k] === '}' && --depth === 0) return src.slice(start, k + 1);
  }
  throw new Error(`unbalanced braces reading ${name}() out of app.js`);
}

const CMP_FIELDS = eval(src.match(/const CMP_FIELDS = (\[[\s\S]*?\n\]);/)[1]);

// The panel's inputs, id -> current value. A field absent here stands for "the
// panel has not been built yet"; an empty string stands for a cleared box.
let fields = {};
globalThis.$ = (id) =>
  (id in fields ? { value: String(fields[id]), step: '1', min: '0' } : null);

// compareChips() reads the served chips off the case payload for its fallback.
const SERVED = [
  { label: 'served-chip-1', primary: false },
  { label: 'served-chip-2', primary: false },
];
globalThis.state = { caseData: { actionChips: { COMPARE: SERVED } } };

eval([grab('fmtMoney'), grab('readCompare'), grab('compareChips')].join('\n') +
     '\nglobalThis.compareChips = compareChips;');

const defaults = () =>
  Object.fromEntries(CMP_FIELDS.map((f) => [f.id, f.value]));
const isServed = (chips) => chips === SERVED;
const labels = (chips) => chips.map((c) => (typeof c === 'string' ? c : c.label));

// ── tiny harness ───────────────────────────────────────────────────────────
let failed = 0;
let group = '';
function describe(name, fn) { group = name; console.log(`\n${name}`); fn(); }
function it(name, fn) {
  try { fn(); console.log(`  ok    ${name}`); }
  catch (e) { failed++; console.log(`  FAIL  ${name}\n          ${e.message}`); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }

// The six figures the panel holds, as each should appear in a prompt.
const ALL_SIX = (o) => [o.loan, o.rate, o.tenure, o.after, o.rateA, o.rateB];
const DEFAULT_SIX = { loan: 'S$1,200,000', rate: '2.00%', tenure: '360 months',
                      after: '3 months', rateA: '1.55%', rateB: '1.50%' };
const EDITED_FIELDS = { cmpLoan: 850000, cmpRate: 3.25, cmpTenure: 240,
                        cmpAfter: 6, cmpRateA: 2.4, cmpRateB: 2.15 };
const EDITED_SIX = { loan: 'S$850,000', rate: '3.25%', tenure: '240 months',
                     after: '6 months', rateA: '2.40%', rateB: '2.15%' };

// ── every prompt carries all six inputs ────────────────────────────────────
// The point of the split: a prompt missing a figure costs a round-trip, because
// the agent stops to ask the RM for a number that is already on screen.
describe('prompts carry every panel input', () => {
  it('states all six figures at the panel defaults, in every chip', () => {
    fields = defaults();
    for (const { prompt, label } of compareChips()) {
      ALL_SIX(DEFAULT_SIX).forEach(
        (t) => assert(prompt.includes(t), `"${t}" missing from chip "${label}"`));
    }
  });

  it('follows the RM after every input changes, in every chip', () => {
    fields = EDITED_FIELDS;
    for (const { prompt, label } of compareChips()) {
      ALL_SIX(EDITED_SIX).forEach(
        (t) => assert(prompt.includes(t), `"${t}" missing from chip "${label}"`));
    }
  });

  it('leaves no default behind when the inputs change', () => {
    fields = EDITED_FIELDS;
    // The bug this guards is a prompt built from a mix of live and hard-coded terms.
    for (const { prompt } of compareChips()) {
      ['1,200,000', '2.00%', '360', '1.55%'].forEach(
        (stale) => assert(!prompt.includes(stale), `stale "${stale}" in: ${prompt}`));
    }
  });
});

// ── the button face stays short ────────────────────────────────────────────
describe('labels are short enough for the narrow pane', () => {
  it('keeps every label well under the prompt length', () => {
    fields = defaults();
    for (const { label, prompt } of compareChips()) {
      assert(label.length <= 42, `label too long (${label.length}): ${label}`);
      assert(prompt.length > label.length * 2,
             `prompt is not carrying more than the label: ${label}`);
    }
  });

  it('keeps figures OUT of the label, so it never goes stale on screen', () => {
    fields = defaults();
    for (const { label } of compareChips()) {
      assert(!/S\$|\d+\s*months|\d+\.\d+%/.test(label),
             `label states a figure and will look stale: ${label}`);
    }
  });

  it('gives every chip both a label and a prompt', () => {
    fields = defaults();
    for (const c of compareChips()) {
      assert(typeof c.label === 'string' && c.label.trim(), 'missing label');
      assert(typeof c.prompt === 'string' && c.prompt.trim(), 'missing prompt');
    }
  });
});

// ── inputs only, never the computed result ─────────────────────────────────
describe('chips carry inputs only', () => {
  it('never quotes a computed saving', () => {
    // Interpolating the panel's savings would leave the assistant restating
    // arithmetic it never ran, breaking the tool-call audit trail.
    fields = defaults();
    for (const { prompt } of compareChips()) {
      assert(!/sav(es|ings) (of|:)\s*S\$/i.test(prompt), `quotes a saving: ${prompt}`);
      // The outstanding balance is the only money figure the panel feeds in; any
      // other S$ amount would be a computed result leaking into the prompt.
      for (const money of prompt.match(/S\$[\d,]*\d/g) || []) {
        assert(money === 'S$1,200,000', `unexpected money "${money}" in: ${prompt}`);
      }
    }
  });
});

// ── the empty / half-typed panel ───────────────────────────────────────────
describe('fallback when the panel has no usable loan', () => {
  it('falls back before the panel is built', () => {
    fields = {};                       // $() returns null for every field
    assert(isServed(compareChips()), 'built an "S$0 loan" prompt instead');
  });

  it('falls back while the loan box is cleared', () => {
    fields = { ...defaults(), cmpLoan: '' };
    assert(isServed(compareChips()), 'built an "S$0 loan" prompt instead');
  });

  it('falls back while the rate box is cleared', () => {
    fields = { ...defaults(), cmpRate: '' };
    assert(isServed(compareChips()), 'built an "at 0.00%" prompt instead');
  });

  it('falls back while the tenure box is cleared', () => {
    fields = { ...defaults(), cmpTenure: '' };
    assert(isServed(compareChips()), 'built a "0 months left" prompt instead');
  });

  it('does not fall back on a 0% target rate, which is a real input', () => {
    fields = { ...defaults(), cmpRateA: 0 };
    const chips = compareChips();
    assert(!isServed(chips), 'treated a legitimate 0% target rate as "not ready"');
    assert(chips[0].prompt.includes('0.00% p.a.'), chips[0].prompt);
  });

  it('does not fall back on converting now (0 months wait)', () => {
    fields = { ...defaults(), cmpAfter: 0 };
    const chips = compareChips();
    assert(!isServed(chips), 'treated "convert now" as "not ready"');
    assert(chips[0].prompt.includes('0 months'), chips[0].prompt);
  });

  it('returns something sendable in every state', () => {
    // renderChips() falls back to `label` when a chip carries no `prompt`, so the
    // served (fallback) chips must survive this too.
    for (const f of [{}, defaults(), { ...defaults(), cmpLoan: '' }]) {
      fields = f;
      const chips = compareChips();
      assert(Array.isArray(chips) && chips.length, 'empty chip row');
      for (const c of chips) {
        const label = typeof c === 'string' ? c : c.label;
        const sent = (typeof c === 'object' && c.prompt) || label;
        assert(label && label.trim(), 'blank chip label');
        assert(sent && sent.trim(), 'chip would send an empty message');
      }
    }
  });
});

console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);
