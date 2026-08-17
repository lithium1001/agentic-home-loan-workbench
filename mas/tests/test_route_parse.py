"""Guards the orchestrator's routing-JSON parser against leaking raw model text
into the RM chat.

Regression: the model sometimes prepends noise (e.g. a stray "SS") before the
routing JSON, which broke json.loads and made the fallback echo the raw blob —
so the routing JSON leaked into the RM's chat bubble, and the corrupted "previous
reply" then destabilised the next turn's routing. _parse_route now carves out the
first balanced {...} object, and on a genuine parse failure returns a CLEAN
message, never the raw text. No LLM / network.
"""

from graph import _parse_route, _first_json_object


def test_recovers_json_with_leading_noise():
    raw = ('SS\n{"intent":"draft IPA letter","stage":"IPA","agent":"full_ipa",'
           '"applicant_id":"APP0001","question":"Draft the IPA letter"}')
    d = _parse_route(raw)
    assert d["agent"] == "full_ipa"
    assert d["stage"] == "IPA"
    assert d["applicant_id"] == "APP0001"


def test_clean_json_passes_through():
    d = _parse_route('{"agent":"policy_product","stage":"LO"}')
    assert d["agent"] == "policy_product"


def test_trailing_junk_is_tolerated():
    d = _parse_route('{"agent":"borrower_profile","stage":"IPA"}  \n done.')
    assert d["agent"] == "borrower_profile"


def test_nested_braces_in_field():
    d = _parse_route('{"agent":"full_lo","question":"use {rate} of 1.2%"}')
    assert d["agent"] == "full_lo"


def test_unparseable_gives_clean_fallback_not_raw():
    raw = "total nonsense no json here"
    d = _parse_route(raw)
    assert d["agent"] == "none"
    # The raw model text must NEVER be surfaced as the answer.
    assert raw not in d.get("answer", "")
    assert "rephrase" in d["answer"].lower()


def test_first_json_object_helper():
    assert _first_json_object("no braces") == ""
    assert _first_json_object('x {"a":1} y') == '{"a":1}'
    assert _first_json_object('{"a":{"b":2}} tail') == '{"a":{"b":2}}'
