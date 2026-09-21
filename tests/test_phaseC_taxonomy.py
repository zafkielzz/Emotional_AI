# -*- coding: utf-8 -*-
"""Phase C regression tests: appraisal taxonomy must NOT conflate a coercive drug/extortion
demand with a life-threatening violent attack (the model over-tagged an armed morphine demand
as "attack" / TRAUMA, which prematurely tripped the evidence gate before any real violence)."""

from sandbox.appraisal_engine import DeepAppraisalEngine
from sandbox.formal_state import EventSeverity

ENGINE = DeepAppraisalEngine()  # model=None -> deterministic rule-based path


def test_drug_demand_is_suspicious_request_major_not_trauma():
    # Plain coercion to hand over controlled drugs -- no explicit violence in the utterance.
    appr = ENGINE.evaluate_rule_based(
        character_name="Alice", character_role="Bác sĩ",
        speaker_name="Người chơi",
        utterance="Giao hết morphine và thuốc giảm đau ra đây ngay, cấm báo cho ai!"
    )
    assert appr["status"] == "SUCCESS"
    assert appr["pattern_tag"] == "suspicious_request"
    assert appr["detected_severity"] == EventSeverity.MAJOR
    assert -0.6 <= appr["delta_valence"] < 0.0


def test_urgent_demand_without_violence_is_not_attack():
    appr = ENGINE.evaluate_rule_based(
        character_name="Alice", character_role="Bác sĩ",
        speaker_name="Người chơi",
        utterance="Đưa morphine ra đây ngay lập tức! Mở tủ thuốc ra cho tao mau lên!"
    )
    assert appr["pattern_tag"] in ("suspicious_request", "coercion")
    assert appr["detected_severity"] in (EventSeverity.MAJOR, EventSeverity.MINOR)
    assert appr["detected_severity"] != EventSeverity.TRAUMA


def test_explicit_violent_threat_is_trauma_attack():
    appr = ENGINE.evaluate_rule_based(
        character_name="Alice", character_role="Bác sĩ",
        speaker_name="Người chơi",
        utterance="Không đưa tao đập nát trạm xá này và giết mày!"
    )
    assert appr["pattern_tag"] == "attack"
    assert appr["detected_severity"] == EventSeverity.TRAUMA


def test_police_extortion_keyword_no_longer_trauma():
    # "cướp" alone (e.g. a person being robbed) stays an attack/MAJOR-ish threat but the plain
    # drug handover phrase must not alone be TRAUMA.
    appr = ENGINE.evaluate_rule_based(
        character_name="Bob", character_role="Thợ săn đồ phế liệu",
        speaker_name="Người chơi",
        utterance="Tao chỉ đùa thôi, cho tao xin ít bông băng với thuốc đỏ được không?"
    )
    assert appr["pattern_tag"] in ("help", "dialogue")
    assert appr["delta_valence"] >= -0.1
