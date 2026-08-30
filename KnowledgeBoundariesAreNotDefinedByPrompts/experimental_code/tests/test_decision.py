

from qprobe.backends import NLILabel
from qprobe.decision import DecisionKind, decide_recovery
from qprobe.semantic import analyze_prompt_semantics
from qprobe.views import PromptView


def make_view(view_id: str, answer: str) -> PromptView:
    semantics = analyze_prompt_semantics(((None,),))
    return PromptView(
        view_id=view_id,
        prompt=f"Prompt {view_id}",
        answers=(answer,),
        semantics=semantics,
    )


def test_singleton_conflict_does_not_veto_qualified_group() -> None:
    views = (make_view("a1", "A"), make_view("a2", "A"), make_view("b1", "B"))
    labels = (
        (None, NLILabel.ENTAILMENT, NLILabel.CONTRADICTION),
        (NLILabel.ENTAILMENT, None, NLILabel.CONTRADICTION),
        (NLILabel.CONTRADICTION, NLILabel.CONTRADICTION, None),
    )

    decision = decide_recovery(views, labels, uncertainty_threshold=1.0)

    assert decision.kind == DecisionKind.RECOVERED
    assert decision.supporting_view_ids == ("a1", "a2")


def test_two_qualified_conflicting_groups_abstain() -> None:
    views = tuple(make_view(name, name[0].upper()) for name in ("a1", "a2", "b1", "b2"))
    labels = (
        (None, NLILabel.ENTAILMENT, NLILabel.CONTRADICTION, NLILabel.CONTRADICTION),
        (NLILabel.ENTAILMENT, None, NLILabel.CONTRADICTION, NLILabel.CONTRADICTION),
        (NLILabel.CONTRADICTION, NLILabel.CONTRADICTION, None, NLILabel.ENTAILMENT),
        (NLILabel.CONTRADICTION, NLILabel.CONTRADICTION, NLILabel.ENTAILMENT, None),
    )

    decision = decide_recovery(views, labels, uncertainty_threshold=1.0)

    assert decision.kind == DecisionKind.ABSTAIN
    assert decision.reason == "multiple_competing_stable_answer_groups"
