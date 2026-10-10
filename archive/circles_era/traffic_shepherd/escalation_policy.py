"""One-wall report-only scheduler, disabled without a verified qualification bundle.

Artifact loading checks frozen identities, recorded numerical criteria and their
provenance. It does not itself manufacture physical validation or calibration.
No accepted qualification artifact is supplied with this implementation.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from math import exp, isfinite
from pathlib import Path
from types import MappingProxyType

from .formation import FormationController, FormationProfile, authorize_qualified_gates
from .paired_experiments import PolicyInput, PolicyOutput, current_identity
from .proposal_features import SCHEMA, feature_packet, latest_received, select_proposal


CHECKS = {
    "controller_integrity": "implementation_verification",
    "formation_physical": "physical_simulation",
    "incremental_benefit_physical": "matched_physical_simulation",
    "observation_label_audit": "acquired_report_and_offline_label_audit",
    "episode_split_audit": "archive_exposure_and_episode_split_audit",
    "gate_calibration": "held_out_episode_calibration",
    "prospective_freeze": "preregistered_policy_freeze",
}
OPERATORS = {"<": lambda x,y: x < y, "<=": lambda x,y: x <= y,
             ">": lambda x,y: x > y, ">=": lambda x,y: x >= y, "==": lambda x,y: x == y}
_LOADED = object()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in {int, float} and isfinite(value)


def frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: frozen(v) for k,v in value.items()})
    if isinstance(value, list):
        return tuple(frozen(v) for v in value)
    return value


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _referenced(root, reference):
    if set(reference) != {"path", "sha256"} or not isinstance(reference["path"], str):
        raise ValueError("a relative artifact path and exact SHA256 are required")
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("qualification references must be existing files inside the frozen bundle")
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("qualification artifact bytes changed")
    return _read_json(path)


@dataclass(frozen=True)
class SchedulerConfig:
    required_lanes: tuple[str, ...]
    opportunity_times_s: tuple[int, ...]
    cooldown_s: int
    maximum_attempts: int
    recruitment_band_m: tuple[float, float] = (200., 1900.)
    history_s: float = 20.
    maximum_members_per_lane: int = 16
    version: str = "one_wall_report_scheduler/1"

    def __post_init__(self):
        if not isinstance(self.required_lanes, tuple) or not 1 <= len(self.required_lanes) <= 2:
            raise ValueError("one or two immutable static map lanes required")
        if len(set(self.required_lanes)) != len(self.required_lanes) or any(not lane for lane in self.required_lanes):
            raise ValueError("distinct named map lanes required")
        if (not isinstance(self.opportunity_times_s, tuple) or not self.opportunity_times_s
                or any(type(t) is not int or t < 0 for t in self.opportunity_times_s)
                or tuple(sorted(set(self.opportunity_times_s))) != self.opportunity_times_s):
            raise ValueError("distinct ascending predeclared opportunity times required")
        if type(self.cooldown_s) is not int or self.cooldown_s < 1:
            raise ValueError("explicit positive control-grid cooldown required")
        if type(self.maximum_attempts) is not int or self.maximum_attempts < 1:
            raise ValueError("explicit positive action budget required")
        if not isinstance(self.recruitment_band_m, tuple) or len(self.recruitment_band_m) != 2:
            raise ValueError("explicit immutable recruitment band required")
        lo, hi = self.recruitment_band_m
        if not all(finite(x) for x in (lo, hi, self.history_s)) or not 0 <= lo < hi or self.history_s < 1:
            raise ValueError("invalid recruitment band or retained report history")
        if type(self.maximum_members_per_lane) is not int or not 1 <= self.maximum_members_per_lane <= 16:
            raise ValueError("bounded candidate enumeration required")


@dataclass(frozen=True)
class GateApproval:
    kind: str
    approved: bool
    abstained: bool
    reason: str
    probability: float | None
    decision_time_s: float
    proposal_sha256: str
    evidence_sha256: str
    assessment_evidence_sha256: str
    feature_row_sha256: str
    artifact_sha256: str
    qualification_sha256: str
    method: str


@dataclass(frozen=True, init=False)
class FrozenGate:
    """Only declared analytical/rule/logistic artifacts; no online manual scores."""
    kind: str
    method: str
    data: object
    sha256: str

    def __init__(self, data, artifact_sha256):
        if data.get("schema") != "frozen_escalation_gate/1" or data.get("feature_schema") != SCHEMA:
            raise ValueError("unsupported gate or acquired-feature schema")
        object.__setattr__(self, "kind", data["kind"])
        object.__setattr__(self, "method", data["method"])
        if self.kind not in {"feasibility", "benefit"} or self.method not in {
                "analytical_formation", "conjunction_rule", "logistic"}:
            raise ValueError("unsupported gate method")
        if self.method == "analytical_formation" and self.kind != "feasibility":
            raise ValueError("analytical formation is not an incremental-benefit model")
        names = data["feature_names"]
        if not isinstance(names, list) or len(names) != len(set(names)) or any(not isinstance(n, str) for n in names):
            raise ValueError("distinct feature names required")
        for name in names + data["required_true_features"]:
            if name in {"seed", "episode_id", "time_s", "penetration", "loading", "density", "health", "driver_type"}:
                raise ValueError("privileged or episode metadata cannot be gate inputs")
        params = data["parameters"]
        if self.method == "logistic":
            if not 1 <= len(names) <= 256 or set(params) != {"coefficients", "intercept", "means", "scales", "threshold"}:
                raise ValueError("explicit small logistic coefficients/scaler/threshold required")
            if any(len(params[k]) != len(names) for k in ("coefficients", "means", "scales")):
                raise ValueError("logistic dimensions differ from frozen features")
            values = params["coefficients"]+params["means"]+params["scales"]+[params["intercept"], params["threshold"]]
            if any(not finite(x) for x in values) or min(params["scales"]) <= 0 or not 0 <= params["threshold"] <= 1:
                raise ValueError("finite frozen logistic parameters and positive scales required")
            if data.get("provenance_kind") != "trained_from_offline_matched_physical_pairs":
                raise ValueError("manually supplied score cannot be presented as a fitted model")
        elif self.method == "conjunction_rule":
            if data.get("provenance_kind") != "preregistered_transparent_rule" or set(params) != {"rules"} or not params["rules"]:
                raise ValueError("declare a nonempty transparent rule; it is not a fitted predictor")
            for rule in params["rules"]:
                if set(rule) != {"feature", "operator", "value"} or rule["feature"] not in names:
                    raise ValueError("rule must use declared acquired features")
                if rule["operator"] not in OPERATORS or not finite(rule["value"]):
                    raise ValueError("invalid frozen numerical rule")
        else:
            if set(params) != {"minimum_estimated_runway_slack_m", "maximum_estimated_assembly_s"}:
                raise ValueError("declare analytical screening bounds explicitly")
            if any(not finite(v) for v in params.values()) or params["maximum_estimated_assembly_s"] <= 0:
                raise ValueError("invalid analytical screening bounds")
            if data.get("provenance_kind") != "qualified_analytical_rule":
                raise ValueError("analytical screening needs physical qualification")
        object.__setattr__(self, "data", frozen(data))
        object.__setattr__(self, "sha256", artifact_sha256)

    def decide(self, packet, assessment, qualification_sha256):
        values = packet["values"]
        binding = dict(kind=self.kind, decision_time_s=packet["now_s"], proposal_sha256=packet["proposal_sha256"],
            evidence_sha256=packet["evidence_sha256"], assessment_evidence_sha256=packet["assessment_evidence_sha256"],
            feature_row_sha256=packet["row_sha256"], artifact_sha256=self.sha256,
            qualification_sha256=qualification_sha256, method=self.method)
        def result(approved=False, abstained=False, reason="", probability=None):
            return GateApproval(approved=approved, abstained=abstained, reason=reason, probability=probability, **binding)
        if any(n not in values or not finite(values[n]) for n in self.data["feature_names"]):
            return result(abstained=True, reason="required_feature_unavailable")
        if any(values.get(n) != 1. for n in self.data["required_true_features"]):
            return result(abstained=True, reason="required_observation_presence_unavailable")
        params = self.data["parameters"]
        if self.method == "analytical_formation":
            if not assessment.approved:
                return result(abstained=assessment.status in {"insufficient_information", "insufficient_members"},
                              reason=assessment.status+":"+assessment.reason)
            slack, assembly = assessment.estimated_runway_slack_m, assessment.estimated_assembly_s
            if not finite(slack) or not finite(assembly):
                return result(abstained=True, reason="analytical_estimate_unavailable")
            approved = slack >= params["minimum_estimated_runway_slack_m"] and assembly <= params["maximum_estimated_assembly_s"]
            return result(approved=approved, reason="qualified_analytical_rule" if approved else "analytical_screen_rejected")
        if self.method == "conjunction_rule":
            approved = all(OPERATORS[r["operator"]](values[r["feature"]], r["value"]) for r in params["rules"])
            return result(approved=approved, reason="declared_rule_approved" if approved else "declared_rule_rejected")
        z = params["intercept"]+sum(w*(values[n]-m)/s for n,w,m,s in zip(self.data["feature_names"],
            params["coefficients"], params["means"], params["scales"]))
        if not finite(z):
            return result(abstained=True, reason="nonfinite_model_score")
        probability = 1./(1.+exp(-max(-40., min(40., z))))
        approved = probability >= params["threshold"]
        return result(approved=approved, probability=probability,
                      reason="frozen_logistic_threshold_approved" if approved else "frozen_logistic_threshold_rejected")


@dataclass(frozen=True, init=False)
class QualifiedGateBundle:
    manifest: object
    gates: object
    sha256: str

    def __init__(self, manifest, gates, *, _token=None):
        if _token is not _LOADED:
            raise ValueError("load a verified immutable qualification artifact; manual scores are not accepted")
        object.__setattr__(self, "manifest", frozen(manifest))
        object.__setattr__(self, "gates", MappingProxyType(dict(gates)))
        object.__setattr__(self, "sha256", manifest["qualification_sha256"])

    @classmethod
    def load(cls, path):
        root, manifest = Path(path).resolve().parent, _read_json(path)
        if manifest.get("schema") != "escalation_gate_qualification/1" or manifest.get("status") != "qualified_for_policy_execution":
            raise ValueError("this policy has not completed qualification")
        if manifest.get("qualification_sha256") != digest({k:v for k,v in manifest.items() if k != "qualification_sha256"}):
            raise ValueError("qualification manifest changed")
        if manifest["runtime_identity"] != current_identity() or manifest["feature_schema"] != SCHEMA:
            raise ValueError("runtime/model/feature identities differ from the frozen qualification")
        if set(manifest["gates"]) != {"feasibility", "benefit"} or set(manifest["evidence"]) != set(CHECKS):
            raise ValueError("both gates and every required qualification check are mandatory")
        bindings = dict(runtime_identity=manifest["runtime_identity"], feature_schema=manifest["feature_schema"],
            scheduler_config=manifest["scheduler_config"], formation_profile=manifest["formation_profile"],
            gate_sha256={k:v["sha256"] for k,v in manifest["gates"].items()})
        target = digest(bindings)
        for check, origin in CHECKS.items():
            evidence = _referenced(root, manifest["evidence"][check])
            if (evidence.get("schema") != "escalation_qualification_evidence/1" or evidence.get("check") != check
                    or evidence.get("status") != "passed" or evidence.get("origin") != origin
                    or evidence.get("qualification_target_sha256") != target):
                raise ValueError("missing or wrong-scope qualification evidence")
            prior = _referenced(root, evidence["preregistered_criteria"])
            if prior.get("schema") != "registered_qualification_criteria/1" or prior["check"] != check:
                raise ValueError("qualification numerical criteria lack a matching registration")
            criteria = prior["criteria"]
            if not criteria or criteria != evidence["criteria"]:
                raise ValueError("qualification criteria changed after registration")
            for criterion in criteria:
                if set(criterion) != {"metric", "operator", "limit"} or criterion["operator"] not in OPERATORS:
                    raise ValueError("explicit numerical qualification criterion required")
                observed = evidence["observed"].get(criterion["metric"])
                if not finite(observed) or not finite(criterion["limit"]) or not OPERATORS[criterion["operator"]](observed, criterion["limit"]):
                    raise ValueError("a recorded numerical qualification requirement fails")
            if not evidence["supporting_artifacts"]:
                raise ValueError("a passed assertion without preserved evidence is insufficient")
            for artifact in evidence["supporting_artifacts"]:
                _referenced(root, artifact)
        gates = {}
        for kind, reference in manifest["gates"].items():
            raw = _referenced(root, reference)
            gate = FrozenGate(raw, reference["sha256"])
            if gate.kind != kind:
                raise ValueError("gate artifact occupies the wrong role")
            if gate.method == "logistic":
                provenance = _referenced(root, raw["training_provenance"])
                required = {"training", "calibration", "prospective"}
                if (provenance.get("schema") != "offline_logistic_fit_provenance/1"
                        or provenance.get("origin") != "offline_matched_physical_pairs"
                        or provenance["feature_schema"] != SCHEMA
                        or provenance["parameters_sha256"] != digest(raw["parameters"])
                        or set(provenance["seed_partitions"]) != required):
                    raise ValueError("logistic fit lacks exact offline provenance")
                partitions = provenance["seed_partitions"]
                combined = []
                for name in required:
                    if not partitions[name] or any(type(seed) is not int for seed in partitions[name]):
                        raise ValueError("entire seed/episode partitions required")
                    combined.extend(partitions[name])
                if len(combined) != len(set(combined)):
                    raise ValueError("seed leakage across model fit/calibration/prospective partitions")
                _referenced(root, provenance["dataset_manifest"])
                _referenced(root, provenance["exposure_audit"])
            gates[kind] = gate
        return cls(manifest, gates, _token=_LOADED)

    def validate_binding(self, proposal, assessment, packet):
        binding = {k:packet[k] for k in ("schema", "proposal_sha256", "evidence_sha256", "assessment_evidence_sha256",
                                        "now_s", "history_s", "values")}
        if packet.get("row_sha256") != digest(binding) or packet["schema"] != SCHEMA:
            raise ValueError("feature row was altered or has an unknown acquired schema")
        if (packet["proposal_sha256"] != proposal.fingerprint or packet["now_s"] != proposal.start_s
                or assessment.now_s != proposal.start_s or packet["assessment_evidence_sha256"] != assessment.evidence_fingerprint
                or assessment.proposal_fingerprint != proposal.fingerprint):
            raise ValueError("gates must bind the same proposal and exact current report evidence")
        if digest(asdict(proposal.profile)) != digest(dict(self.manifest["formation_profile"])):
            raise ValueError("proposal treatment differs from qualified frozen profile")

    def decisions(self, proposal, assessment, packet):
        self.validate_binding(proposal, assessment, packet)
        return tuple(self.gates[kind].decide(packet, assessment, self.sha256) for kind in ("feasibility", "benefit"))


class EscalationScheduler:
    """Repeated opportunities with one owner; absent qualification means abstain."""
    def __init__(self, config, *, profile=None, qualification=None):
        if not isinstance(config, SchedulerConfig):
            raise TypeError("explicit SchedulerConfig required")
        self.config, self.profile, self.qualification = config, profile or FormationProfile(), qualification
        if qualification is not None:
            if not isinstance(qualification, QualifiedGateBundle):
                raise TypeError("qualified artifact bundle required, not a manual gate callback or score")
            if digest(asdict(config)) != digest(dict(qualification.manifest["scheduler_config"])):
                raise ValueError("scheduler differs from prospective freeze")
            if digest(asdict(self.profile)) != digest(dict(qualification.manifest["formation_profile"])):
                raise ValueError("treatment differs from qualification")
        self._active, self._time, self._cooldown_until = None, None, -1.
        self._attempts, self._events = 0, []

    @classmethod
    def from_manifest(cls, config, *, qualification_path=None, profile=None):
        if qualification_path is None or not Path(qualification_path).is_file():
            return cls(config, profile=profile)
        return cls(config, profile=profile, qualification=QualifiedGateBundle.load(qualification_path))

    @property
    def events(self):
        return tuple(frozen(row) for row in self._events)

    @property
    def attempts(self):
        return self._attempts

    def _event(self, now, reason, **details):
        self._events.append(dict(time_s=now, reason=reason, attempts=self._attempts, **details))

    def _active_step(self, context):
        controller = self._active
        members = frozenset(controller.proposal.member_ids)
        # Numeric/native references are current local execution feedback, not
        # an inference that a cached report means the CAV can still execute.
        # The formation layer enforces complete cohort availability too.
        result = controller.step(context.reports, now_s=context.now_s,
            damping_commands={v:a for v,a in context.damping_commands.items() if v in members},
            damping_native_ids=context.damping_native_ids & members)
        if result.phase in {"release", "abort"}:
            self._cooldown_until = context.now_s+self.config.cooldown_s
            self._active = None
            self._event(context.now_s, result.reason, event="wall_terminal", phase=result.phase,
                        proposal_sha256=result.proposal_fingerprint, cooldown_until_s=self._cooldown_until)
        p = controller.proposal.profile
        return PolicyOutput(result.commands, result.phase, result.reason, result.member_ids,
            result.proposal_fingerprint, result.evidence_fingerprint,
            dict(qualification_sha256=self.qualification.sha256, readiness_count=result.readiness_count,
                span_m=result.span_m, speed_spread_mps=result.speed_spread_mps, target_speed_mps=result.target_speed_mps,
                compact=result.compact, assembly_completed_at_s=result.assembly_completed_at_s,
                readiness_samples_required=p.readiness_samples, formed_span_m=p.formed_span_m,
                formed_speed_spread_mps=p.formed_speed_spread_mps, target_speed_tolerance_mps=p.target_speed_tolerance_mps))

    def __call__(self, context):
        if not isinstance(context, PolicyInput):
            raise TypeError("only the typed CAV-report runtime boundary is accepted")
        now = context.now_s
        if not finite(now) or now < 0 or now % 1. or self._time is not None and now != self._time+1.:
            raise ValueError("scheduler must advance once on each 1 s control tick")
        self._time = now
        opportunity = now in self.config.opportunity_times_s
        if self._active is not None:
            if opportunity:
                self._event(now, "wall_active", event="opportunity", approval=False)
            return self._active_step(context)
        if not opportunity:
            return PolicyOutput(reason="no_registered_opportunity")
        if now < self._cooldown_until:
            self._event(now, "cooldown", event="opportunity", approval=False)
            return PolicyOutput(reason="cooldown")
        if self._attempts >= self.config.maximum_attempts:
            self._event(now, "action_budget_exhausted", event="opportunity", approval=False)
            return PolicyOutput(reason="action_budget_exhausted")
        # Reconstruct current reports from actual receipt history; a caller cannot
        # supply a different latest-report set to feasibility and controller.
        latest = latest_received(context.history, now_s=now, maximum_age_s=self.profile.max_report_age_s)
        if tuple(r for _,r in latest) != context.reports:
            raise ValueError("latest reports disagree with their actual receipt history")
        proposal, assessment, metadata = select_proposal(context.history, now_s=now,
            required_lanes=self.config.required_lanes, recruitment_band_m=self.config.recruitment_band_m,
            profile=self.profile, maximum_members_per_lane=self.config.maximum_members_per_lane)
        if proposal is None:
            self._event(now, metadata["status"], event="opportunity", approval=False, selection=metadata)
            return PolicyOutput(reason=metadata["status"], details=dict(selection=metadata))
        packet = feature_packet(proposal, context.history, now_s=now, history_s=self.config.history_s)
        common = dict(proposal_sha256=proposal.fingerprint, evidence_sha256=packet["evidence_sha256"],
                      feature_row_sha256=packet["row_sha256"], selection=metadata)
        if not assessment.approved:
            reason = assessment.status+":"+assessment.reason
            self._event(now, reason, event="opportunity", approval=False, **common)
            return PolicyOutput(reason=reason, member_ids=proposal.member_ids, proposal_sha256=proposal.fingerprint,
                                evidence_sha256=packet["evidence_sha256"], details=dict(selection=metadata))
        if self.qualification is None:
            self._event(now, "qualification_artifacts_unavailable", event="opportunity", approval=False, **common)
            return PolicyOutput(reason="qualification_artifacts_unavailable", member_ids=proposal.member_ids,
                proposal_sha256=proposal.fingerprint, evidence_sha256=packet["evidence_sha256"],
                details=dict(selection=metadata, scientifically_qualified=False))
        approvals = self.qualification.decisions(proposal, assessment, packet)
        if not all(a.approved for a in approvals):
            reason = next(a.kind+":"+a.reason for a in approvals if not a.approved)
            self._event(now, reason, event="opportunity", approval=False, **common,
                        gates=[asdict(a) for a in approvals])
            return PolicyOutput(reason=reason, member_ids=proposal.member_ids,
                proposal_sha256=proposal.fingerprint, evidence_sha256=packet["evidence_sha256"],
                details=dict(gates=[asdict(a) for a in approvals], selection=metadata))
        authorization = authorize_qualified_gates(proposal, assessment, packet=packet, qualification=self.qualification)
        self._active = FormationController(proposal, authorization)
        self._attempts += 1
        self._event(now, "both_qualified_gates_approved", event="opportunity", approval=True,
                    **common, gates=[asdict(a) for a in approvals], qualification_sha256=self.qualification.sha256)
        return self._active_step(context)
