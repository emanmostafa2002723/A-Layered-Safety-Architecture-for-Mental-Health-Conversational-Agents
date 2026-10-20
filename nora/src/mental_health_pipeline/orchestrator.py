"""End-to-end, fail-closed orchestration for one complete chat turn."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import uuid
import re
from typing import Any, Mapping, Protocol

from .config import ConfigurationError, PipelineConfig, RuntimeMode
from .contact_display import apply_contact_display_policy, explicit_evidence
from .conversation_store import InMemoryConversationStore
from .generation import GenerationRequest, MistralGenerator
from .guardrails import ParallelInputGuardrails
from .output_guard import PlaceholderOutputGuard
from .rag import RAGService, RetrievalResult
from .resources import ResourceBundle, ResourceDirectory
from .response_policy import (
    OutputGuard,
    ResponsePolicyBuilder,
    ResponsePolicyPacket,
    render_fallback,
    render_response_policy,
    resolve_guarded_output,
)
from .response_policy.session_state import update_recent_safety_state
from .routing import resolve_route
from .schemas import Citation, PipelineRequest, PipelineResponse, SystemRoute
from .session_store import InMemorySafetyStateStore


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> Any: ...

    def readiness(self) -> Any: ...


_RESOURCE_MARKERS = {
    "worldwide_emergency_fallback",
    "local_resources_if_eligible",
}


@dataclass(frozen=True, slots=True)
class PipelineComponents:
    input_guardrails: ParallelInputGuardrails
    response_policy: ResponsePolicyBuilder
    resources: ResourceDirectory
    state_store: InMemorySafetyStateStore
    output_guard: OutputGuard
    rag: RAGService | None = None
    generator: Generator | None = None
    final_gate: Any | None = None


class FullPipeline:
    """Resolve safety once, then permit only route-compatible downstream work.

    No method exposes a stream. Generated text is buffered, combined with any
    application-owned resource block, and inspected before it can be returned.
    """

    version = "full-pipeline/1.0"

    def __init__(self, config: PipelineConfig, components: PipelineComponents) -> None:
        self.config = config
        if config.runtime_mode is RuntimeMode.PRODUCTION and (
            config.experimental_parallel_rag or config.experimental_incremental_output
        ):
            raise ConfigurationError('Parallel latency experiments are development-only')
        self.components = components
        self._turn_locks: dict[str, threading.Lock] = {}
        self._turn_locks_guard = threading.Lock()
        self.conversation_store = InMemoryConversationStore()
        from .parallel_execution import SpeculativeRetriever
        self._speculative = SpeculativeRetriever() if config.experimental_parallel_rag else None
        self._execution_local = threading.local()

    @property
    def execution_diagnostics(self):
        """Current caller's metadata only, never speculative or rejected text."""
        return dict(getattr(self._execution_local, 'diagnostics', {}))

    @property
    def production_ready(self) -> bool:
        return bool(self.readiness()["ready"])

    def process(self, request: PipelineRequest) -> PipelineResponse:
        request.validate()
        request_id = request.request_id or str(uuid.uuid4())
        lock = self._lock_for(request.session_id)
        with lock:
            self._execution_local.diagnostics = {}
            self._execution_local.ticket = None
            try:
                response = self._process_locked(request, request_id)
            finally:
                ticket = self._execution_local.ticket
                if ticket is not None and not ticket.cancel_event.is_set():
                    ticket.discard('turn_finished')
            # Delivery facts, not requested policy goals, determine whether
            # resources/a safety question were actually shown.
            def delivery_observation(state):
                from .response_policy.schemas import CrisisTier
                question_shown = bool(response.response and re.search(
                    r"[^.!?]*\b(?:safe|danger|acting|harm|hurt|suicide)\b[^.!?]*\?",
                    response.response, re.I))
                pending = state.direct_safety_question_pending or (
                    state.highest_recent_tier == CrisisTier.CRISIS and question_shown)
                return replace(state, direct_safety_question_pending=pending,
                    resources_last_shown_turn=state.last_updated_turn if response.resource_payload
                    and response.response else state.resources_last_shown_turn), None
            self.components.state_store.transaction(request.session_id, delivery_observation)
            # Save only displayed, guarded content, never a rejected draft.
            body = response.response
            if response.resource_payload and "\n\n" in body:
                body = body.split("\n\n", 1)[1]
            if body.strip():
                self.conversation_store.append(request.session_id, request.latest_user_message, body)
            return response

    def startup(self) -> None:
        """Eagerly validate worker startup before accepting API traffic."""

        for component in (
            self.components.input_guardrails.crisis_guardrail,
            self.components.input_guardrails.boundary_guardrail,
            self.components.final_gate,
        ):
            worker = getattr(component, "worker", None)
            start = getattr(worker, "start", None)
            if callable(start):
                start()

    def close(self) -> None:
        """Release persistent worker processes during graceful shutdown."""

        if self._speculative is not None:
            self._speculative.close()

        for component in (
            self.components.input_guardrails.crisis_guardrail,
            self.components.input_guardrails.boundary_guardrail,
            self.components.final_gate,
        ):
            worker = getattr(component, "worker", None)
            close = getattr(worker, "close", None)
            if callable(close):
                close()

    def _process_locked(self, request: PipelineRequest, request_id: str) -> PipelineResponse:
        from .query_semantics import contextual_search_query
        search_query,context_used=contextual_search_query(request.latest_user_message,
            self.conversation_store.get(request.session_id))
        if context_used:
            self._execution_local.diagnostics['retrieval_context']={'status':'explicit_previous_user_subject',
                'classifiers_and_generator_receive_original':True}
        ticket = None
        if self._speculative is not None and request.scope.is_supported and self.config.enable_rag and self.components.rag is not None:
            ticket = self._speculative.start(self.components.rag, search_query)
            self._execution_local.ticket = ticket
            self._execution_local.diagnostics['speculative_rag'] = {
                'status':'started' if ticket is not None else 'capacity_busy_serial_fallback'}
        crisis, boundary = self.components.input_guardrails.assess(
            request.latest_user_message
        )
        combined = resolve_route(crisis, boundary, scope=request.scope)

        def advance_state(state):
            packet = self.components.response_policy.build(
                combined.to_policy_dict(), recent_state=state
            )
            packet = self._prepare_display_policy(packet, request)
            resource_needed = _requires_resources(packet)
            turn_index = max(request.turn_number, state.last_updated_turn + 1)
            updated = update_recent_safety_state(
                state,
                packet.state_update,
                turn_index=turn_index,
                resources_shown=False,
                message_classified_crisis=crisis.tier == 'crisis',
            )
            return updated, packet

        packet = self.components.state_store.transaction(
            request.session_id, advance_state
        )
        resource_bundle = self._resource_bundle(request, packet)

        # Exact requests to inspect remembered facts are database operations,
        # not creative recall. Never ask a model to invent missing records.
        # Classifiers/policy still run first, and full output approval is mandatory.
        from .personal_memory import exact_recall_request, factual_recall_fallback
        if (packet.system_route.value == 'normal_chat' and packet.generation_allowed
                and not self.conversation_store.get(request.session_id)
                and exact_recall_request(request.latest_user_message)):
            if ticket is not None:
                self._execution_local.diagnostics['speculative_rag'] = ticket.discard('exact_personal_recall')
            self._execution_local.diagnostics['personal_recall'] = {'source':'dated_user_records',
                'notes':len(request.personal_memory), 'generator_called':False, 'final_guard_required':True}
            return self._guard_delivery(PipelineResponse(request_id=request_id,
                response=factual_recall_fallback(request.personal_memory),
                route=packet.system_route.value, used_fallback=False,
                output_guard_status='personal_recall_candidate'), packet)

        if (
            not packet.generation_allowed
            or self.components.generator is None
            or not _component_operational(self.components.generator)
        ):
            if ticket is not None:
                self._execution_local.diagnostics['speculative_rag'] = ticket.discard('generation_not_allowed_or_available')
            return self._fallback_response(
                request_id, packet, resource_bundle, status="generation_not_available"
            )

        retrieval: RetrievalResult | None = None
        if self.config.enable_rag and self.components.rag is not None:
            try:
                if ticket is not None:
                    retrieval, info = self._speculative.resolve(ticket, self.components.rag,
                        search_query, packet.system_route.value,
                        timeout=self.config.qwen_timeout_seconds+30)
                    self._execution_local.diagnostics['speculative_rag'] = info
                else:
                    retrieval = self.components.rag.retrieve(
                        search_query, packet.system_route.value
                    )
            except Exception:
                # RAG is optional context, never a safety dependency. Continue
                # without it rather than changing or downgrading the route.
                retrieval = None
                if ticket is not None:
                    self._execution_local.diagnostics['speculative_rag'] = ticket.discard('retrieval_failed')

        trusted_policy = render_response_policy(packet,profile=self.config.generator_prompt_profile)
        locale = _locale_for(request)
        try:
            generated = self.components.generator.generate(GenerationRequest(
                user_text=request.latest_user_message,
                response_policy=trusted_policy,
                retrieval=retrieval,
                locale=locale,
                history=self.conversation_store.get(request.session_id),
                incremental_output=self.config.experimental_incremental_output,
                personal_memory=request.personal_memory if packet.system_route.value == 'normal_chat' else (),
            ))
            if getattr(generated,'evidence_snapshot_available',False):
                retrieval=generated.effective_retrieval
            candidate = _with_fixed_resources(generated.text, resource_bundle)
            decision = self._inspect_output(candidate, packet, request)
            decision = self._inspect_conversation_quality(candidate, packet, request, decision, resource_bundle)
            resolution = resolve_guarded_output(candidate, packet, decision)

            if resolution.retry_allowed:
                retry_policy = (
                    trusted_policy
                    + "\nThe previous complete candidate was rejected by the output-safety layer. "
                    "Rewrite it conservatively while following every policy constraint."
                )
                if "missing_immediate_safety_question" in decision.reason_codes:
                    retry_policy += "\nInclude one gentle direct question about whether they may act on these thoughts now or have already harmed themselves. Invite a trusted person to stay with them. Do not finish with only a referral."
                if "insufficient_supportive_content" in decision.reason_codes:
                    retry_policy += "\nAnswer the latest message directly. Acknowledge its specific content, offer one useful next step and ask one relevant question when appropriate. Do not return 'OK', 'Acknowledged' or only an apology."
                if 'unsupported_greeting_distress' in decision.reason_codes:
                    retry_policy += ('\nThis is a neutral greeting with no disclosed problem. '
                        'Reply briefly and naturally. Do not apologize for distress or invent a difficult situation.')
                if {"repeated_previous_response", "repeated_generic_apology", "near_copied_previous_response"}.intersection(decision.reason_codes):
                    retry_policy += (
                        "\nThe prior candidate copied an earlier reply or repeated its generic apology. "
                        "Write a fresh answer to the LAST user message. Treat previous assistant replies as "
                        "context, not templates. Do not begin with 'I'm sorry to hear that'. "
                        "Use the new detail the person just supplied. Do not ask a question they have already answered. "
                        "Do not return to an earlier topic when the person has given a new detail."
                    )
                if 'reasked_answered_reason' in decision.reason_codes:
                    retry_policy += ('\nThe latest message already answers your previous question about the reason '
                        'for their feelings. Build on that specific answer; do not ask why they feel that way again. '
                        'A new specific question is optional, not required.')
                if 'ignored_topic_correction' in decision.reason_codes:
                    retry_policy += ('\nThe person explicitly corrected the topic in the latest message. '
                        'Use the corrected topic rather than continuing advice about the rejected one. '
                        'Do not invent additional details.')
                if 'unsupported_personal_memory' in decision.reason_codes:
                    retry_policy += ('\nThe candidate invented a past conversation or prior advice not present in the supplied context. '
                        'Do not say I remember, I suggested, we discussed, or invent what the user previously said. '
                        'If dated user notes exist, acknowledge ONLY that user report and ask how it is now. '
                        'If no history or notes exist, honestly say you do not have those earlier details available here '
                        'and invite the user to share them. Do not pretend that missing information is remembered.')
                if 'missed_available_memory' in decision.reason_codes:
                    retry_policy += ('\nA dated summary of the USER report is supplied, even though no old transcript is available. '
                        'The user asks what THEY told you, not what YOU advised. Acknowledge only the supplied user report '
                        'and ask one brief question about whether it has changed. Do not deny the available note or invent past advice. '
                        'Do not produce a generic coping list instead of answering the recall question.')
                if 'ignored_listening_preference' in decision.reason_codes:
                    retry_policy += ('\nThe person previously preferred listening rather than advice, and has not asked '
                        'for practical advice now. Acknowledge this specific message and invite them to share more '
                        'with one gentle open question. Do not prescribe exercises, offer a numbered list or repeat a generic apology. '
                        'This preference never overrides required current safety guidance.')
                if "unsupported_suicide_assumption" in decision.reason_codes:
                    retry_policy += (
                        "\nThe person has not disclosed suicidal thoughts in the available conversation. "
                        "Do not assume such thoughts or ask about acting on them. Respond to the actual latest "
                        "message. An injury caused by another person is not a disclosure of self-inflicted harm; "
                        "acknowledge it and ask about present safety and medical support when relevant."
                    )
                if 'model_generated_contact_details' in decision.reason_codes:
                    retry_policy += (
                        '\nRemove every telephone number and crisis-service URL from your generated text. '
                        'The application alone supplies the authorized contact block when this turn requires it. '
                        'Do not add unrequested resource advice to a greeting or ordinary concern. '
                        'Answer the actual latest message with proportionate support; preserve any required safety guidance.'
                    )
                generated = self.components.generator.generate(GenerationRequest(
                    user_text=request.latest_user_message,
                    response_policy=retry_policy,
                    retrieval=retrieval,
                    locale=locale,
                    history=self.conversation_store.get(request.session_id),
                    incremental_output=self.config.experimental_incremental_output,
                    personal_memory=request.personal_memory if packet.system_route.value == 'normal_chat' else (),
                ))
                if getattr(generated,'evidence_snapshot_available',False):
                    retrieval=generated.effective_retrieval
                candidate = _with_fixed_resources(generated.text, resource_bundle)
                decision = self._inspect_output(candidate, packet, request)
                decision = self._inspect_conversation_quality(candidate, packet, request, decision, resource_bundle)
                resolution = resolve_guarded_output(
                    candidate, packet, decision, retry_count=1
                )
        except Exception as error:
            from .incremental_output import IncrementalOutputStopped
            if isinstance(error, IncrementalOutputStopped):
                self._execution_local.diagnostics['incremental_output'] = error.diagnostics
                return self._fallback_response(request_id, packet, resource_bundle,
                    status='incremental_output_veto')
            return self._fallback_response(
                request_id, packet, resource_bundle, status="generation_or_guard_failure"
            )

        if not resolution.approved_generated_response or resolution.display_text is None:
            fallback_text = resolution.display_text or render_fallback(packet).text
            memory_reasons = {'unsupported_personal_memory', 'missed_available_memory'}
            if (packet.system_route.value == 'normal_chat' and decision.reason_codes
                    and set(decision.reason_codes) <= memory_reasons):
                # Do not let a model invent history or deny a real saved report.
                # Only after the bounded rewrite fails, render exact dated notes;
                # this replacement still goes through the complete final guard.
                from .personal_memory import factual_recall_fallback
                fallback_text = factual_recall_fallback(request.personal_memory)
            return self._guard_delivery(PipelineResponse(
                request_id=request_id,
                response=_with_fixed_resources(fallback_text, resource_bundle),
                route=packet.system_route.value,
                resource_payload=resource_bundle.to_dict() if resource_bundle else None,
                used_fallback=True,
                output_guard_status="fallback:" + ",".join(resolution.reason_codes),
            ), packet, retrieval)

        citations = tuple(
            Citation(
                source_id=item.source_id,
                title=item.title,
                source=item.source,
                page=item.page,
                url=item.url,
            )
            for item in generated.citations
        )
        return self._guard_delivery(PipelineResponse(
            request_id=request_id,
            response=resolution.display_text,
            route=packet.system_route.value,
            citations=citations,
            resource_payload=resource_bundle.to_dict() if resource_bundle else None,
            used_fallback=False,
            output_guard_status="approved",
        ), packet, retrieval)

    def _fallback_response(
        self,
        request_id: str,
        packet: ResponsePolicyPacket,
        resources: ResourceBundle | None,
        *,
        status: str,
    ) -> PipelineResponse:
        fallback = render_fallback(packet)
        return self._guard_delivery(PipelineResponse(
            request_id=request_id,
            response=_with_fixed_resources(fallback.text, resources),
            route=packet.system_route.value,
            resource_payload=resources.to_dict() if resources else None,
            used_fallback=True,
            output_guard_status=status,
        ), packet)

    def _guard_delivery(self, response, packet, retrieval=None):
        gate = self.components.final_gate
        if gate is None:
            return response
        result = gate.deliver(response.response, packet, retrieval)
        if (self.config.output_guard_backend in {'local_hybrid','local_qwen'}
                and result.get('action')=='DISPLAY_CHECKED_REPLACEMENT'
                and result.get('replacement_checked') is True
                and isinstance(result.get('text'),str) and result['text'].strip()):
            return replace(response, response=result['text'], citations=(), resource_payload=None,
                used_fallback=True, output_guard_status=('hybrid_checked_medical_replacement'
                    if self.config.output_guard_backend=='local_hybrid' else 'qwen_checked_medical_replacement'))
        if result.get('action') == 'DISPLAY' and result.get('text') == response.response:
            status=('hybrid_output_display_approved' if self.config.output_guard_backend=='local_hybrid' else
                    'local_output_display_approved' if self.config.output_guard_backend=='local_qwen' else 'standalone_display_approved')
            return replace(response, output_guard_status=status)
        # ABSTAIN releases no candidate, fallback, contacts, or source details.
        return replace(response, response='', citations=(), resource_payload=None,
                       used_fallback=False, output_guard_status='output_guard_abstained')

    def _inspect_output(self, candidate, packet, request):
        contextual = getattr(self.components.output_guard, "inspect_with_context", None)
        if callable(contextual):
            return contextual(candidate, packet, user_text=request.latest_user_message)
        return self.components.output_guard.inspect(candidate, packet)

    def _prepare_display_policy(self, packet, request):
        prepared = apply_contact_display_policy(
            packet, request.latest_user_message, self.config.guardrail_repo,
            history=self.conversation_store.get(request.session_id),
        )
        matches = explicit_evidence(request.latest_user_message, self.config.guardrail_repo)
        contextual = bool(matches)
        boundary_safety = prepared.generator_guidance_id in {'final_boundary_child_v1', 'urgent_medical_response_v1'}
        trusted_urgent = prepared.severity.value == 'urgent' and prepared.temporal_status.value in {'current', 'imminent', 'mixed'}
        # Preserve all model labels/routes for audit, but do not record an
        # unconfirmed category-only signal as a confirmed recent danger event.
        if not contextual and not boundary_safety and not prepared.rule_kinds and not trusted_urgent:
            from .response_policy.schemas import CrisisTier, Severity, TemporalStatus
            update = replace(prepared.state_update, categories=(), risk_signal_turn=False,
                highest_turn_tier=CrisisTier.NORMAL, highest_turn_severity=Severity.NONE,
                temporal_status=TemporalStatus.UNSPECIFIED, direct_safety_question_requested=False)
            return replace(prepared, state_update=update, policy_flags=tuple(dict.fromkeys(
                (*prepared.policy_flags, 'uncertain_category_not_confirmed_recent_danger'))))
        # A real contextual concern can be retained before delivery, but policy
        # intent alone must never claim that its question was displayed.
        update = replace(prepared.state_update, direct_safety_question_requested=False)
        from .output_guard import current_first_person_safety_disclosure
        if (any(m.temporal.value in {'current', 'imminent'} for m in matches)
                or current_first_person_safety_disclosure(request.latest_user_message)):
            from .response_policy.schemas import TemporalStatus
            update = replace(update, temporal_status=TemporalStatus.CURRENT)
        if boundary_safety:
            from .response_policy.schemas import CrisisTier
            update = replace(update, risk_signal_turn=True, highest_turn_tier=CrisisTier.CRISIS)
        return replace(prepared, state_update=update)

    def _inspect_conversation_quality(self, candidate, packet, request, decision, resources):
        """Request one buffered rewrite for a copied prior conversational reply."""
        from .response_policy.output_contract import OutputGuardAction, OutputGuardDecision
        if decision.action != OutputGuardAction.APPROVE:
            return decision
        history = self.conversation_store.get(request.session_id)
        previous = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        # Match only conversational content, excluding the application-owned contacts.
        body = candidate
        if resources is not None:
            prefix = resources.render_plain_text().strip() + "\n\n"
            if body.startswith(prefix):
                body = body[len(prefix):]
        normalize = lambda text: " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
        current, old = normalize(body), normalize(previous)
        from .output_contacts import has_generated_contact_details
        reasons = ['model_generated_contact_details'] if has_generated_contact_details(body) else []
        from .conversation_quality import unsupported_personal_memory, missed_available_memory, ignored_listening_preference
        if unsupported_personal_memory(request.latest_user_message, body, history, request.personal_memory, packet):
            reasons.append('unsupported_personal_memory')
        if missed_available_memory(request.latest_user_message, body, history, request.personal_memory, packet):
            reasons.append('missed_available_memory')
        if ignored_listening_preference(request.latest_user_message,body,request.personal_memory,packet):
            reasons.append('ignored_listening_preference')
        # An explicit request to repeat is legitimate. Safety enforcement remains
        # the existing output guard's responsibility and runs before this check.
        if not reasons and re.search(r"\b(?:repeat|say (?:that|it) again|copy (?:that|it)|same (?:words|answer))\b", request.latest_user_message, re.I):
            return decision
        user_context = " ".join([request.latest_user_message, *[
            m["content"] for m in history[-8:] if m["role"] == "user"
        ]])
        suicide_context = re.search(r"\b(?:suicid\w*|kill(?:ing)? myself|end(?:ing)? my life)\b", user_context, re.I)
        assumed_thoughts = re.search(
            r"\b(?:act(?:ing)? on (?:these|those|your) thoughts|your suicidal thoughts)\b",
            body, re.I,
        )
        if assumed_thoughts and not suicide_context:
            reasons.append("unsupported_suicide_assumption")
        from .conversation_quality import unsupported_greeting_distress,reasked_answered_reason,ignored_topic_correction,near_copied_reply
        if unsupported_greeting_distress(request.latest_user_message, body, history, packet):
            reasons.append('unsupported_greeting_distress')
        if reasked_answered_reason(request.latest_user_message,body,history,packet):
            reasons.append('reasked_answered_reason')
        if ignored_topic_correction(request.latest_user_message,body,packet):
            reasons.append('ignored_topic_correction')
        if old and len(current.split()) >= 8 and current == old:
            reasons.append("repeated_previous_response")
        elif near_copied_reply(body,previous,packet):
            reasons.append('near_copied_previous_response')
        apology = re.compile(r"^i(?: m| am) sorry to hear that\b")
        if old and apology.search(current) and apology.search(old):
            reasons.append("repeated_generic_apology")
        if not reasons:
            return decision
        return OutputGuardDecision(action=OutputGuardAction.RETRY,
            guard_version=decision.guard_version, policy_version=decision.policy_version,
            output_guard_profile=decision.output_guard_profile,
            reason_codes=tuple(reasons), evaluated_complete_response=True)

    def _resource_bundle(
        self, request: PipelineRequest, packet: ResponsePolicyPacket
    ) -> ResourceBundle | None:
        if not _requires_resources(packet):
            return None
        country = (
            request.scope.normalized_country()
            if request.scope.location_confirmed
            else None
        )
        bundle = self.components.resources.resolve(country)
        # A medical emergency or abuse disclosure does not by itself justify
        # showing a suicide-specific number. Retain emergency help and the
        # worldwide directory; do not invent a specialized contact.
        nonsuicide = "contact_display:other_emergency" in packet.policy_flags or (
            bool(packet.crisis_labels)
            and not {"suicidal_ideation", "self_directed_harm"}.intersection(packet.crisis_labels)
        )
        if nonsuicide:
            bundle = replace(bundle, resources=tuple(
                resource for resource in bundle.resources
                if resource.identifier not in {"ca.988", "us.988"}
            ))
        return bundle

    def readiness(self) -> dict[str, Any]:
        issues = list(self.config.readiness_issues())
        for name, component in (
            ("crisis_guardrail", self.components.input_guardrails.crisis_guardrail),
            ("boundary_guardrail", self.components.input_guardrails.boundary_guardrail),
            ("generator", self.components.generator),
            ("rag", self.components.rag if self.config.enable_rag else None),
        ):
            if component is None:
                if name == "generator" and self.config.enable_generation:
                    issues.append("generator_not_configured")
                continue
            probe = getattr(component, "readiness", None)
            if callable(probe):
                try:
                    result = probe()
                    ready = _readiness_bool(result)
                except Exception:
                    ready = False
                if not ready:
                    issues.append(f"{name}_not_ready")

        if self.config.enable_rag and self.components.rag is None:
            issues.append("rag_not_configured")
        warnings: list[str] = []
        if not bool(getattr(self.components.output_guard, "production_ready", False)):
            target = issues if self.config.runtime_mode is RuntimeMode.PRODUCTION else warnings
            target.append("output_guard_not_production_ready")
        resource_issues = self.components.resources.readiness_issues()
        if self.config.runtime_mode is RuntimeMode.PRODUCTION:
            issues.extend(resource_issues)
        else:
            warnings.extend(resource_issues)
        if self.config.runtime_mode is RuntimeMode.PRODUCTION and not bool(
            getattr(self.components.state_store, "production_ready", False)
        ):
            issues.append("production_shared_state_store_required")
        unique = tuple(dict.fromkeys(issues))
        return {
            "ready": not unique,
            "issues": list(unique),
            "warnings": list(dict.fromkeys(warnings)),
            "version": self.version,
        }

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._turn_locks_guard:
            return self._turn_locks.setdefault(session_id, threading.Lock())


def _requires_resources(packet: ResponsePolicyPacket) -> bool:
    return (
        packet.system_route.value
        == SystemRoute.URGENT_MEDICAL.value
        or bool(_RESOURCE_MARKERS.intersection(packet.required_visible_elements))
    )


def _with_fixed_resources(text: str, resources: ResourceBundle | None) -> str:
    body = text.strip()
    if resources is None:
        return body
    return resources.render_plain_text().strip() + "\n\n" + body


def _locale_for(request: PipelineRequest) -> str:
    country = request.scope.normalized_country()
    if request.scope.location_confirmed and country:
        return f"en-{country}"
    return "en"


def _readiness_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return bool(value.get("ready", False))
    ready = getattr(value, "ready", None)
    return bool(ready) if isinstance(ready, bool) else False


def _component_operational(component: Any) -> bool:
    probe = getattr(component, "readiness", None)
    if not callable(probe):
        return True
    try:
        return _readiness_bool(probe())
    except Exception:
        return False


__all__ = ["FullPipeline", "PipelineComponents"]
