from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, literal, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.camera import Camera
from app.models.incident import DetectionType, Incident
from app.models.video_rag import VideoRagChunk, VideoRagIndex, VideoRagIndexStatus
from app.schemas.video_rag import (
    VideoRagEvidence,
    VideoRagIndexFreshness,
    VideoRagQueryRequest,
    VideoRagQueryResponse,
    VideoRagStatusResponse,
)
from app.services.ollama import OllamaClient
from app.services.ollama import OllamaUnavailableError

_CITATION_PATTERN = re.compile(r"\[incident:([0-9a-fA-F-]{36})\]")
_FILTER_HINT_PATTERN = re.compile(
    r"\b(?:camera|gate|entrance|exit|lobby|parking|today|tonight|yesterday|"
    r"night|morning|afternoon|evening|week|month|hour|between|before|after|"
    r"since|from|until|ago)\b",
    re.IGNORECASE,
)
_LEXICAL_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "at",
    "did",
    "detect",
    "detected",
    "do",
    "happen",
    "happened",
    "in",
    "is",
    "on",
    "spotted",
    "the",
    "there",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
}
_IDENTITY_QUERY_PATTERN = re.compile(
    r"\b(?:identity|person|recognized|spotted|seen|detect(?:ed)?|who|when\s+was)\b",
    re.IGNORECASE,
)
_CO_OCCURRENCE_PATTERN = re.compile(
    r"\b(?:together|alongside|same\s+(?:time|frame|camera|scene))\b",
    re.IGNORECASE,
)
_IDENTITY_WEAPON_PATTERN = re.compile(
    r"(?:\b(?:with|and|carrying|holding)\b.*\b(?:weapon|gun|knife)\b|"
    r"\b(?:weapon|gun|knife)\b.*\b(?:with|and)\b)",
    re.IGNORECASE,
)
_IDENTITY_COOCCURRENCE_WINDOW = timedelta(seconds=15)
_DISPLAY_TIMEZONE = timezone(timedelta(hours=5), name="PKT")
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_COUNT_TOKEN = r"(?:[1-9]|1\d|20|one|two|three|four|five|six|seven|eight|nine|ten)"
_RESULT_LIMIT_PATTERNS = (
    re.compile(
        rf"\b(?:only\s+)?(?P<count>{_COUNT_TOKEN})\s+"
        r"(?:latest|newest|oldest|earliest|most\s+recent)\s+"
        r"(?:records?|incidents?|results?|events?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:latest|newest|oldest|earliest|top|first|last)\s+"
        rf"(?P<count>{_COUNT_TOKEN})(?!\s+(?:minutes?|hours?|days?|weeks?))"
        r"(?:\s+(?:records?|incidents?|results?|events?))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:list|show|return|give|display)\s+(?:only\s+)?"
        rf"(?P<count>{_COUNT_TOKEN})\s+(?:records?|incidents?|results?|events?)\b",
        re.IGNORECASE,
    ),
)
_RELATIVE_TIME_PATTERN = re.compile(
    rf"\b(?:last|past|previous|within)\s+(?P<count>{_COUNT_TOKEN})\s+"
    r"(?P<unit>minutes?|hours?|days?|weeks?)\b",
    re.IGNORECASE,
)
_INCIDENT_SUMMARY_PATTERN = re.compile(
    r"\b(?:incident|incidents|event|events|happen(?:ed)?|detect(?:ed|ion)?|"
    r"show|list|recent|latest)\b",
    re.IGNORECASE,
)
_STRUCTURED_GENERIC_TERMS = {
    "all",
    "any",
    "afternoon",
    "data",
    "day",
    "evening",
    "event",
    "events",
    "give",
    "hour",
    "hours",
    "incident",
    "incidents",
    "information",
    "know",
    "last",
    "latest",
    "list",
    "me",
    "model",
    "month",
    "morning",
    "night",
    "of",
    "past",
    "person",
    "persons",
    "previous",
    "recent",
    "show",
    "tell",
    "today",
    "tonight",
    "week",
    "within",
    "yesterday",
}
_RESERVED_CAMERA_QUERY_TERMS = {
    "fire",
    "person",
    "smoke",
    "weapon",
}


@dataclass(slots=True)
class RetrievedChunk:
    chunk: VideoRagChunk
    incident: Incident
    camera: Camera
    score: float


class VideoRagQueryService:
    def __init__(self, session: AsyncSession, ollama: OllamaClient | None = None) -> None:
        self.session = session
        self.ollama = ollama or OllamaClient(timeout=settings.video_rag_query_timeout_seconds)

    async def query(self, payload: VideoRagQueryRequest) -> VideoRagQueryResponse:
        payload = payload.model_copy(
            update={
                "limit": self._requested_result_limit(payload.question, payload.limit),
            }
        )
        camera_ids, start_at, end_at = await self._resolve_filters(payload)
        identity_response = await self._query_authoritative_identities(
            payload,
            camera_ids=camera_ids,
            start_at=start_at,
            end_at=end_at,
        )
        if identity_response is not None:
            return identity_response

        # Detector records are authoritative and immediately available. They
        # must remain searchable while slower clip/frame enrichment is queued.
        incident_response = await self._query_authoritative_incidents(
            payload,
            camera_ids=camera_ids,
            start_at=start_at,
            end_at=end_at,
        )
        if incident_response is not None:
            return incident_response

        warnings: list[str] = []
        embedding_error: OllamaUnavailableError | None = None
        try:
            query_embedding = (await self.ollama.embed(payload.question))[0]
        except OllamaUnavailableError as exc:
            query_embedding = None
            embedding_error = exc
            warnings.append(
                "Semantic retrieval is temporarily busy; results use indexed text metadata."
            )
        latest_requested = bool(
            re.search(r"\b(?:latest|newest|most\s+recent)\b", payload.question, re.IGNORECASE)
        )
        retrieval_limit = 20 if latest_requested else payload.limit
        matches = await self._retrieve(
            payload.question,
            query_embedding,
            camera_ids=camera_ids,
            start_at=start_at,
            end_at=end_at,
            limit=retrieval_limit,
        )
        if latest_requested:
            matches = sorted(
                matches,
                key=lambda item: item.incident.occurred_at,
                reverse=True,
            )[: payload.limit]
        freshness = await self.freshness()
        if freshness.queued or freshness.processing or freshness.failed:
            warnings.append("Some retained incidents have not been indexed successfully yet.")
        if not matches:
            if embedding_error is not None:
                raise embedding_error
            return VideoRagQueryResponse(
                answer=self._not_found_answer(payload.question),
                evidence=[],
                warnings=warnings,
                freshness=freshness,
            )

        # A few compact contexts are enough to synthesize the answer. The full
        # ranked match list is still returned as evidence, while keeping local
        # CPU inference responsive.
        contexts = [self._context(item) for item in matches[:3]]
        if embedding_error is not None:
            answer = self._fallback_answer(matches[0])
            warnings.append(
                "Answer generation was busy, so this response summarizes authoritative incident metadata."
            )
        else:
            try:
                generated = await self.ollama.answer(payload.question, contexts)
                answer, grounding_warning = self._ground_answer(
                    generated, matches, question=payload.question
                )
                if grounding_warning:
                    warnings.append(grounding_warning)
            except OllamaUnavailableError:
                answer = self._fallback_answer(matches[0])
                warnings.append(
                    "Answer generation was busy, so this response summarizes authoritative incident metadata."
                )
        evidence = [self._evidence(item) for item in matches]
        return VideoRagQueryResponse(
            answer=answer, evidence=evidence, warnings=warnings, freshness=freshness
        )

    async def status(self) -> VideoRagStatusResponse:
        freshness = await self.freshness()
        return VideoRagStatusResponse(enabled=settings.video_rag_enabled, **freshness.model_dump())

    async def freshness(self) -> VideoRagIndexFreshness:
        rows = (
            await self.session.execute(
                select(VideoRagIndex.status, func.count(VideoRagIndex.incident_id)).group_by(
                    VideoRagIndex.status
                )
            )
        ).all()
        counts = {status.value: int(count) for status, count in rows}
        latest = await self.session.scalar(select(func.max(VideoRagIndex.indexed_at)))
        return VideoRagIndexFreshness(
            latest_indexed_at=latest,
            ready=counts.get("ready", 0),
            queued=counts.get("queued", 0),
            processing=counts.get("processing", 0),
            failed=counts.get("failed", 0),
        )

    async def _query_authoritative_identities(
        self,
        payload: VideoRagQueryRequest,
        *,
        camera_ids: list[UUID],
        start_at: datetime | None,
        end_at: datetime | None,
    ) -> VideoRagQueryResponse | None:
        identity_weapon_query = bool(_IDENTITY_WEAPON_PATTERN.search(payload.question))
        if not _IDENTITY_QUERY_PATTERN.search(payload.question) and not identity_weapon_query:
            return None

        query = (
            select(Incident, Camera)
            .join(Camera, Camera.id == Incident.camera_id)
            .where(
                Incident.archived_at.is_(None),
                Incident.recognized_identity.is_not(None),
                or_(Incident.snapshot_path.is_not(None), Incident.clip_path.is_not(None)),
            )
            .order_by(Incident.occurred_at.desc(), Incident.id)
        )
        if camera_ids:
            query = query.where(Incident.camera_id.in_(camera_ids))
        if start_at:
            query = query.where(Incident.occurred_at >= start_at)
        if end_at:
            query = query.where(Incident.occurred_at <= end_at)
        rows = (await self.session.execute(query)).all()
        if not rows:
            return None

        question = payload.question.casefold()
        question_terms = set(re.findall(r"[\w'-]+", question))
        requested: dict[str, tuple[str, int]] = {}
        for incident, _ in rows:
            label = str(
                (incident.recognized_identity or {}).get("identity_label") or ""
            ).strip()
            normalized = label.casefold()
            if not normalized:
                continue
            label_terms = [term for term in re.findall(r"[\w'-]+", normalized) if len(term) >= 2]
            positions = [question.find(term) for term in label_terms if term in question_terms]
            if normalized in question or positions:
                requested.setdefault(
                    normalized,
                    (label, question.find(normalized) if normalized in question else min(positions)),
                )
        if not requested:
            return None

        ordered_identities = sorted(requested, key=lambda item: requested[item][1])
        requested_rows = [
            (incident, camera)
            for incident, camera in rows
            if str(
                (incident.recognized_identity or {}).get("identity_label") or ""
            ).strip().casefold()
            in requested
        ]
        freshness = await self.freshness()
        warnings = ["This answer uses authoritative recognition metadata, not visual inference."]
        if freshness.queued or freshness.processing or freshness.failed:
            warnings.append("Visual indexing is still in progress for some retained incidents.")

        if identity_weapon_query:
            identity = ordered_identities[0]
            identity_name = requested[identity][0]
            identity_rows = [
                (incident, camera)
                for incident, camera in requested_rows
                if str(
                    (incident.recognized_identity or {}).get("identity_label") or ""
                ).strip().casefold()
                == identity
            ]
            weapon_query = (
                select(Incident, Camera)
                .join(Camera, Camera.id == Incident.camera_id)
                .where(
                    Incident.archived_at.is_(None),
                    Incident.detection_type == DetectionType.weapon,
                    or_(Incident.snapshot_path.is_not(None), Incident.clip_path.is_not(None)),
                )
                .order_by(Incident.occurred_at.desc(), Incident.id)
            )
            if camera_ids:
                weapon_query = weapon_query.where(Incident.camera_id.in_(camera_ids))
            if start_at:
                weapon_query = weapon_query.where(Incident.occurred_at >= start_at)
            if end_at:
                weapon_query = weapon_query.where(Incident.occurred_at <= end_at)
            weapon_rows = (await self.session.execute(weapon_query)).all()
            pairs = [
                (identity_incident, identity_camera, weapon_incident)
                for identity_incident, identity_camera in identity_rows
                for weapon_incident, weapon_camera in weapon_rows
                if identity_camera.id == weapon_camera.id
                and abs(weapon_incident.occurred_at - identity_incident.occurred_at)
                <= _IDENTITY_COOCCURRENCE_WINDOW
            ]
            pairs.sort(key=lambda item: max(item[0].occurred_at, item[2].occurred_at), reverse=True)
            selected_pairs = pairs[: payload.limit]
            if not selected_pairs:
                display_name = (
                    identity_name.title()
                    if identity_name == identity_name.casefold()
                    else identity_name
                )
                return VideoRagQueryResponse(
                    answer=f"No incident involving {display_name} and a weapon was recorded.",
                    evidence=[],
                    warnings=warnings,
                    freshness=freshness,
                )

            summaries = []
            evidence = []
            for identity_incident, camera, weapon_incident in selected_pairs:
                display_name = (
                    identity_name.title()
                    if identity_name == identity_name.casefold()
                    else identity_name
                )
                readable_date, readable_time = self._readable_local_datetime(
                    identity_incident.occurred_at
                )
                summaries.append(
                    f"Yes, an incident involving {display_name} and a weapon was recorded "
                    f"in {camera.name} on {readable_date} at {readable_time} with "
                    f"{identity_incident.confidence:.0%} identity confidence and "
                    f"{weapon_incident.confidence:.0%} weapon confidence."
                )
                evidence.extend(
                    [
                        self._identity_evidence(identity_incident, camera),
                        VideoRagEvidence(
                            incident_id=weapon_incident.id,
                            camera_id=camera.id,
                            camera_name=camera.name,
                            occurred_at=weapon_incident.occurred_at,
                            detection_type=weapon_incident.detection_type,
                            confidence=weapon_incident.confidence,
                            matched_excerpt=self._authoritative_excerpt(
                                weapon_incident, camera
                            ),
                            relevance_score=1.0,
                            snapshot_url=(
                                f"/api/v1/incidents/{weapon_incident.id}/snapshot"
                                if weapon_incident.snapshot_path
                                else None
                            ),
                            clip_url=(
                                f"/api/v1/incidents/{weapon_incident.id}/clip"
                                if weapon_incident.clip_path
                                else None
                            ),
                        ),
                    ]
                )
            return VideoRagQueryResponse(
                answer=" ".join(summaries),
                evidence=evidence,
                warnings=warnings,
                freshness=freshness,
            )

        if len(ordered_identities) > 1 and _CO_OCCURRENCE_PATTERN.search(payload.question):
            candidates: dict[
                tuple[str, ...], dict[str, tuple[Incident, Camera]]
            ] = {}
            for anchor_incident, anchor_camera in requested_rows:
                matches: dict[str, tuple[Incident, Camera]] = {}
                for identity in ordered_identities:
                    options = [
                        (incident, camera)
                        for incident, camera in requested_rows
                        if camera.id == anchor_camera.id
                        and str(
                            (incident.recognized_identity or {}).get("identity_label") or ""
                        ).strip().casefold()
                        == identity
                        and abs(incident.occurred_at - anchor_incident.occurred_at)
                        <= _IDENTITY_COOCCURRENCE_WINDOW
                    ]
                    if not options:
                        break
                    matches[identity] = min(
                        options,
                        key=lambda item: abs(
                            item[0].occurred_at - anchor_incident.occurred_at
                        ),
                    )
                if len(matches) != len(ordered_identities):
                    continue
                times = [item[0].occurred_at for item in matches.values()]
                if max(times) - min(times) > _IDENTITY_COOCCURRENCE_WINDOW:
                    continue
                key = tuple(sorted(str(item[0].id) for item in matches.values()))
                candidates[key] = matches

            candidate_observations = list(candidates.values())
            candidate_observations.sort(
                key=lambda matches: max(item[0].occurred_at for item in matches.values()),
                reverse=True,
            )
            together: list[dict[str, tuple[Incident, Camera]]] = []
            used_incidents: set[UUID] = set()
            for matches in candidate_observations:
                incident_ids = {item[0].id for item in matches.values()}
                if incident_ids & used_incidents:
                    continue
                together.append(matches)
                used_incidents.update(incident_ids)
            if not together:
                names = " and ".join(requested[item][0] for item in ordered_identities)
                return VideoRagQueryResponse(
                    answer=(
                        f"No authoritative recognition records place {names} on the same "
                        "camera within 15 seconds."
                    ),
                    evidence=[],
                    warnings=warnings,
                    freshness=freshness,
                )

            selected_observations = together[: payload.limit]
            selected_rows = [
                matches[identity]
                for matches in selected_observations
                for identity in ordered_identities
            ]
            evidence = [self._identity_evidence(incident, camera) for incident, camera in selected_rows]
            names = " and ".join(requested[item][0] for item in ordered_identities)
            summaries: list[str] = []
            for matches in selected_observations[:3]:
                camera = matches[ordered_identities[0]][1]
                detections = ", ".join(
                    f"{requested[identity][0]} at {matches[identity][0].occurred_at.isoformat()}"
                    for identity in ordered_identities
                )
                summaries.append(f"{camera.name} ({detections})")
            if len(together) == 1:
                answer = f"{names} were recorded on the same camera within 15 seconds: {summaries[0]}."
            else:
                qualifier = "Most recent observations" if len(together) > 3 else "Observations"
                answer = (
                    f"{len(together)} same-camera observations placed {names} within 15 seconds. "
                    f"{qualifier}: {'; '.join(summaries)}."
                )
            warnings.append(
                "Together means recognition events from the same camera within 15 seconds; "
                "it does not confirm both identities in a single frame."
            )
            return VideoRagQueryResponse(
                answer=answer,
                evidence=evidence,
                warnings=warnings,
                freshness=freshness,
            )

        # Return the latest record for every requested identity so repeated
        # detections of one person cannot crowd the other names out.
        latest_by_identity: dict[str, tuple[Incident, Camera]] = {}
        for incident, camera in requested_rows:
            normalized = str(
                (incident.recognized_identity or {}).get("identity_label") or ""
            ).strip().casefold()
            latest_by_identity.setdefault(normalized, (incident, camera))
        selected_identities = [
            identity for identity in ordered_identities if identity in latest_by_identity
        ][: payload.limit]
        selected_rows = [latest_by_identity[identity] for identity in selected_identities]
        evidence = [self._identity_evidence(incident, camera) for incident, camera in selected_rows]
        summaries = [
            self._concise_incident_answer(
                incident,
                camera,
                identity_label=requested[identity][0],
            )
            for identity, (incident, camera) in zip(selected_identities, selected_rows, strict=True)
        ]
        return VideoRagQueryResponse(
            answer=" ".join(summaries),
            evidence=evidence,
            warnings=warnings,
            freshness=freshness,
        )

    @staticmethod
    def _identity_evidence(incident: Incident, camera: Camera) -> VideoRagEvidence:
        label = (incident.recognized_identity or {}).get("identity_label") or "recognized person"
        return VideoRagEvidence(
            incident_id=incident.id,
            camera_id=camera.id,
            camera_name=camera.name,
            occurred_at=incident.occurred_at,
            detection_type=incident.detection_type,
            confidence=incident.confidence,
            matched_excerpt=(
                f"Authoritative recognition metadata records {label} at {camera.name} "
                f"on {incident.occurred_at.isoformat()}."
            ),
            relevance_score=1.0,
            snapshot_url=f"/api/v1/incidents/{incident.id}/snapshot"
            if incident.snapshot_path
            else None,
            clip_url=f"/api/v1/incidents/{incident.id}/clip" if incident.clip_path else None,
        )

    async def _query_authoritative_incidents(
        self,
        payload: VideoRagQueryRequest,
        *,
        camera_ids: list[UUID],
        start_at: datetime | None,
        end_at: datetime | None,
    ) -> VideoRagQueryResponse | None:
        query = (
            select(Incident, Camera)
            .join(Camera, Camera.id == Incident.camera_id)
            .where(Incident.archived_at.is_(None))
        )
        if camera_ids:
            query = query.where(Incident.camera_id.in_(camera_ids))
        if start_at:
            query = query.where(Incident.occurred_at >= start_at)
        if end_at:
            query = query.where(Incident.occurred_at <= end_at)
        rows = (await self.session.execute(query.order_by(Incident.occurred_at.desc()))).all()
        if not rows:
            return None

        terms = self._structured_terms(payload.question)
        generic_summary = bool(_INCIDENT_SUMMARY_PATTERN.search(payload.question))
        ranked: list[tuple[int, int, Incident, Camera]] = []
        for incident, camera in rows:
            primary_score, secondary_score = self._authoritative_match_score(
                terms, incident, camera
            )
            if primary_score or secondary_score or (generic_summary and not terms):
                ranked.append((primary_score, secondary_score, incident, camera))
        if not ranked:
            return None

        # If detector/subtype/notes matched, do not let a coincidental camera
        # name (for example, a camera named "weapon") pollute those results.
        if terms and any(item[0] for item in ranked):
            ranked = [item for item in ranked if item[0]]
        ranked.sort(
            key=lambda item: (item[0], item[1], item[2].occurred_at),
            reverse=True,
        )
        selected = ranked[: payload.limit]
        evidence = [
            VideoRagEvidence(
                incident_id=incident.id,
                camera_id=camera.id,
                camera_name=camera.name,
                occurred_at=incident.occurred_at,
                detection_type=incident.detection_type,
                confidence=incident.confidence,
                matched_excerpt=self._authoritative_excerpt(incident, camera),
                relevance_score=1.0 if primary_score else 0.75,
                snapshot_url=f"/api/v1/incidents/{incident.id}/snapshot"
                if incident.snapshot_path
                else None,
                clip_url=f"/api/v1/incidents/{incident.id}/clip" if incident.clip_path else None,
            )
            for primary_score, _, incident, camera in selected
        ]
        broad_summary = generic_summary and not terms
        summaries = [
            self._concise_incident_answer(
                incident,
                camera,
                include_confirmation=not broad_summary,
            )
            for _, _, incident, camera in selected
        ]
        if broad_summary:
            period = " yesterday" if re.search(r"\byesterday\b", payload.question, re.I) else ""
            answer = f"These are a few incidents recorded{period}: {' '.join(summaries)}"
        else:
            answer = " ".join(summaries)
        freshness = await self.freshness()
        warnings = ["This answer uses authoritative detector metadata."]
        if freshness.queued or freshness.processing or freshness.failed:
            warnings.append(
                "Visual clip analysis is still processing; visual-detail questions may have incomplete results."
            )
        return VideoRagQueryResponse(
            answer=answer,
            evidence=evidence,
            warnings=warnings,
            freshness=freshness,
        )

    @classmethod
    def _concise_incident_answer(
        cls,
        incident: Incident,
        camera: Camera,
        *,
        identity_label: str | None = None,
        include_confirmation: bool = True,
    ) -> str:
        identity = identity_label or str(
            (incident.recognized_identity or {}).get("identity_label") or ""
        ).strip()
        if identity:
            display_identity = identity.title() if identity == identity.casefold() else identity
            observation = f"{display_identity} was spotted"
        else:
            event_name = incident.detection_type.value.replace("_", " ")
            subtype = cls._incident_subtype(incident)
            if subtype and subtype.casefold() != event_name.casefold():
                event_name = f"{event_name} ({subtype})"
            article = "an" if event_name[:1].casefold() in "aeiou" else "a"
            observation = f"{article} {event_name} was detected"

        readable_date, readable_time = cls._readable_local_datetime(incident.occurred_at)
        if not include_confirmation:
            observation = observation[:1].upper() + observation[1:]
        prefix = "Yes, " if include_confirmation else ""
        return (
            f"{prefix}{observation} in {camera.name} on {readable_date} at {readable_time} "
            f"with {incident.confidence:.0%} confidence."
        )

    @staticmethod
    def _readable_local_datetime(occurred_at: datetime) -> tuple[str, str]:
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        local_time = occurred_at.astimezone(_DISPLAY_TIMEZONE)
        readable_time = (
            f"{local_time.hour % 12 or 12}:{local_time.minute:02d} "
            f"{'AM' if local_time.hour < 12 else 'PM'}"
        )
        readable_date = f"{local_time.month}/{local_time.day}/{local_time.year}"
        return readable_date, readable_time

    async def _resolve_filters(
        self, payload: VideoRagQueryRequest
    ) -> tuple[list[UUID], datetime | None, datetime | None]:
        if payload.camera_ids and payload.start_at and payload.end_at:
            return payload.camera_ids, payload.start_at, payload.end_at
        cameras = list(await self.session.scalars(select(Camera).order_by(Camera.name)))
        question = payload.question.casefold()
        matched_camera_ids = [
            camera.id
            for camera in cameras
            if re.search(rf"(?<!\w){re.escape(camera.name.casefold())}(?!\w)", question)
            and (
                camera.name.casefold() not in _RESERVED_CAMERA_QUERY_TERMS
                or re.search(
                    rf"\bcamera\s+{re.escape(camera.name.casefold())}\b",
                    question,
                )
            )
        ]
        camera_ids = payload.camera_ids or matched_camera_ids
        relative_start, relative_end = (
            (None, None)
            if payload.start_at or payload.end_at
            else self._relative_time_range(payload.question)
        )
        start_at = payload.start_at or relative_start
        end_at = payload.end_at or relative_end
        if relative_start or relative_end:
            return camera_ids, start_at, end_at
        if not _FILTER_HINT_PATTERN.search(payload.question):
            return camera_ids, start_at, end_at

        extracted = await self.ollama.extract_filters(
            payload.question,
            [{"id": str(camera.id), "name": camera.name} for camera in cameras],
            datetime.now().astimezone().isoformat(),
        )
        if not camera_ids:
            requested_names = extracted.get("camera_names", [])
            if isinstance(requested_names, list):
                normalized = {str(name).strip().casefold() for name in requested_names}
                camera_ids = [
                    camera.id for camera in cameras if camera.name.strip().casefold() in normalized
                ]
        return (
            camera_ids,
            start_at or self._parse_datetime(extracted.get("start_at")),
            end_at or self._parse_datetime(extracted.get("end_at")),
        )

    @classmethod
    def _requested_result_limit(cls, question: str, default: int) -> int:
        for pattern in _RESULT_LIMIT_PATTERNS:
            if match := pattern.search(question):
                return cls._count_value(match.group("count"))
        return default

    @classmethod
    def _relative_time_range(
        cls, question: str, *, now: datetime | None = None
    ) -> tuple[datetime | None, datetime | None]:
        reference = now or datetime.now(UTC)
        local_reference = reference.astimezone(_DISPLAY_TIMEZONE)
        if re.search(r"\byesterday\b", question, re.IGNORECASE):
            end_local = local_reference.replace(hour=0, minute=0, second=0, microsecond=0)
            start_local = end_local - timedelta(days=1)
            return (
                start_local.astimezone(UTC),
                (end_local - timedelta(microseconds=1)).astimezone(UTC),
            )
        if re.search(r"\btoday\b", question, re.IGNORECASE):
            start_local = local_reference.replace(hour=0, minute=0, second=0, microsecond=0)
            return start_local.astimezone(UTC), reference

        match = _RELATIVE_TIME_PATTERN.search(question)
        if not match:
            return None, None
        count = cls._count_value(match.group("count"))
        unit = match.group("unit").casefold().rstrip("s")
        delta = timedelta(**{f"{unit}s": count})
        end_at = reference
        return end_at - delta, end_at

    @staticmethod
    def _count_value(value: str) -> int:
        normalized = value.casefold()
        return _COUNT_WORDS.get(normalized, int(normalized) if normalized.isdigit() else 1)

    async def _retrieve(
        self,
        question: str,
        embedding: list[float] | None,
        *,
        camera_ids: list[UUID],
        start_at: datetime | None,
        end_at: datetime | None,
        limit: int,
    ) -> list[RetrievedChunk]:
        base = (
            select(VideoRagChunk, Incident, Camera)
            .join(Incident, Incident.id == VideoRagChunk.incident_id)
            .join(Camera, Camera.id == Incident.camera_id)
            .join(VideoRagIndex, VideoRagIndex.incident_id == Incident.id)
            .where(
                Incident.archived_at.is_(None),
                VideoRagIndex.status == VideoRagIndexStatus.ready,
            )
        )
        if camera_ids:
            base = base.where(Incident.camera_id.in_(camera_ids))
        if start_at:
            base = base.where(Incident.occurred_at >= start_at)
        if end_at:
            base = base.where(Incident.occurred_at <= end_at)

        candidate_limit = max(20, limit * 6)
        if self.session.bind and self.session.bind.dialect.name == "postgresql":
            vector_rows: list = []
            distance = literal(1.0)
            if embedding is not None:
                distance = VideoRagChunk.embedding.cosine_distance(embedding)
                vector_rows = (
                    await self.session.execute(
                        base.add_columns(distance).order_by(distance).limit(candidate_limit)
                    )
                ).all()
            english = literal_column("'english'")
            lexical_query = func.to_tsquery(english, self._lexical_tsquery(question))
            rank = func.ts_rank_cd(
                func.to_tsvector(english, VideoRagChunk.content),
                lexical_query,
            )
            lexical_rows = (
                await self.session.execute(
                    base.add_columns(rank, distance)
                    .where(func.to_tsvector(english, VideoRagChunk.content).op("@@")(lexical_query))
                    .order_by(rank.desc())
                    .limit(candidate_limit)
                )
            ).all()
            return self._fuse(vector_rows, lexical_rows, limit)

        rows = (await self.session.execute(base)).all()
        if embedding is None:
            terms = self._lexical_terms(question)
            ranked_rows = sorted(
                rows,
                key=lambda row: sum(term in row[0].content.casefold() for term in terms),
                reverse=True,
            )
            return [
                RetrievedChunk(chunk, incident, camera, settings.video_rag_min_relevance)
                for chunk, incident, camera in ranked_rows
                if any(term in chunk.content.casefold() for term in terms)
            ][:limit]
        ranked = sorted(
            rows,
            key=lambda row: self._cosine_similarity(list(row[0].embedding), embedding),
            reverse=True,
        )
        results: list[RetrievedChunk] = []
        seen: set[UUID] = set()
        for chunk, incident, camera in ranked:
            score = self._cosine_similarity(list(chunk.embedding), embedding)
            if score < settings.video_rag_min_relevance or incident.id in seen:
                continue
            seen.add(incident.id)
            results.append(RetrievedChunk(chunk, incident, camera, score))
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _lexical_terms(question: str) -> list[str]:
        terms = [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9_]+", question)
            if token.casefold() not in _LEXICAL_STOP_WORDS and len(token) > 1
        ]
        return terms or ["incident"]

    @classmethod
    def _structured_terms(cls, question: str) -> list[str]:
        terms: list[str] = []
        for term in cls._lexical_terms(question):
            if term in _STRUCTURED_GENERIC_TERMS:
                continue
            normalized = term[:-1] if term.endswith("s") and len(term) > 3 else term
            if normalized and normalized not in terms:
                terms.append(normalized)
        return terms

    @staticmethod
    def _incident_subtype(incident: Incident) -> str | None:
        return next(
            (
                str(box.get("label"))
                for box in incident.bounding_boxes
                if box.get("label") and box.get("label") != "face"
            ),
            None,
        )

    @classmethod
    def _authoritative_match_score(
        cls, terms: list[str], incident: Incident, camera: Camera
    ) -> tuple[int, int]:
        identity = incident.recognized_identity or {}
        event_text = " ".join(
            [
                incident.detection_type.value,
                incident.detection_type.value.replace("_", " "),
                incident.priority.value,
                incident.status.value,
                incident.operator_notes or "",
                str(identity.get("identity_label") or ""),
                *(str(box.get("label") or "") for box in incident.bounding_boxes),
            ]
        ).casefold()
        camera_text = " ".join(
            [
                camera.name,
                camera.location or "",
            ]
        ).casefold()
        return (
            sum(term in event_text for term in terms),
            sum(term in camera_text for term in terms),
        )

    @classmethod
    def _authoritative_excerpt(cls, incident: Incident, camera: Camera) -> str:
        subtype = cls._incident_subtype(incident)
        subtype_text = f", subtype {subtype}" if subtype else ""
        identity = (incident.recognized_identity or {}).get("identity_label")
        identity_text = f", recognized identity {identity}" if identity else ""
        notes_text = (
            f", operator notes: {incident.operator_notes}" if incident.operator_notes else ""
        )
        return (
            f"Authoritative detector record: {incident.detection_type.value.replace('_', ' ')}"
            f"{subtype_text} at {camera.name} ({camera.location or 'location unspecified'}) on "
            f"{incident.occurred_at.isoformat()}, confidence {incident.confidence:.0%}, "
            f"status {incident.status.value}{identity_text}{notes_text}."
        )

    @classmethod
    def _lexical_tsquery(cls, question: str) -> str:
        return " | ".join(cls._lexical_terms(question))

    def _fuse(self, vector_rows: list, lexical_rows: list, limit: int) -> list[RetrievedChunk]:
        fused: dict[UUID, dict[str, object]] = {}
        for rank, row in enumerate(vector_rows, start=1):
            chunk, incident, camera, distance = row
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            fused[chunk.id] = {
                "item": RetrievedChunk(chunk, incident, camera, similarity),
                "rrf": 1 / (60 + rank),
            }
        for rank, row in enumerate(lexical_rows, start=1):
            chunk, incident, camera, _, distance = row
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            entry = fused.setdefault(
                chunk.id,
                {
                    "item": RetrievedChunk(
                        chunk,
                        incident,
                        camera,
                        max(settings.video_rag_min_relevance, similarity),
                    ),
                    "rrf": 0.0,
                },
            )
            existing_item = entry["item"]
            if isinstance(existing_item, RetrievedChunk):
                existing_item.score = max(existing_item.score, settings.video_rag_min_relevance)
            entry["rrf"] = float(entry["rrf"]) + 1 / (60 + rank)
        ordered = sorted(fused.values(), key=lambda entry: float(entry["rrf"]), reverse=True)
        results: list[RetrievedChunk] = []
        seen: set[UUID] = set()
        for entry in ordered:
            item = entry["item"]
            assert isinstance(item, RetrievedChunk)
            if item.score < settings.video_rag_min_relevance or item.incident.id in seen:
                continue
            seen.add(item.incident.id)
            results.append(item)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _context(item: RetrievedChunk) -> dict[str, object]:
        return {
            "incident_id": str(item.incident.id),
            "camera": item.camera.name,
            "location": item.camera.location,
            "occurred_at": item.incident.occurred_at.isoformat(),
            "detection_type": item.incident.detection_type.value,
            "confidence": item.incident.confidence,
            "recognized_identity": item.incident.recognized_identity,
            "evidence_excerpt": item.chunk.content[:500],
            "clip_start_seconds": item.chunk.clip_start_seconds,
        }

    @staticmethod
    def _evidence(item: RetrievedChunk) -> VideoRagEvidence:
        return VideoRagEvidence(
            incident_id=item.incident.id,
            camera_id=item.camera.id,
            camera_name=item.camera.name,
            occurred_at=item.incident.occurred_at,
            detection_type=item.incident.detection_type,
            confidence=item.incident.confidence,
            matched_excerpt=item.chunk.content,
            relevance_score=round(item.score, 4),
            clip_start_seconds=item.chunk.clip_start_seconds,
            clip_end_seconds=item.chunk.clip_end_seconds,
            snapshot_url=f"/api/v1/incidents/{item.incident.id}/snapshot"
            if item.incident.snapshot_path
            else None,
            clip_url=f"/api/v1/incidents/{item.incident.id}/clip"
            if item.incident.clip_path
            else None,
        )

    @classmethod
    def _fallback_answer(cls, item: RetrievedChunk) -> str:
        return cls._concise_incident_answer(
            item.incident,
            item.camera,
        )

    @staticmethod
    def _ground_answer(
        generated: str,
        matches: list[RetrievedChunk],
        *,
        question: str = "the question",
    ) -> tuple[str, str | None]:
        normalized = generated.strip().rstrip(".").casefold().replace(" ", "_")
        if normalized == "not_found":
            return VideoRagQueryService._not_found_answer(question), None
        if generated.count("[incident:") > generated.count("]"):
            incomplete = generated.rfind("[incident:")
            prefix = generated[:incomplete].rstrip(" ,;:")
            sentence_end = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
            generated = prefix[: sentence_end + 1] if sentence_end >= 0 else prefix
        allowed = {str(item.incident.id) for item in matches}
        cited = set(_CITATION_PATTERN.findall(generated))
        invalid = cited - allowed
        if invalid:
            for incident_id in invalid:
                generated = generated.replace(f"[incident:{incident_id}]", "")
        valid = cited & allowed
        if not valid:
            return (
                "A verified response could not be produced from the retrieved evidence.",
                "The generated answer was replaced because it did not cite retrieved evidence.",
            )
        warning = (
            "Unsupported incident citations were removed from the generated answer."
            if invalid
            else None
        )
        answer = _CITATION_PATTERN.sub("", generated)
        answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
        answer = re.sub(r" {2,}", " ", answer)
        return answer.strip(), warning

    @staticmethod
    def _not_found_answer(_question: str) -> str:
        return "The available incident records do not contain sufficient evidence to answer this question."

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None

    @staticmethod
    def _cosine_similarity(first: list[float], second: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(first, second, strict=False))
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if not first_norm or not second_norm:
            return 0.0
        return numerator / (first_norm * second_norm)


def question_digest(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
