"""Versioned, location-aware crisis resources with production approval gates.

Incorrect crisis contact information is worse than a conservative worldwide
fallback. The built-in catalog contains a small set of contacts checked against
primary sources, but is deliberately marked development-only until a project
owner supplies an explicitly approved JSON registry. Location is never inferred
from IP address, language, timezone, region, or telephone number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping


RESOURCE_REGISTRY_SCHEMA: Final = "emergency-resources/1.0"
RESOURCE_CATALOG_VERSION: Final = "2026.10.02"
RESOURCE_CATALOG_REVIEWED_ON: Final = date(2026, 10, 2)
MAX_REGISTRY_BYTES: Final = 262_144


class ResourceRegistryError(ValueError):
    """The registry is malformed or is not authorized for its requested mode."""


@dataclass(frozen=True, slots=True)
class CrisisResource:
    """A crisis resource whose geographic scope and provenance are explicit."""

    identifier: str
    name: str
    scope: str
    contact: str
    action: str
    source_url: str
    channels: tuple[str, ...] = ()
    reviewed_on: date = RESOURCE_CATALOG_REVIEWED_ON
    approved_for_production: bool = False
    approval_reference: str = "pending_project_owner_approval"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "name": self.name,
            "scope": self.scope,
            "contact": self.contact,
            "action": self.action,
            "source_url": self.source_url,
            "channels": list(self.channels),
            "reviewed_on": self.reviewed_on.isoformat(),
            "approved_for_production": self.approved_for_production,
            "approval_reference": self.approval_reference,
        }


@dataclass(frozen=True, slots=True)
class ResourceBundle:
    """Resolved resources for an explicitly supplied country, or worldwide."""

    catalog_version: str
    reviewed_on: date
    country_code: str | None
    location_basis: str
    immediate_danger_instruction: str
    resources: tuple[CrisisResource, ...]
    approved_for_production: bool = False

    @property
    def locale_is_known(self) -> bool:
        return self.country_code is not None

    def render_plain_text(self) -> str:
        """Render a deterministic block suitable for a response-policy packet."""

        lines = [self.immediate_danger_instruction]
        for resource in self.resources:
            lines.append(f"{resource.name}: {resource.action}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_version": self.catalog_version,
            "reviewed_on": self.reviewed_on.isoformat(),
            "country_code": self.country_code,
            "location_basis": self.location_basis,
            "immediate_danger_instruction": self.immediate_danger_instruction,
            "resources": [resource.to_dict() for resource in self.resources],
            "approved_for_production": self.approved_for_production,
        }


@dataclass(frozen=True, slots=True)
class _CountryEntry:
    aliases: tuple[str, ...]
    immediate_danger_instruction: str
    resources: tuple[CrisisResource, ...]


_WORLDWIDE_DIRECTORY: Final = CrisisResource(
    identifier="worldwide.findahelpline",
    name="Find A Helpline",
    scope="worldwide directory",
    contact="https://findahelpline.com/",
    action=(
        "Visit https://findahelpline.com/ and select your country to find a "
        "verified local emotional-support or crisis service."
    ),
    source_url="https://findahelpline.com/about",
    channels=("web",),
)

_CANADA_988: Final = CrisisResource(
    identifier="ca.988",
    name="Canada 9-8-8 Suicide Crisis Helpline",
    scope="Canada",
    contact="9-8-8",
    action="Call or text 9-8-8 for 24/7 suicide-crisis support in Canada.",
    source_url="https://www.canada.ca/en/public-health/services/suicide-prevention.html",
    channels=("call", "text"),
)

_CANADA_911: Final = CrisisResource(
    identifier="ca.911",
    name="Canadian emergency services",
    scope="Canada",
    contact="9-1-1",
    action="Call 9-1-1 if you or someone else is in immediate danger in Canada.",
    source_url=(
        "https://www.canada.ca/en/public-health/services/suicide-prevention/"
        "warning-signs.html"
    ),
    channels=("call",),
)

_UNITED_STATES_988: Final = CrisisResource(
    identifier="us.988",
    name="988 Suicide & Crisis Lifeline",
    scope="United States",
    contact="988",
    action=(
        "Call or text 988 for 24/7 crisis support in the United States, or use "
        "chat at https://988lifeline.org/."
    ),
    source_url="https://988lifeline.org/",
    channels=("call", "text", "web chat"),
)

_UNITED_STATES_911: Final = CrisisResource(
    identifier="us.911",
    name="United States emergency services",
    scope="United States",
    contact="911",
    action="Call 911 if you or someone else is in immediate danger in the United States.",
    source_url="https://www.911.gov/calling-911/",
    channels=("call",),
)

_BUILTIN_WORLDWIDE_INSTRUCTION: Final = (
    "If you or someone else may be in immediate danger, contact local emergency "
    "services or go to the nearest emergency department now. If there is no "
    "immediate danger but urgent support is needed, contact a local crisis line, "
    "health worker, or trusted person."
)

_BUILTIN_ENTRIES: Final[Mapping[str, _CountryEntry]] = MappingProxyType(
    {
        "CA": _CountryEntry(
            aliases=("ca", "can", "canada"),
            immediate_danger_instruction=(
                "If you or someone else may be in immediate danger in Canada, "
                "call 9-1-1 or go to the nearest emergency department now."
            ),
            resources=(_CANADA_911, _CANADA_988),
        ),
        "US": _CountryEntry(
            aliases=("us", "usa", "united states", "united states of america"),
            immediate_danger_instruction=(
                "If you or someone else may be in immediate danger in the United "
                "States, call 911 or go to the nearest emergency department now."
            ),
            resources=(_UNITED_STATES_911, _UNITED_STATES_988),
        ),
    }
)


def _clean_country_key(country: str) -> str:
    return " ".join(str(country).strip().lower().replace("_", " ").split())


def normalize_country(country: str | None) -> str | None:
    """Normalize only built-in supported aliases, without geographic guessing."""

    if country is None:
        return None
    cleaned = _clean_country_key(country)
    for code, entry in _BUILTIN_ENTRIES.items():
        if cleaned == code.lower() or cleaned in entry.aliases:
            return code
    return None


class ResourceDirectory:
    """Resolve a reviewed registry using only explicit location information.

    Pass ``production=True`` only with a registry for which the catalog *and
    every selectable record* carry current, explicit approval. Construction
    fails otherwise, so an unapproved contact cannot silently reach production.
    """

    def __init__(
        self,
        registry_path: str | Path | None = None,
        *,
        production: bool = False,
        as_of: date | None = None,
        max_review_age_days: int = 180,
    ) -> None:
        if max_review_age_days < 1:
            raise ValueError("max_review_age_days must be positive")
        self._production = production
        self._as_of = as_of
        self._max_review_age_days = max_review_age_days

        if registry_path is None:
            self.version = RESOURCE_CATALOG_VERSION
            self.reviewed_on = RESOURCE_CATALOG_REVIEWED_ON
            self.schema_version = RESOURCE_REGISTRY_SCHEMA
            self.approved_for_production = False
            self.approval_reference = "pending_project_owner_approval"
            self._worldwide_instruction = _BUILTIN_WORLDWIDE_INSTRUCTION
            self._worldwide_resources = (_WORLDWIDE_DIRECTORY,)
            self._entries = dict(_BUILTIN_ENTRIES)
        else:
            self._load(Path(registry_path))

        self._aliases = self._build_aliases()
        self._validate_catalog()
        if self._production:
            self.require_production_ready()

    def _load(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ResourceRegistryError("resource registry is not readable") from error
        if size > MAX_REGISTRY_BYTES:
            raise ResourceRegistryError("resource registry exceeds the size limit")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ResourceRegistryError("resource registry is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ResourceRegistryError("resource registry root must be an object")
        _reject_extra_keys(
            payload,
            {
                "schema_version",
                "catalog_version",
                "reviewed_on",
                "approved_for_production",
                "approval_reference",
                "worldwide",
                "countries",
            },
            "registry",
        )
        if payload.get("schema_version") != RESOURCE_REGISTRY_SCHEMA:
            raise ResourceRegistryError("unsupported resource registry schema_version")

        self.schema_version = RESOURCE_REGISTRY_SCHEMA
        self.version = _required_text(payload, "catalog_version", "registry")
        self.reviewed_on = _required_date(payload, "reviewed_on", "registry")
        self.approved_for_production = _required_bool(
            payload, "approved_for_production", "registry"
        )
        self.approval_reference = _required_text(payload, "approval_reference", "registry")

        worldwide = _required_mapping(payload, "worldwide", "registry")
        _reject_extra_keys(
            worldwide,
            {"immediate_danger_instruction", "resources"},
            "worldwide",
        )
        self._worldwide_instruction = _required_text(
            worldwide, "immediate_danger_instruction", "worldwide"
        )
        self._worldwide_resources = _parse_resources(
            worldwide.get("resources"), scope="worldwide directory", country_code=None
        )
        if not self._worldwide_resources:
            raise ResourceRegistryError("worldwide.resources must not be empty")

        raw_countries = _required_mapping(payload, "countries", "registry")
        entries: dict[str, _CountryEntry] = {}
        for raw_code, raw_entry in raw_countries.items():
            code = str(raw_code).strip().upper()
            if len(code) != 2 or not code.isalpha():
                raise ResourceRegistryError("country keys must be two-letter ISO-style codes")
            if not isinstance(raw_entry, Mapping):
                raise ResourceRegistryError(f"country {code} must be an object")
            _reject_extra_keys(
                raw_entry,
                {"aliases", "immediate_danger_instruction", "resources"},
                f"country {code}",
            )
            aliases = _required_text_list(raw_entry, "aliases", f"country {code}")
            instruction = _required_text(
                raw_entry, "immediate_danger_instruction", f"country {code}"
            )
            _enforce_911_988_scope(instruction, code, f"country {code} instruction")
            records = _parse_resources(
                raw_entry.get("resources"), scope=code, country_code=code
            )
            if not records:
                raise ResourceRegistryError(f"country {code}.resources must not be empty")
            entries[code] = _CountryEntry(
                aliases=tuple(_clean_country_key(item) for item in aliases),
                immediate_danger_instruction=instruction,
                resources=records,
            )
        self._entries = entries

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for code, entry in self._entries.items():
            candidates = (code.lower(), *entry.aliases)
            for candidate in candidates:
                cleaned = _clean_country_key(candidate)
                existing = aliases.get(cleaned)
                if existing is not None and existing != code:
                    raise ResourceRegistryError("country alias is assigned more than once")
                aliases[cleaned] = code
        return aliases

    def _validate_catalog(self) -> None:
        if not self.version.strip():
            raise ResourceRegistryError("catalog_version must not be blank")
        if not self._worldwide_instruction.strip():
            raise ResourceRegistryError("worldwide instruction must not be blank")
        if self.approved_for_production and not self.approval_reference.strip():
            raise ResourceRegistryError("approved catalog requires approval_reference")
        identifiers: set[str] = set()
        for resource in self._all_resources():
            if resource.identifier in identifiers:
                raise ResourceRegistryError("resource identifiers must be unique")
            identifiers.add(resource.identifier)
            if not resource.source_url.startswith("https://"):
                raise ResourceRegistryError(
                    f"resource {resource.identifier} source_url must use HTTPS"
                )
            if resource.approved_for_production and not resource.approval_reference.strip():
                raise ResourceRegistryError(
                    f"approved resource {resource.identifier} requires approval_reference"
                )

    def _all_resources(self) -> tuple[CrisisResource, ...]:
        records = list(self._worldwide_resources)
        for entry in self._entries.values():
            records.extend(entry.resources)
        return tuple(records)

    def normalize_country(self, country: str | None) -> str | None:
        if country is None:
            return None
        return self._aliases.get(_clean_country_key(country))

    def resolve(self, country: str | None = None) -> ResourceBundle:
        if self._production:
            self.require_production_ready()
        normalized = self.normalize_country(country)
        if normalized is not None:
            entry = self._entries[normalized]
            resources = (*entry.resources, *self._worldwide_resources)
            return ResourceBundle(
                catalog_version=self.version,
                reviewed_on=self.reviewed_on,
                country_code=normalized,
                location_basis="explicit_user_or_trusted_client_country",
                immediate_danger_instruction=entry.immediate_danger_instruction,
                resources=resources,
                approved_for_production=self.production_ready,
            )

        return ResourceBundle(
            catalog_version=self.version,
            reviewed_on=self.reviewed_on,
            country_code=None,
            location_basis="location_not_assumed",
            immediate_danger_instruction=self._worldwide_instruction,
            resources=self._worldwide_resources,
            approved_for_production=self.production_ready,
        )

    @property
    def production_ready(self) -> bool:
        return not self.readiness_issues()

    def readiness_issues(self, *, as_of: date | None = None) -> tuple[str, ...]:
        effective_date = as_of or self._as_of or date.today()
        issues: list[str] = []
        if not self.approved_for_production:
            issues.append("catalog_not_explicitly_approved")
        if not _review_is_current(
            self.reviewed_on, effective_date, self._max_review_age_days
        ):
            issues.append("catalog_review_not_current")
        for resource in self._all_resources():
            if not resource.approved_for_production:
                issues.append(f"resource_not_approved:{resource.identifier}")
            if not _review_is_current(
                resource.reviewed_on, effective_date, self._max_review_age_days
            ):
                issues.append(f"resource_review_not_current:{resource.identifier}")
        return tuple(issues)

    def require_production_ready(self, *, as_of: date | None = None) -> None:
        issues = self.readiness_issues(as_of=as_of)
        if issues:
            raise ResourceRegistryError(
                "resource registry is not production-ready: " + ", ".join(issues)
            )

    def is_review_current(self, *, as_of: date | None = None, max_age_days: int = 180) -> bool:
        if max_age_days < 1:
            raise ValueError("max_age_days must be positive")
        return _review_is_current(self.reviewed_on, as_of or date.today(), max_age_days)


def _parse_resources(
    value: Any,
    *,
    scope: str,
    country_code: str | None,
) -> tuple[CrisisResource, ...]:
    if not isinstance(value, list):
        raise ResourceRegistryError(f"{scope}.resources must be an array")
    records: list[CrisisResource] = []
    allowed = {
        "id",
        "name",
        "contact",
        "action",
        "channels",
        "source_url",
        "reviewed_on",
        "approved_for_production",
        "approval_reference",
    }
    for index, raw in enumerate(value):
        context = f"{scope}.resources[{index}]"
        if not isinstance(raw, Mapping):
            raise ResourceRegistryError(f"{context} must be an object")
        _reject_extra_keys(raw, allowed, context)
        contact = _required_text(raw, "contact", context)
        action = _required_text(raw, "action", context)
        _enforce_911_988_scope(f"{contact} {action}", country_code, context)
        records.append(
            CrisisResource(
                identifier=_required_text(raw, "id", context),
                name=_required_text(raw, "name", context),
                scope=scope,
                contact=contact,
                action=action,
                source_url=_required_text(raw, "source_url", context),
                channels=tuple(_required_text_list(raw, "channels", context)),
                reviewed_on=_required_date(raw, "reviewed_on", context),
                approved_for_production=_required_bool(
                    raw, "approved_for_production", context
                ),
                approval_reference=_required_text(raw, "approval_reference", context),
            )
        )
    return tuple(records)


def _enforce_911_988_scope(text: str, country_code: str | None, context: str) -> None:
    digits = "".join(character for character in text if character.isdigit())
    if country_code not in {"CA", "US"} and ("911" in digits or "988" in digits):
        raise ResourceRegistryError(f"{context} may not use 911 or 988 outside Canada/US")


def _review_is_current(reviewed_on: date, as_of: date, max_age_days: int) -> bool:
    age = (as_of - reviewed_on).days
    return 0 <= age <= max_age_days


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ResourceRegistryError(f"{context}.{key} must be an object")
    return result


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ResourceRegistryError(f"{context}.{key} must be a non-empty string")
    return result.strip()


def _required_bool(value: Mapping[str, Any], key: str, context: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ResourceRegistryError(f"{context}.{key} must be a boolean")
    return result


def _required_date(value: Mapping[str, Any], key: str, context: str) -> date:
    raw = _required_text(value, key, context)
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise ResourceRegistryError(f"{context}.{key} must be YYYY-MM-DD") from error


def _required_text_list(value: Mapping[str, Any], key: str, context: str) -> list[str]:
    result = value.get(key)
    if not isinstance(result, list) or not all(
        isinstance(item, str) and item.strip() for item in result
    ):
        raise ResourceRegistryError(f"{context}.{key} must be an array of non-empty strings")
    return [item.strip() for item in result]


def _reject_extra_keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    extras = set(value) - allowed
    if extras:
        raise ResourceRegistryError(f"{context} contains unsupported fields")


DEFAULT_RESOURCE_DIRECTORY: Final = ResourceDirectory()


__all__ = [
    "CrisisResource",
    "DEFAULT_RESOURCE_DIRECTORY",
    "MAX_REGISTRY_BYTES",
    "RESOURCE_CATALOG_REVIEWED_ON",
    "RESOURCE_CATALOG_VERSION",
    "RESOURCE_REGISTRY_SCHEMA",
    "ResourceBundle",
    "ResourceDirectory",
    "ResourceRegistryError",
    "normalize_country",
]
