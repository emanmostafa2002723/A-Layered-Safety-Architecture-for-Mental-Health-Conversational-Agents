from __future__ import annotations

from datetime import date, timedelta
import json

import pytest

from mental_health_pipeline.resources import (
    RESOURCE_CATALOG_VERSION,
    ResourceDirectory,
    ResourceRegistryError,
    normalize_country,
)


def _digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def test_location_is_never_assumed() -> None:
    bundle = ResourceDirectory().resolve(None)

    assert bundle.country_code is None
    assert bundle.location_basis == "location_not_assumed"
    rendered = bundle.render_plain_text()
    assert "local emergency services" in rendered
    assert "findahelpline.com" in rendered
    assert "911" not in _digits(rendered)
    assert "988" not in _digits(rendered)


@pytest.mark.parametrize("country", ["CA", "ca", "CAN", "Canada", " canada "])
def test_canada_uses_reviewed_canada_contacts(country: str) -> None:
    bundle = ResourceDirectory().resolve(country)

    assert bundle.country_code == "CA"
    contacts = {_digits(item.contact) for item in bundle.resources}
    assert "911" in contacts
    assert "988" in contacts
    assert bundle.resources[-1].identifier == "worldwide.findahelpline"
    assert any("canada.ca" in item.source_url for item in bundle.resources)


@pytest.mark.parametrize("country", ["US", "USA", "United States", "united_states_of_america"])
def test_united_states_uses_reviewed_us_contacts(country: str) -> None:
    bundle = ResourceDirectory().resolve(country)

    assert bundle.country_code == "US"
    contacts = {_digits(item.contact) for item in bundle.resources}
    assert "911" in contacts
    assert "988" in contacts
    assert any("988lifeline.org" in item.source_url for item in bundle.resources)


@pytest.mark.parametrize("country", ["GB", "Egypt", "Toronto", "unknown", ""])
def test_unsupported_or_ambiguous_location_uses_worldwide_fallback(country: str) -> None:
    bundle = ResourceDirectory().resolve(country)

    assert bundle.country_code is None
    assert [item.identifier for item in bundle.resources] == ["worldwide.findahelpline"]
    assert "911" not in _digits(bundle.render_plain_text())
    assert "988" not in _digits(bundle.render_plain_text())


def test_catalog_is_versioned_and_serializable() -> None:
    bundle = ResourceDirectory().resolve("CA")
    payload = bundle.to_dict()

    assert payload["catalog_version"] == RESOURCE_CATALOG_VERSION
    assert payload["reviewed_on"] == "2026-10-02"
    assert isinstance(payload["resources"], list)
    assert all(resource["source_url"].startswith("https://") for resource in payload["resources"])


def test_review_currency_is_enforceable() -> None:
    directory = ResourceDirectory()

    assert directory.is_review_current(as_of=directory.reviewed_on + timedelta(days=180))
    assert not directory.is_review_current(as_of=directory.reviewed_on + timedelta(days=181))
    assert not directory.is_review_current(as_of=date(2026, 10, 1))
    with pytest.raises(ValueError):
        directory.is_review_current(max_age_days=0)


def test_normalize_country_does_not_guess_cities_or_regions() -> None:
    assert normalize_country("Canada") == "CA"
    assert normalize_country("Toronto") is None
    assert normalize_country(None) is None


def _registry(*, approved: bool = True, worldwide_contact: str = "https://findahelpline.com/"):
    approval = "owner-review-2026-10-02"
    resource = {
        "id": "worldwide.findahelpline",
        "name": "Find A Helpline",
        "contact": worldwide_contact,
        "action": "Select your country to locate a verified local crisis service.",
        "channels": ["web"],
        "source_url": "https://findahelpline.com/about",
        "reviewed_on": "2026-10-02",
        "approved_for_production": approved,
        "approval_reference": approval,
    }
    return {
        "schema_version": "emergency-resources/1.0",
        "catalog_version": "test-1",
        "reviewed_on": "2026-10-02",
        "approved_for_production": approved,
        "approval_reference": approval,
        "worldwide": {
            "immediate_danger_instruction": (
                "Contact local emergency services or go to an emergency department."
            ),
            "resources": [resource],
        },
        "countries": {
            "CA": {
                "aliases": ["ca", "can", "canada"],
                "immediate_danger_instruction": "In Canada, call 9-1-1 for immediate danger.",
                "resources": [
                    {
                        "id": "ca.988",
                        "name": "Canada 9-8-8 Suicide Crisis Helpline",
                        "contact": "9-8-8",
                        "action": "Call or text 9-8-8 in Canada.",
                        "channels": ["call", "text"],
                        "source_url": "https://www.canada.ca/en/public-health/services/suicide-prevention.html",
                        "reviewed_on": "2026-10-02",
                        "approved_for_production": approved,
                        "approval_reference": approval,
                    }
                ],
            }
        },
    }


def test_approved_current_json_registry_can_enter_production(tmp_path) -> None:
    path = tmp_path / "emergency_resources.json"
    path.write_text(json.dumps(_registry()), encoding="utf-8")

    directory = ResourceDirectory(path, production=True, as_of=date(2026, 10, 2))
    bundle = directory.resolve("Canada")

    assert directory.production_ready is True
    assert bundle.approved_for_production is True
    assert bundle.country_code == "CA"


def test_unapproved_registry_is_blocked_in_production(tmp_path) -> None:
    path = tmp_path / "emergency_resources.json"
    path.write_text(json.dumps(_registry(approved=False)), encoding="utf-8")

    with pytest.raises(ResourceRegistryError, match="not production-ready"):
        ResourceDirectory(path, production=True, as_of=date(2026, 10, 2))


def test_registry_rejects_911_or_988_in_worldwide_record(tmp_path) -> None:
    path = tmp_path / "emergency_resources.json"
    path.write_text(json.dumps(_registry(worldwide_contact="988")), encoding="utf-8")

    with pytest.raises(ResourceRegistryError, match="outside Canada/US"):
        ResourceDirectory(path)
