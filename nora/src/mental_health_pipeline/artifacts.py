"""Hash-bound artifact inventory used by startup and deployment checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .guardrails import sha256_file


class ArtifactManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactCheck:
    artifact_id: str
    ready: bool
    issues: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "ready": self.ready, "issues": list(self.issues)}


def load_manifest(path: str | Path) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "pipeline-artifacts/1.0":
        raise ArtifactManifestError("unsupported artifact manifest schema")
    if not isinstance(data.get("artifacts"), list):
        raise ArtifactManifestError("manifest artifacts must be a list")
    return data


def verify_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[ArtifactCheck, ...]:
    data = load_manifest(path)
    env = os.environ if environ is None else environ
    root = Path(project_root).resolve()
    checks: list[ArtifactCheck] = []
    seen: set[str] = set()
    for item in data["artifacts"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("artifact_id"), str):
            raise ArtifactManifestError("invalid artifact entry")
        artifact_id = item["artifact_id"]
        if artifact_id in seen:
            raise ArtifactManifestError(f"duplicate artifact id: {artifact_id}")
        seen.add(artifact_id)
        issues: list[str] = []
        base = _resolve_base(item, root, env, issues)
        if base is not None:
            for file_info in item.get("files", []):
                relative = file_info.get("path") if isinstance(file_info, Mapping) else None
                expected = file_info.get("sha256") if isinstance(file_info, Mapping) else None
                if not isinstance(relative, str) or not isinstance(expected, str):
                    raise ArtifactManifestError(f"invalid file entry for {artifact_id}")
                if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                    raise ArtifactManifestError(
                        f"invalid SHA-256 for {artifact_id}:{relative}"
                    )
                target = (base / relative).resolve()
                try:
                    target.relative_to(base.resolve())
                except ValueError as error:
                    raise ArtifactManifestError("artifact path escapes its configured root") from error
                if not target.is_file():
                    issues.append(f"missing:{relative}")
                elif sha256_file(target).lower() != expected.lower():
                    issues.append(f"hash_mismatch:{relative}")
            any_of = item.get("required_any", [])
            if not isinstance(any_of, list) or not all(
                isinstance(name, str) and name for name in any_of
            ):
                raise ArtifactManifestError(f"invalid required_any for {artifact_id}")
            resolved_any: list[Path] = []
            for name in any_of:
                candidate = (base / name).resolve()
                try:
                    candidate.relative_to(base.resolve())
                except ValueError as error:
                    raise ArtifactManifestError(
                        "required_any path escapes its configured root"
                    ) from error
                resolved_any.append(candidate)
            if resolved_any and not any(candidate.is_file() for candidate in resolved_any):
                issues.append("missing_any:" + "|".join(any_of))
        checks.append(ArtifactCheck(artifact_id, not issues, tuple(issues)))
    return tuple(checks)


def _resolve_base(
    item: Mapping[str, Any],
    project_root: Path,
    env: Mapping[str, str],
    issues: list[str],
) -> Path | None:
    if "root_env" in item:
        name = item["root_env"]
        if not isinstance(name, str):
            raise ArtifactManifestError("root_env must be a string")
        raw = env.get(name)
        if not raw:
            issues.append(f"environment_missing:{name}")
            return None
        base = Path(raw).expanduser().resolve()
    elif "root_relative" in item:
        relative = item["root_relative"]
        if not isinstance(relative, str):
            raise ArtifactManifestError("root_relative must be a string")
        base = (project_root / relative).resolve()
        try:
            base.relative_to(project_root)
        except ValueError as error:
            raise ArtifactManifestError("artifact root escapes project root") from error
    else:
        raise ArtifactManifestError("artifact requires root_env or root_relative")
    if not base.is_dir():
        issues.append("root_missing")
        return None
    return base
