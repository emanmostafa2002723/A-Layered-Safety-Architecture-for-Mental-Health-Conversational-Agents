class Severity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision:
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class Action:
    DISPLAY = "DISPLAY"
    REWRITE = "REWRITE"
    SAFE_FALLBACK = "SAFE_FALLBACK"
    ABSTAIN = "ABSTAIN"