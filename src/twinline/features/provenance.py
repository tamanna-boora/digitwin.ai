"""Per-feature data provenance — flows through to the UI so operators know
whether a number came off a real sensor, a soft-sensor estimate, or a
manual checklist.
"""

from enum import Enum


class Provenance(str, Enum):
    REAL = "real"
    SOFT = "soft"
    MANUAL = "manual"
