"""Shared vocabulary used across plant, model, and record schemas."""

from enum import Enum


class AreaCode(str, Enum):
    BODY = "BODY"
    PAINT = "PAINT"
    FINAL_ASSEMBLY = "FINAL_ASSEMBLY"


class InstrumentationTier(str, Enum):
    RICH = "rich"
    PARTIAL = "partial"
    MANUAL = "manual"


class ShiftId(str, Enum):
    SHIFT_1 = "SHIFT_1"
    SHIFT_2 = "SHIFT_2"


class DefectType(str, Enum):
    WELD_DEFECT = "weld_defect"
    PAINT_DEFECT = "paint_defect"
    ASSEMBLY_DEFECT = "assembly_defect"


class FaultKind(str, Enum):
    TOOL_WEAR = "tool_wear"
    SUPPLIER_BATCH = "supplier_batch"
    OPERATOR_VARIATION = "operator_variation"
    AMBIENT = "ambient"
