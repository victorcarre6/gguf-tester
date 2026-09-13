"""GGUF binary parser. Reads metadata + tensor info without loading weights."""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import Any

GGUF_ALIGNMENT = 32
MAX_HEADER_ENTRIES = 1_000_000
MAX_ARRAY_ITEMS = 1_000_000
MAX_DIMS = 8


class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29


GGML_TYPE_INFO: dict[int, dict[str, Any]] = {
    GGMLType.F32: {"name": "F32", "block_size": 1, "type_size": 4, "bits": 32},
    GGMLType.F16: {"name": "F16", "block_size": 1, "type_size": 2, "bits": 16},
    GGMLType.Q4_0: {"name": "Q4_0", "block_size": 32, "type_size": 18, "bits": 4.5},
    GGMLType.Q4_1: {"name": "Q4_1", "block_size": 32, "type_size": 20, "bits": 5},
    GGMLType.Q5_0: {"name": "Q5_0", "block_size": 32, "type_size": 22, "bits": 5.5},
    GGMLType.Q5_1: {"name": "Q5_1", "block_size": 32, "type_size": 24, "bits": 6},
    GGMLType.Q8_0: {"name": "Q8_0", "block_size": 32, "type_size": 34, "bits": 8.5},
    GGMLType.Q8_1: {"name": "Q8_1", "block_size": 32, "type_size": 36, "bits": 9},
    GGMLType.Q2_K: {"name": "Q2_K", "block_size": 256, "type_size": 82, "bits": 2.5625},
    GGMLType.Q3_K: {"name": "Q3_K", "block_size": 256, "type_size": 110, "bits": 3.4375},
    GGMLType.Q4_K: {"name": "Q4_K", "block_size": 256, "type_size": 144, "bits": 4.5},
    GGMLType.Q5_K: {"name": "Q5_K", "block_size": 256, "type_size": 176, "bits": 5.5},
    GGMLType.Q6_K: {"name": "Q6_K", "block_size": 256, "type_size": 210, "bits": 6.5625},
    GGMLType.Q8_K: {"name": "Q8_K", "block_size": 256, "type_size": 292, "bits": 9.125},
    GGMLType.IQ2_XXS: {"name": "IQ2_XXS", "block_size": 256, "type_size": 66, "bits": 2.0625},
    GGMLType.IQ2_XS: {"name": "IQ2_XS", "block_size": 256, "type_size": 74, "bits": 2.3125},
    GGMLType.IQ3_XXS: {"name": "IQ3_XXS", "block_size": 256, "type_size": 98, "bits": 3.0625},
    GGMLType.IQ1_S: {"name": "IQ1_S", "block_size": 256, "type_size": 50, "bits": 1.5625},
    GGMLType.IQ4_NL: {"name": "IQ4_NL", "block_size": 32, "type_size": 18, "bits": 4.5},
    GGMLType.IQ3_S: {"name": "IQ3_S", "block_size": 256, "type_size": 110, "bits": 3.4375},
    GGMLType.IQ2_S: {"name": "IQ2_S", "block_size": 256, "type_size": 82, "bits": 2.5625},
    GGMLType.IQ4_XS: {"name": "IQ4_XS", "block_size": 256, "type_size": 136, "bits": 4.25},
    GGMLType.I8: {"name": "I8", "block_size": 1, "type_size": 1, "bits": 8},
    GGMLType.I16: {"name": "I16", "block_size": 1, "type_size": 2, "bits": 16},
    GGMLType.I32: {"name": "I32", "block_size": 1, "type_size": 4, "bits": 32},
    GGMLType.I64: {"name": "I64", "block_size": 1, "type_size": 8, "bits": 64},
    GGMLType.F64: {"name": "F64", "block_size": 1, "type_size": 8, "bits": 64},
    GGMLType.IQ1_M: {"name": "IQ1_M", "block_size": 256, "type_size": 56, "bits": 1.75},
}


@dataclass
class ParsedGGUF:
    file_path: str
    filename: str
    file_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors_info: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    @property
    def total_size_mb(self) -> float:
        return sum(t["size_mb"] for t in self.tensors_info.values())

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("general.name", "Unknown"))


class GGUFParser:
    """Parse GGUF binary header (metadata + tensor info). Does not load weights."""

    def __init__(self, file_path: str, filename: str | None = None):
        self.file_path = file_path
        self.filename = filename or os.path.basename(file_path)
        self.file_size = os.path.getsize(file_path)
        self.metadata: dict[str, Any] = {}
        self.tensors_info: dict[str, dict[str, Any]] = {}

    def parse(self) -> ParsedGGUF:
        with open(self.file_path, "rb") as f:
            magic = self._read_exact(f, 4)
            if magic != b"GGUF":
                raise ValueError(f"invalid GGUF magic: expected b'GGUF', got {magic!r}")

            version = self._unpack(f, "<I")
            if version not in (2, 3):
                raise ValueError(f"unsupported GGUF version: {version}")

            tensor_count = self._unpack(f, "<Q")
            metadata_count = self._unpack(f, "<Q")
            if tensor_count > MAX_HEADER_ENTRIES or metadata_count > MAX_HEADER_ENTRIES:
                raise ValueError(
                    "implausible GGUF header counts: "
                    f"tensors={tensor_count}, metadata={metadata_count}"
                )

            self._read_metadata(f, metadata_count)
            self._read_tensor_info(f, tensor_count)

        return ParsedGGUF(
            file_path=self.file_path,
            filename=self.filename,
            file_size=self.file_size,
            metadata=self.metadata,
            tensors_info=self.tensors_info,
        )

    def _read_metadata(self, f, count: int) -> None:
        for _ in range(count):
            key = self._read_string(f)
            value_type = self._unpack(f, "<I")
            self.metadata[key] = self._read_value_by_type(f, value_type)

    def _read_value_by_type(self, f, value_type: int) -> Any:
        formats = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            10: "<Q",
            11: "<q",
            12: "<d",
        }
        if value_type in formats:
            return self._unpack(f, formats[value_type])
        if value_type == 7:
            return self._unpack(f, "<B") != 0
        if value_type == 8:
            return self._read_string(f)
        if value_type == 9:
            item_type = self._unpack(f, "<I")
            item_count = self._unpack(f, "<Q")
            if item_count > MAX_ARRAY_ITEMS:
                raise ValueError(f"metadata array is too large: {item_count}")

            return [self._read_value_by_type(f, item_type) for _ in range(item_count)]

        raise ValueError(f"unsupported GGUF metadata type: {value_type}")

    def _read_tensor_info(self, f, count: int) -> None:
        for _ in range(count):
            name = self._read_string(f)
            dimensions = self._unpack(f, "<I")
            if dimensions > MAX_DIMS:
                raise ValueError(f"tensor {name!r} has too many dimensions: {dimensions}")

            shape = [self._unpack(f, "<Q") for _ in range(dimensions)]
            ggml_type = self._unpack(f, "<I")
            self._unpack(f, "<Q")  # tensor-data offset
            self.tensors_info[name] = self._tensor_details(shape, ggml_type)

    def _tensor_details(self, shape: list[int], ggml_type: int) -> dict[str, Any]:
        element_count = math.prod(shape) if shape else 0
        type_info = GGML_TYPE_INFO.get(
            ggml_type,
            {"name": "Unknown", "block_size": 1, "type_size": 1, "bits": 32},
        )
        block_size = type_info["block_size"]
        blocks = (element_count + block_size - 1) // block_size
        size_bytes = blocks * type_info["type_size"]

        return {
            "shape": shape,
            "type_name": type_info["name"],
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024 * 1024),
            "element_count": element_count,
            "is_quantized": block_size > 1,
            "bits_per_element": type_info.get("bits", 32),
        }

    def _read_string(self, f) -> str:
        length = self._unpack(f, "<Q")
        if length > self.file_size:
            raise ValueError(f"string length exceeds file size: {length}")

        return self._read_exact(f, length).decode("utf-8", errors="replace")

    @staticmethod
    def _read_exact(f, size: int) -> bytes:
        data = f.read(size)
        if len(data) != size:
            raise ValueError(f"unexpected end of GGUF header: wanted {size} bytes")

        return data

    @classmethod
    def _unpack(cls, f, fmt: str):
        return struct.unpack(fmt, cls._read_exact(f, struct.calcsize(fmt)))[0]


def parse(file_path: str, filename: str | None = None) -> ParsedGGUF:
    """Module-level shortcut."""
    return GGUFParser(file_path, filename).parse()
