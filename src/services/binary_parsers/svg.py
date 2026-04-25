"""SVG parser that converts diagram text into markdown-like content."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from .base import BinaryParseResult


class SvgDocumentParser:
    """Extract human-readable text from SVG documents."""

    format = "svg"
    parser_name = "svg"
    parser_version = 1

    def parse(self, *, content: str, path: str) -> BinaryParseResult:
        root = ET.fromstring(content)

        titles = self._collect_text(root, {"title"})
        descriptions = self._collect_text(root, {"desc"})
        text_labels = self._collect_text(root, {"text", "tspan"})
        aria_labels = self._collect_attributes(root, {"aria-label", "inkscape:label"})

        lines = ["# SVG Diagram", f"Source path: `{path}`"]
        metadata: dict[str, str] = {}

        view_box = root.attrib.get("viewBox")
        if view_box:
            metadata["viewBox"] = view_box
        width = root.attrib.get("width")
        height = root.attrib.get("height")
        if width or height:
            metadata["size"] = f"{width or '?'} x {height or '?'}"

        if titles:
            lines.extend(["", "## Title"])
            lines.extend(f"- {value}" for value in titles[:5])

        if descriptions:
            lines.extend(["", "## Description"])
            lines.extend(f"- {value}" for value in descriptions[:8])

        if text_labels:
            lines.extend(["", "## Text Labels"])
            lines.extend(f"- {value}" for value in text_labels[:40])

        if aria_labels:
            lines.extend(["", "## Accessibility Labels"])
            lines.extend(f"- {value}" for value in aria_labels[:20])

        if metadata:
            lines.extend(["", "## Metadata"])
            lines.extend(f"- {key}: {value}" for key, value in metadata.items())

        if len(lines) <= 2:
            lines.extend(
                [
                    "",
                    "## Extracted Content",
                    "- No human-readable labels were found in this SVG.",
                ]
            )

        return BinaryParseResult(
            content="\n".join(lines).strip(),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            metadata=metadata,
        )

    @staticmethod
    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.rsplit("}", 1)[-1]
        if ":" in tag:
            return tag.rsplit(":", 1)[-1]
        return tag

    def _collect_text(self, root: ET.Element, names: set[str]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for element in root.iter():
            if self._local_name(element.tag) not in names:
                continue
            value = " ".join("".join(element.itertext()).split()).strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
        return values

    def _collect_attributes(self, root: ET.Element, names: set[str]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for element in root.iter():
            for attr_name, attr_value in element.attrib.items():
                local_name = self._local_name(attr_name)
                if attr_name not in names and local_name not in names:
                    continue
                value = " ".join(attr_value.split()).strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        return values
