"""Strict reader for the TOML subset used by ``configs/*.toml``.

Supported: comments, ``key = value`` with bare/quoted/dotted keys, ``[table]`` and
``[dotted.table]`` headers, basic ``"..."`` and literal ``'...'`` strings, integers
(with ``_`` separators), floats (including ``inf``/``nan``), booleans, arrays
(possibly multi-line, trailing comma allowed) and inline tables.

Anything else (dates, multi-line strings, arrays of tables, hex/octal) raises
:class:`TomlError` rather than being silently misread.
"""
from __future__ import annotations

import math
import re
from typing import Any

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_NUMBER = re.compile(r"[+-]?(?:inf|nan|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d[\d_]*)?)")
_ESCAPES = {'"': '"', "\\": "\\", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class TomlError(ValueError):
    """The text is not valid (or not supported) TOML."""


class _Parser:
    """Single-pass recursive-descent parser over ``text``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.root: dict[str, Any] = {}
        self.current = self.root
        self.defined_tables: set[tuple[str, ...]] = set()

    # -- low-level scanning --------------------------------------------
    def fail(self, message: str) -> TomlError:
        line = self.text.count("\n", 0, self.pos) + 1
        return TomlError(f"line {line}: {message}")

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def skip_blank(self, newlines: bool) -> None:
        """Skip spaces and comments (and line breaks when ``newlines``)."""
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char in " \t" or (newlines and char in "\r\n"):
                self.pos += 1
            elif char == "#":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            else:
                break

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise self.fail(f"expected {char!r}")
        self.pos += 1

    # -- keys ----------------------------------------------------------
    def parse_key_path(self) -> tuple[str, ...]:
        parts = []
        while True:
            self.skip_blank(False)
            parts.append(self.parse_one_key())
            self.skip_blank(False)
            if self.peek() != ".":
                return tuple(parts)
            self.pos += 1

    def parse_one_key(self) -> str:
        if self.peek() in ('"', "'"):
            return self.parse_string()
        match = _BARE_KEY.match(self.text, self.pos)
        if not match:
            raise self.fail("expected a key")
        self.pos = match.end()
        return match.group()

    # -- values --------------------------------------------------------
    def parse_value(self) -> Any:
        char = self.peek()
        if char in ('"', "'"):
            return self.parse_string()
        if char == "[":
            return self.parse_array()
        if char == "{":
            return self.parse_inline_table()
        for literal, value in (("true", True), ("false", False)):
            if self.text.startswith(literal, self.pos):
                self.pos += len(literal)
                return value
        match = _NUMBER.match(self.text, self.pos)
        if not match:
            raise self.fail("unsupported value")
        self.pos = match.end()
        return self.convert_number(match.group())

    def convert_number(self, token: str) -> Any:
        if "__" in token or token.endswith("_") or "_." in token or "._" in token:
            raise self.fail(f"invalid number {token!r}")
        clean = token.replace("_", "")
        body = clean.lstrip("+-")
        if body == "inf":
            return -math.inf if clean[0] == "-" else math.inf
        if body == "nan":
            return math.nan
        if len(body) > 1 and body[0] == "0" and body[1].isdigit():
            raise self.fail(f"leading zeros are not allowed: {token!r}")
        if any(c in clean for c in ".eE"):
            return float(clean)
        return int(clean)

    def parse_string(self) -> str:
        quote = self.text[self.pos]
        if self.text.startswith(quote * 3, self.pos):
            raise self.fail("multi-line strings are not supported")
        self.pos += 1
        out: list[str] = []
        while True:
            if self.pos >= len(self.text) or self.text[self.pos] == "\n":
                raise self.fail("unterminated string")
            char = self.text[self.pos]
            self.pos += 1
            if char == quote:
                return "".join(out)
            if quote == '"' and char == "\\":
                out.append(self.parse_escape())
            else:
                out.append(char)

    def parse_escape(self) -> str:
        char = self.peek()
        self.pos += 1
        if char in _ESCAPES:
            return _ESCAPES[char]
        if char in ("u", "U"):
            width = 4 if char == "u" else 8
            digits = self.text[self.pos:self.pos + width]
            if len(digits) != width or not re.fullmatch(r"[0-9a-fA-F]+", digits):
                raise self.fail("invalid unicode escape")
            self.pos += width
            return chr(int(digits, 16))
        raise self.fail(f"invalid escape \\{char}")

    def parse_array(self) -> list:
        self.expect("[")
        items: list[Any] = []
        while True:
            self.skip_blank(True)
            if self.peek() == "]":
                self.pos += 1
                return items
            items.append(self.parse_value())
            self.skip_blank(True)
            if self.peek() == ",":
                self.pos += 1
            elif self.peek() != "]":
                raise self.fail("expected ',' or ']' in array")

    def parse_inline_table(self) -> dict:
        self.expect("{")
        table: dict[str, Any] = {}
        self.skip_blank(False)
        if self.peek() == "}":
            self.pos += 1
            return table
        while True:
            path = self.parse_key_path()
            self.skip_blank(False)
            self.expect("=")
            self.skip_blank(False)
            self.assign(table, path, self.parse_value())
            self.skip_blank(False)
            if self.peek() == ",":
                self.pos += 1
                continue
            self.expect("}")
            return table

    # -- structure -----------------------------------------------------
    def descend(self, base: dict, path: tuple[str, ...]) -> dict:
        node = base
        for part in path:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise self.fail(f"key {part!r} is not a table")
        return node

    def assign(self, table: dict, path: tuple[str, ...], value: Any) -> None:
        parent = self.descend(table, path[:-1])
        if path[-1] in parent:
            raise self.fail(f"duplicate key {'.'.join(path)!r}")
        parent[path[-1]] = value

    def parse_header(self) -> None:
        self.expect("[")
        if self.peek() == "[":
            raise self.fail("arrays of tables are not supported")
        path = self.parse_key_path()
        self.skip_blank(False)
        self.expect("]")
        if path in self.defined_tables:
            raise self.fail(f"table [{'.'.join(path)}] defined twice")
        self.defined_tables.add(path)
        self.current = self.descend(self.root, path)

    def end_of_line(self) -> None:
        self.skip_blank(False)
        if self.peek() not in ("", "\n", "\r"):
            raise self.fail("unexpected text after value")

    def parse(self) -> dict[str, Any]:
        while True:
            self.skip_blank(True)
            if self.pos >= len(self.text):
                return self.root
            if self.peek() == "[":
                self.parse_header()
            else:
                path = self.parse_key_path()
                self.skip_blank(False)
                self.expect("=")
                self.skip_blank(False)
                self.assign(self.current, path, self.parse_value())
            self.end_of_line()


def loads(text: str) -> dict[str, Any]:
    """Parse TOML ``text`` (supported subset) into nested dicts."""
    return _Parser(text).parse()
