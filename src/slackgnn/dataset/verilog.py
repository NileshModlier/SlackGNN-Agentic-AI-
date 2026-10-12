"""Structural Verilog reader for a post-route gate netlist.

The reader accepts modules, buses, escaped names, and concatenations.
It does not evaluate behavioral Verilog.
"""

from __future__ import annotations

from dataclasses import dataclass


class VerilogError(Exception):
    """The netlist is not structural Verilog this reader can flatten."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int


@dataclass(frozen=True)
class Bus:
    """A scalar or a vector, with bit names ordered from MSB to LSB."""

    name: str
    msb: int | None
    lsb: int | None
    direction: str | None

    @property
    def bits(self) -> tuple[str, ...]:
        if self.msb is None or self.lsb is None:
            return (self.name,)
        step = 1 if self.lsb >= self.msb else -1
        return tuple(f"{self.name}[{index}]" for index in range(self.msb, self.lsb + step, step))


@dataclass(frozen=True)
class NameExpr:
    name: str


@dataclass(frozen=True)
class BitExpr:
    name: str
    index: int


@dataclass(frozen=True)
class SliceExpr:
    name: str
    msb: int
    lsb: int


@dataclass(frozen=True)
class ConcatExpr:
    parts: tuple[NameExpr | BitExpr | SliceExpr, ...]


Expr = NameExpr | BitExpr | SliceExpr | ConcatExpr


@dataclass(frozen=True)
class Connection:
    pin: str
    expr: Expr
    line: int


@dataclass(frozen=True)
class Instance:
    cell: str
    name: str
    pins: tuple[Connection, ...]
    line: int


@dataclass(frozen=True)
class Module:
    name: str
    ports: dict[str, Bus]
    wires: dict[str, Bus]
    instances: tuple[Instance, ...]

    def symbol(self, name: str) -> Bus | None:
        if name in self.ports:
            return self.ports[name]
        return self.wires.get(name)


def parse_netlist(text: str) -> dict[str, Module]:
    """Return the modules declared in ``text``, keyed by module name."""
    if not text.strip():
        raise VerilogError("Verilog netlist is empty")
    parser = _Parser(tokenize(text.lstrip("\ufeff")))
    modules = parser.parse_modules()
    if not modules:
        raise VerilogError("Verilog netlist does not declare a module")
    return modules


def expand_expr(expr: Expr, module: Module) -> tuple[str, ...]:
    """Expand a connection into scalar net names, MSB first."""
    if isinstance(expr, ConcatExpr):
        bits: list[str] = []
        for part in expr.parts:
            bits.extend(expand_expr(part, module))
        return tuple(bits)
    if isinstance(expr, NameExpr):
        symbol = module.symbol(expr.name)
        if symbol is None:
            return (expr.name,)
        return symbol.bits
    if isinstance(expr, BitExpr):
        return (f"{expr.name}[{expr.index}]",)
    step = 1 if expr.lsb >= expr.msb else -1
    return tuple(f"{expr.name}[{index}]" for index in range(expr.msb, expr.lsb + step, step))


def tokenize(text: str) -> list[Token]:
    """Split structural Verilog into identifiers, numbers, and punctuation."""
    tokens: list[Token] = []
    index = 0
    line = 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        if char.isspace():
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            index += 2
            while index + 1 < length and not (text[index] == "*" and text[index + 1] == "/"):
                if text[index] == "\n":
                    line += 1
                index += 1
            if index + 1 >= length:
                raise VerilogError(f"Unclosed block comment starting near line {line}")
            index += 2
            continue
        if char == "\\":
            index += 1
            start = index
            while index < length and not text[index].isspace():
                index += 1
            if start == index:
                raise VerilogError(f"Empty escaped identifier on line {line}")
            tokens.append(Token("ident", text[start:index], line))
            if index < length and text[index].isspace() and text[index] != "\n":
                index += 1
            continue
        if char.isdigit():
            start = index
            while index < length and text[index].isdigit():
                index += 1
            if index < length and text[index] == "'":
                index += 1
                if index < length and text[index] in "bBhHdDoO":
                    index += 1
                while index < length and text[index] in "01xXzZ_":
                    index += 1
            tokens.append(Token("number", text[start:index], line))
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (text[index].isalnum() or text[index] in "_$"):
                index += 1
            tokens.append(Token("ident", text[start:index], line))
            continue
        if char in "()[]{},.;:":
            tokens.append(Token("punct", char, line))
            index += 1
            continue
        raise VerilogError(f"Unexpected character {char!r} on line {line}")
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse_modules(self) -> dict[str, Module]:
        modules: dict[str, Module] = {}
        while self._peek() is not None:
            module = self._parse_module()
            if module.name in modules:
                raise VerilogError(f"Module {module.name} is declared more than once")
            modules[module.name] = module
        return modules

    def _parse_module(self) -> Module:
        self._expect_ident("module")
        name = self._expect_kind("ident").value
        self._expect_punct("(")
        while self._accept_punct(")") is None:
            self._expect_kind("ident")
            if self._accept_punct(")") is not None:
                break
            self._expect_punct(",")
        self._expect_punct(";")

        ports: dict[str, Bus] = {}
        wires: dict[str, Bus] = {}
        instances: list[Instance] = []
        while True:
            token = self._peek()
            if token is None:
                raise VerilogError(f"Module {name} ends before endmodule")
            if token.kind == "ident" and token.value == "endmodule":
                self._index += 1
                self._accept_punct(";")
                break
            if token.kind == "ident" and token.value in {"input", "output", "inout", "wire"}:
                self._parse_declaration(ports, wires)
                continue
            instances.append(self._parse_instance())
        return Module(name=name, ports=ports, wires=wires, instances=tuple(instances))

    def _parse_declaration(self, ports: dict[str, Bus], wires: dict[str, Bus]) -> None:
        keyword = self._expect_kind("ident")
        if keyword.value == "inout":
            raise VerilogError(f"Inout ports are not supported (line {keyword.line})")
        span = self._parse_range()
        while True:
            name = self._expect_kind("ident").value
            bus = Bus(
                name=name,
                msb=None if span is None else span[0],
                lsb=None if span is None else span[1],
                direction=None if keyword.value == "wire" else keyword.value,
            )
            target = wires if keyword.value == "wire" else ports
            if name in ports or name in wires:
                raise VerilogError(f"Name {name} is declared more than once (line {keyword.line})")
            target[name] = bus
            if self._accept_punct(";") is not None:
                return
            self._expect_punct(",")

    def _parse_instance(self) -> Instance:
        cell = self._expect_kind("ident")
        name = self._expect_kind("ident").value
        self._expect_punct("(")
        pins: list[Connection] = []
        if self._accept_punct(")") is None:
            while True:
                self._expect_punct(".")
                pin = self._expect_kind("ident").value
                self._expect_punct("(")
                if self._accept_punct(")") is not None:
                    expr: Expr = ConcatExpr(())
                else:
                    expr = self._parse_expr()
                    self._expect_punct(")")
                pins.append(Connection(pin=pin, expr=expr, line=cell.line))
                if self._accept_punct(")") is not None:
                    break
                self._expect_punct(",")
        self._expect_punct(";")
        return Instance(cell=cell.value, name=name, pins=tuple(pins), line=cell.line)

    def _parse_expr(self) -> Expr:
        if self._accept_punct("{") is not None:
            parts: list[NameExpr | BitExpr | SliceExpr] = []
            while True:
                part = self._parse_primary()
                if isinstance(part, ConcatExpr):
                    raise VerilogError("Nested concatenations are not supported")
                parts.append(part)
                if self._accept_punct("}") is not None:
                    break
                self._expect_punct(",")
            if not parts:
                raise VerilogError("Empty concatenation")
            return ConcatExpr(tuple(parts))
        return self._parse_primary()

    def _parse_primary(self) -> NameExpr | BitExpr | SliceExpr:
        name = self._expect_kind("ident").value
        if self._accept_punct("[") is None:
            return NameExpr(name)
        msb = int(self._expect_kind("number").value)
        if self._accept_punct(":") is not None:
            lsb = int(self._expect_kind("number").value)
            self._expect_punct("]")
            return SliceExpr(name, msb, lsb)
        self._expect_punct("]")
        return BitExpr(name, msb)

    def _parse_range(self) -> tuple[int, int] | None:
        if self._accept_punct("[") is None:
            return None
        msb = int(self._expect_kind("number").value)
        self._expect_punct(":")
        lsb = int(self._expect_kind("number").value)
        self._expect_punct("]")
        return msb, lsb

    def _peek(self) -> Token | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _accept_punct(self, value: str) -> Token | None:
        token = self._peek()
        if token is not None and token.kind == "punct" and token.value == value:
            self._index += 1
            return token
        return None

    def _expect_punct(self, value: str) -> Token:
        token = self._accept_punct(value)
        if token is None:
            raise VerilogError(f"Expected {value!r} {self._where()}")
        return token

    def _expect_ident(self, value: str) -> Token:
        token = self._peek()
        if token is None or token.kind != "ident" or token.value != value:
            raise VerilogError(f"Expected {value} {self._where()}")
        self._index += 1
        return token

    def _expect_kind(self, kind: str) -> Token:
        token = self._peek()
        if token is None or token.kind != kind:
            raise VerilogError(f"Expected {kind} {self._where()}")
        self._index += 1
        return token

    def _where(self) -> str:
        token = self._peek()
        if token is None:
            return "at end of file"
        return f"near {token.value!r} on line {token.line}"
