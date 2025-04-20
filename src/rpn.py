from functools import cached_property
from typing import Callable


def unary_lambda(symbol: str) -> Callable[[str], str]:
    return lambda a: f"({a}){symbol}"


class RegexRPN:
    letters = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    symbols = "%#@~'\"<>/,&:;№!{}"

    quantifiers: set[str] = {"*", "+", "?"}

    unary_regex: dict[str, Callable[[str], str]] = {
        **{symbol: unary_lambda(symbol) for symbol in quantifiers},
        **{
            symbol: unary_lambda(f"{symbol}?") for symbol in quantifiers
        },  # greedy versions
    }
    binary_regex: dict[str, Callable[[str, str], str]] = {
        "concat": lambda a, b: a + b,  # concatenation
        "|": lambda a, b: f"{a}|{b}",
    }
    many_to_one_regex: dict[str, Callable[[list[str]], str]] = {
        "[]": lambda a: "[" + "".join([f"({x})" for x in a]) + "]",  # set
        "^[]": lambda a: "[^" + "".join([f"({x})" for x in a]) + "]",  # not in set
        "concat_all": "".join,
    }

    metacharacters_regex: set[str] = {
        ".",
        "^",
        "$",
    }

    special_sequences: set[str] = {r"\d", r"\s", r"\w"}

    operands_regex: set[str] = {
        *metacharacters_regex,
        *[rf"\{s}" for s in metacharacters_regex],
        *letters,
        *letters.upper(),
        *symbols,
        *digits,
        *[rf"\{s}" for s in quantifiers],
        *special_sequences,
        r"\\",
    }

    def to_infix(self, expression: list[str]) -> str:
        if len(expression) == 0:
            return ""

        if expression[-1] != "concat_all":
            expression.append(
                "concat_all"
            )  # Changes nothing but helps to produce stack of size 1

        stack = []

        for token in expression:
            if token in self.many_to_one_regex:
                stack = [self.many_to_one_regex[token](list(reversed(stack)))]
                continue

            if token in self.binary_regex:
                operand2 = stack.pop()
                operand1 = stack.pop()
                stack.append(self.binary_regex[token](operand1, operand2))
                continue
            if token in self.unary_regex:
                operand = stack.pop()
                stack.append(f"({token}{operand})")
                continue
            if token in self.operands_regex:
                stack.append(token)
                continue
            raise RuntimeError(f"Operand '{token}' is unknown")

        return stack[0]

    @cached_property
    def available_tokens(self) -> list[str]:
        return list(
            set().union(
                self.operands_regex,
                set(self.binary_regex.keys()),
                set(self.unary_regex.keys()),
                set(self.many_to_one_regex.keys()),
            )
        )

    def __len__(self):
        return len(self.available_tokens)
