import re


def _parse_braces(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("No matching closing brace found.")


def _expand_command(latex: str, command: str, definition: str) -> str:
    name_pattern = re.compile(re.escape(command) + r"\{")

    while match := name_pattern.search(latex):
        start = match.start()
        end = match.end() - 1

        args = []
        while latex[end] == "{":
            arg_start = end + 1
            arg_end = _parse_braces(latex, end)
            args.append(latex[arg_start:arg_end])
            end = arg_end + 1
        expanded = definition
        for idx, arg in enumerate(args, 1):
            expanded = expanded.replace(f"#{idx}", arg)
        latex = latex.replace(latex[start:end], expanded)

    return latex


def expand_newcommands(latex: str) -> str:
    newcommand_pattern = re.compile(r"\\newcommand*")

    while match := newcommand_pattern.search(latex):
        i = match.end()

        if latex[i] != "{":
            raise ValueError(f"Expected '{{' for command name at position {i}")
        command_start = i + 1
        command_end = latex.find("}", command_start)
        if command_end == -1:
            raise ValueError("Unclosed command name brace")
        command = latex[command_start:command_end]
        i = command_end + 1

        if latex[i] == "[":
            i = latex.find("]", i) + 1

        if latex[i] != "{":
            raise ValueError("Expected '{' for command definition")
        definition_start = i + 1
        definition_end = _parse_braces(latex, i)
        definition = latex[definition_start:definition_end]

        latex = latex.replace(latex[match.start() : definition_end + 1], "")
        latex = _expand_command(latex, command, definition)

    return latex
