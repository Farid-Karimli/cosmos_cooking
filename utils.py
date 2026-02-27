
def truncate_at_first_answer(text: str, end_marker: str = "</answer>") -> str:
    """Keep only up to and including the first complete answer block to avoid repetition loops."""
    idx = text.find(end_marker)
    if idx != -1:
        return (text[: idx + len(end_marker)]).strip()
    return text.strip()


def _model_response_to_lines(response: list) -> list[str]:
    """Convert model_response (list of one string) to list of lines for readable JSON."""
    if not response or not isinstance(response[0], str):
        return response
    return response[0].split("\n")