from typing import Any
from schemas import StepInputRef
class MissingInput(KeyError):
    pass

class Board: 
    def __init__(self) -> None:
        self._d: dict[tuple[str, str], Any] = {}

    def put(self, step_id: str, output: str, value: Any) -> None:
        self._d[(step_id, output)] = value
    def get(self, ref: StepInputRef) -> Any:
        try:
            return self._d[(ref.step_id, ref.output)]
        except KeyError:
            raise MissingInput(f"{ref.step_id}.{ref.output}") from None