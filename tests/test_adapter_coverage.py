from execution.adapters import ADAPTERS
from execution.records import OUTPUT_MODELS

def test_every_adapter_has_a_typed_output_model():
    assert set(ADAPTERS) == set(OUTPUT_MODELS)