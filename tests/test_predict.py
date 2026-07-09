import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.components.data_transformation import DataTransformation

def test_clean_text_basic():
    dt = DataTransformation()
    result = dt.clean_text("Hello, World! 123")
    assert isinstance(result, str)
    assert len(result) > 0

def test_clean_text_remove_numbers():
    dt = DataTransformation()
    result = dt.clean_text("payment 12345 charge")
    assert "12345" not in result

def test_clean_text_lowercase():
    dt = DataTransformation()
    result = dt.clean_text("HELLO WORLD")
    assert result == "hello world"

def test_clean_text_remove_stopwords():
    dt = DataTransformation()
    result = dt.clean_text("thank you for your support")
    assert "thank" not in result
    assert "support" not in result




