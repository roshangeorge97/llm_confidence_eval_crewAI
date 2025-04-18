import pytest
from unittest.mock import patch, MagicMock
from calm_core import CALMEngine, Config, GeminiClient, DataLoader

# Example of mocking the GeminiClient generate_content method to avoid API calls in tests
@pytest.fixture
def mock_gemini_client():
    # Mocking the GeminiClient to avoid actual API calls
    with patch.object(GeminiClient, 'generate_content', return_value="42") as mock_generate:
        yield mock_generate

@pytest.fixture
def calm_engine(mock_gemini_client):
    config = Config()
    engine = CALMEngine(config)
    engine.client = mock_gemini_client  # Mock the Gemini client in the engine
    return engine

# Test for 'process_item' method
def test_process_item(calm_engine):
    # Example input data
    item = {
        'question': 'What is 2 + 2?',
        'answer': '4',
        'type': 'math',
    }
    idx = 0
    dataset_name = 'math'
    
    # Run the process_item method
    result = calm_engine.process_item(item, idx, dataset_name)
    
    # Check that the result contains the expected structure
    assert 'question' in result
    assert 'initial_response' in result
    assert 'final_response' in result
    assert 'confidence' in result
    assert 'accuracy' in result
    assert 'ground_truth' in result
    assert result['accuracy'] is not None

# Test for 'calculate_confidence' method
def test_calculate_confidence(calm_engine):
    initial = "42"
    refined = "42"
    ground_truth = "42"
    
    confidence = calm_engine.calculate_confidence(initial, refined, ground_truth)
    
    # Check if confidence is in a reasonable range
    assert 0 <= confidence <= 1

# Test for 'evaluate_answer' method (math question)
def test_evaluate_answer_math(calm_engine):
    response = "42"
    ground_truth = "42"
    task_type = "math"
    
    is_correct = calm_engine.evaluate_answer(response, ground_truth, task_type)
    
    assert is_correct is True

# Test for 'evaluate_answer' method (commonsense question)
def test_evaluate_answer_commonsense(calm_engine):
    response = "plausible"
    ground_truth = "plausible"
    task_type = "commonsense"
    
    is_correct = calm_engine.evaluate_answer(response, ground_truth, task_type)
    
    assert is_correct is True

# Test for 'normalize_response' method
def test_normalize_response(calm_engine):
    text = " ##120## "
    
    normalized = calm_engine.normalize_response(text)
    
    assert normalized == "120"

# Test for dataset loading
def test_load_dataset(calm_engine):
    dataset_name = 'GSM8K'
    dataset = calm_engine.loader.load_dataset(dataset_name)
    
    assert len(dataset) > 0  # Ensure the dataset is loaded and not empty

# Test for invalid dataset loading
def test_load_invalid_dataset(calm_engine):
    with pytest.raises(ValueError):
        calm_engine.loader.load_dataset("InvalidDataset")

# Test for 'extract_answer' method with boxed answers
def test_extract_answer(calm_engine):
    response = "\\boxed{42}"
    
    extracted_answer = calm_engine.extract_answer(response)
    
    assert extracted_answer == "42"

