from unittest.mock import Mock, patch

import pytest

from grading_notes import GradingNote, Judge, evaluate_from_csv


@pytest.fixture
def mock_client():
    mock = Mock(spec=Judge)
    mock.properties = {"model": "gpt-4o-2024-08-06"}
    return mock


@pytest.fixture
def mock_evaluate():
    with patch('grading_notes.main.evaluate') as mock:
        yield mock


def test_evaluate_from_csv__base_case(mock_client, mock_evaluate, tmp_path):
    # Create a temporary CSV file
    csv_content = """question,grading_note,answer
What is the capital of Pakistan?,The answer should be 'Islamabad'.,Islamabad
What is the largest planet in our solar system?,The answer should be 'Jupiter'.,Saturn
How many continents are there in the world?,The answer should be '7'.,Seven"""

    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    # Set up mock responses
    mock_evaluate.side_effect = [True, False, True]

    results = evaluate_from_csv(mock_client, str(csv_file))
    assert results == {
        "What is the capital of Pakistan?": True,
        "What is the largest planet in our solar system?": False,
        "How many continents are there in the world?": True,
    }

    # Verify that evaluate was called with correct arguments
    assert mock_evaluate.call_count == 3
    mock_evaluate.assert_any_call(mock_client, GradingNote(question='What is the capital of Pakistan?', grading_note="The answer should be 'Islamabad'."), 'Islamabad')
    mock_evaluate.assert_any_call(mock_client, GradingNote(question='What is the largest planet in our solar system?', grading_note="The answer should be 'Jupiter'."), 'Saturn')
    mock_evaluate.assert_any_call(mock_client, GradingNote(question='How many continents are there in the world?', grading_note="The answer should be '7'."), 'Seven')


def test_evaluate_from_csv__wrong_column_name_raises_error(tmp_path):
    # Create a temporary CSV file
    csv_content = """question,wrong_column"""

    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    # Check that the function raises a ValueError due to wrong column name
    with pytest.raises(ValueError):
        evaluate_from_csv(mock_client, str(csv_file))
