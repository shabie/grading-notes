import csv
import os
from typing import Any, Literal

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field


class GradingNote(BaseModel):
    """Represents the grading criteria for a question."""
    question: str = Field(..., description="The question to be evaluated.")
    grading_note: str = Field(..., description="The desired attributes of a good answer.")


class Evaluation(BaseModel):
    """Represents the evaluation of an answer against a grading note."""
    reasoning: str = Field(..., description="The reasoning for the evaluation verdict.")  # said to improve performance of constrained generation
    verdict: Literal ["Good", "Bad"] = Field(..., description="The evaluation verdict, either 'Good' or 'Bad'")


class Judge(BaseModel):
    """Represents the client for the instructor library."""
    client: Any = Field(..., description="The client for the instructor library.")
    properties: dict[str, Any] = Field(..., description="The properties of the instructor client.")


def evaluate(judge: Judge, grading_note: GradingNote, answer: str) -> bool:
    """Evaluates an answer against a GradingNote."""
    response = judge.client.messages.create(
        model=judge.properties["model"],
        max_tokens=4000,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates AI generated answers against grading notes written by humans.",
            },
            {
                "role": "user",
                "content": f"Question by the user: {grading_note.question}\nAnswer by the user: {answer}\nGrading Note for the question: {grading_note.grading_note}",
            }
        ],
        response_model=Evaluation,
    )
    return response.verdict == "Good"


def get_judge(provider: Literal["anthropic", "openai"] = "anthropic", model: str = None, **kwargs) -> Judge:
    """Creates a Judge client for the instructor library."""
    if provider == "anthropic":
        base_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        default_model = "claude-3-5-sonnet-20240620"
        model = model or default_model
        client = instructor.from_anthropic(base_client)
    elif provider == "openai":
        base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        default_model = "gpt-4o-2024-08-06"
        model = model or default_model
        client = instructor.from_openai(base_client)
    else:
        raise ValueError("Invalid provider. Choose 'anthropic' or 'openai'.")
    return Judge(client=client, properties={"model": model, **kwargs})



def evaluate_from_csv(judge: Judge, csv_file: str, answer_func=None) -> dict[str, bool]:
    """Evaluates questions and grading notes from a CSV file."""
    results = {}
    with open(csv_file) as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames

        if not set(headers).issuperset({'question', 'grading_note'}):
            raise ValueError("CSV file must have 'question' and 'grading_note' columns")

        has_answers = 'answer' in headers
        if not has_answers and answer_func is None:
            raise ValueError("CSV file does not contain 'answer' column and no answer_func provided")

        for row in reader:
            grading_note = GradingNote(question=row['question'], grading_note=row['grading_note'])

            if has_answers:
                answer = row['answer']
            elif answer_func:
                answer = answer_func(grading_note.question)
            else:
                raise ValueError(f"No answer provided for question: {grading_note.question}")

            results[grading_note.question] = evaluate(judge, grading_note, answer)

    return results


if __name__ == "__main__":
    # Sanity check script
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file

    # Create a judge
    # judge = get_judge(provider="anthropic", model="claude-3-5-sonnet-20240620")
    judge = get_judge(provider="openai", model="gpt-4o-2024-08-06")

    # Create a sample grading note
    sample_note = GradingNote(
        question="What is the capital of France?",
        grading_note="The answer should be 'Paris'. Accept variations like 'paris' (case-insensitive)."
    )

    # Test cases
    test_cases = [
        ("Paris", True),
        ("paris", True),
        ("London", False),
        ("New York", False),
    ]

    print("Running sanity check...")
    for answer, expected in test_cases:
        result = evaluate(judge, sample_note, answer)
        print(f"Answer: '{answer}' - Expected: {expected}, Got: {result}")
        assert result == expected, f"Mismatch for answer '{answer}'"

    print("Sanity check completed successfully!")

    # Optional: Test CSV functionality
    import tempfile

    csv_content = """question,grading_note,answer
What is the capital of Japan?,The answer should be 'Tokyo'.,Tokyo
What is the largest planet in our solar system?,The answer should be 'Jupiter'.,Saturn"""

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_csv:
        temp_csv.write(csv_content)
        temp_csv_path = temp_csv.name

    print("\nTesting CSV functionality...")
    csv_results = evaluate_from_csv(judge, temp_csv_path)
    print("CSV Results:", csv_results)

    assert csv_results == {
        "What is the capital of Japan?": True,
        "What is the largest planet in our solar system?": False
    }, "CSV results do not match expected output"

    print("CSV functionality test completed successfully!")
