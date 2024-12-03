from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from typing import List
from tqdm import tqdm

# Structured output for zero-shot classification
class Prediction(BaseModel):
    prediction: int

# Prompt template for zero-shot classification
ZeroShotPrompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a financial sentiment analysis assistant. Classify the text into one of the following categories:
- Negative (0)
- Neutral (1)
- Positive (2)

Text: {text}

Your response must be a JSON object with the following structure:
{{"prediction": category_number}}

For example: {{"prediction": 1}}
"""
)

def get_predictions(dataset: List[str], model_name: str, batch_size: int = 5) -> List[int]:
    """
    Classify a dataset of texts into sentiment categories using a structured output model.

    Args:
        dataset (List[str]): A list of texts to classify.
        model_name (str): The OpenAI model name (e.g., "gpt-3.5-turbo" or "gpt-4").
        batch_size (int): Number of texts to process per batch.

    Returns:
        List[int]: A list of predicted sentiment categories.
    """
    # Initialize the ChatOpenAI model with structured output
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(Prediction)

    predictions = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
        batch = dataset[i:i + batch_size]
        for text in batch:
            # Format the prompt for the current text
            prompt = ZeroShotPrompt.format(text=text)
            # Generate the structured prediction
            try:
                result = llm.invoke(prompt)
                predictions.append(result.prediction)
            except Exception as e:
                # Handle errors or invalid outputs
                print(f"Error processing text: {text}\nError: {e}")
                predictions.append(-1)  
    return predictions