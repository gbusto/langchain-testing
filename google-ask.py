from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, LLMRequestsChain

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Question to ask google")
    args = parser.parse_args()

    template = """Between >>> and <<< are the raw search result text from google.
    Extract the answer to the question '{query}' or say "not found" if the information is not contained.
    Use the format
    Extracted:<answer or "not found">
    >>> {requests_result} <<<
    Extracted:"""

    PROMPT = PromptTemplate(
        input_variables=["query", "requests_result"],
        template=template,
    )

    llm = OpenAI(temperature=0.1)

    llmChain = LLMChain(llm=llm, prompt=PROMPT)
    requestsChain = LLMRequestsChain(llm_chain=llmChain)

    question = args.question
    inputs = {
        "query": question,
        "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
    }

    results = requestsChain(inputs)

    print(results["output"])