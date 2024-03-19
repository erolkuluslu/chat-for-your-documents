
import argparse  # This module provides an easy way to create command-line interfaces.
from dataclasses import dataclass  # This module provides a decorator for adding type annotations to Python classes.
from langchain.vectorstores.chroma import Chroma  # This is a vector store implementation from LangChain.
from langchain.embeddings import OpenAIEmbeddings  # This is an embedding model from LangChain that uses OpenAI's embeddings.
from langchain.chat_models import ChatOpenAI  # This is a chat model from LangChain that uses OpenAI's language model.
from langchain.prompts import ChatPromptTemplate  # This is a prompt template from LangChain for chat models.
"""
The main steps in the code are:

Parse the command-line arguments to get the query text.
Prepare the Chroma vector store with OpenAI embeddings.
Search the vector store for the most relevant documents based on the query text.
Concatenate the content of the relevant documents into a context string.
Format the prompt with the context and query text using a prompt template.
Use the ChatOpenAI model to generate a response based on the prompt.
Format the response with the sources of the relevant documents and print it.
"""


CHROMA_PATH = "chroma"  # The path where the Chroma vector store will be stored.

# This is the prompt template that will be used to format the context and question for the chat model.
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create a command-line interface (CLI) parser and add an argument for the query text.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the Chroma vector store with the OpenAI embeddings.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the vector store for the most relevant documents based on the query text.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # Concatenate the content of the relevant documents into a single context string.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the prompt with the context and query text using the prompt template.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Use the ChatOpenAI model to generate a response based on the prompt.
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # Get the sources of the relevant documents and format the response with the sources.
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
