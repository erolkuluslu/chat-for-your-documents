
# Chroma-Based Question Answering System

This repository contains a Python script that implements a question-answering system using the Chroma vector store and OpenAI's language models. The system allows users to query a corpus of documents and receive answers based on the relevant information found in those documents.

## Features

- **Document Ingestion**: The script can ingest and process documents from a specified directory, splitting them into smaller chunks for efficient storage and retrieval.
- **Vector Embeddings**: It utilizes OpenAI's embeddings to create vector representations of the document chunks and user queries, enabling efficient similarity search.
- **Similarity Search**: The script performs similarity searches in the Chroma vector store to find the most relevant document chunks for a given user query.
- **Context Preparation**: It concatenates the content of the relevant document chunks into a single context string.
- **Prompt Generation**: The context string and the user query are formatted into a prompt using a customizable prompt template.
- **Answer Generation**: The prompt is passed to OpenAI's ChatGPT model, which generates a relevant answer based on the provided context and query.
- **Source Citation**: The script includes the sources of the relevant document chunks in the final answer.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Place your documents in the `data/books` directory (or modify the `DATA_PATH` variable to point to your document directory).
3. Run the script with `python create_database.py"`.
4. Example query can be 'python query_data.py "How does Alice meet the Mad Hatter?"'.
5. The script will generate an answer based on the relevant information found in the documents, along with the sources cited.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
