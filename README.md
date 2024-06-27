# Advanced ChatBot Application
==============================


## Setup

    1. Clone the repository
    2. Setup a virtual environment/pyenv/pipenv with python==3.10.12
    3. Package install :

        ```bash
        cd pkg
        pip install -e .
        ```
    4. Very important:

        copy .env. to .env.local and edit DATA_PATH and OPENAI_API_KEY, once you receive it.
    


## Rag service

RagService is an object (Singleton) 

### Basic Usage

```python
from advanced_chatbot.services.rag_services import RagService
doc_path = "~/home/me/ai_chat/pkg/advanced_chatbot/data/livre_ux_digital_design_lab_1.pdf"

#1. Loading the document (Indexing/Ingestion)
index_id, _ = RagService.create_vector_store_index(doc_path)

#2. Querying the document
query = "What is the role of a UX designer?"

output, sources = RagService.query(query, [], [index_id])


#3. Displaying the output
for x in output:
    print(x)


#4. Displaying the sources
for x in sources:
    print(x.get_text())


#5. Get the language (fr, en of an index)
lang = RagService.detect_document_language(index_id)
```


## MOCK LLM and EMBEDDING

until you get the OpenAI key, you will use MockLLM and MockEmbedding, which typically returns the same question as the answer.

To enable mock models, in config.py

```python [config.py]
USE_MOCK_MODELS = True
```


## Test DATA.
The code comes with five documents in the data folder. You can use them to test the application.(4 in French and 1 in English)


