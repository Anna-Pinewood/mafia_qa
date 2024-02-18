# qa-system

My QA system based on LLM-assistant.

1. To install on gpu machine for running LLM model for answer generation.

```
poetry install --with model

```

2. To install on any machine for running client service – ask questions.

```
poetry install --with call_qa

```


## Make database

If you don't have FAISS index, you should create one.

1. Load documents to "documents" folder in your project.
2. Run `poetry run python src/qa-service/init_db.py`
