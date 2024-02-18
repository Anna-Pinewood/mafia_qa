"""QA service to answer questions."""
import json
from ast import literal_eval
from datetime import datetime
from logging import getLogger
from pathlib import Path

import requests
import uvicorn
import yaml
from fastapi import FastAPI, Request
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from consts import (CONTEXT_PROMPT, EMBEDDINGS_MODEL_NAME, FAISS_INDEX_PATH,
                    MAX_TOKENS_ANSWER, MODEL_URL, SYSTEM_PROMPT, TEMPERATURE,
                    TOP_K_DOCUMENTS)
from script_utils import get_kwargs, get_logger

app = FastAPI(
    title=("MAFIA QA assistant api."),
    description=("Answers questions about mafia rules."),
    version="0.1.0",
    redoc_url=None,
)

LOGGER = getLogger(__name__)


async def call_model(
        query: str,
        url_out: str = MODEL_URL,
        context: str | None = None,
        system_prompt: str | None = None,
        context_prompt: str | None = None,
        max_new_tokens: int = MAX_TOKENS_ANSWER,
        temperature: float = TEMPERATURE,
) -> str:
    """Send query to LLM service.

    Parameters
    ----------
    query: str
        User question.
    context: str | None
        Context if present.
    system_prompt: str| None
        Model sustem prompt.
    context_prompt: str | None
        Prompt to add in fromt of context.
    max_new_tokens: int
        Max tokens to generate.
    temperature: float
        Model temperature.
    """
    data = {
        'query': query,
        'context': context,
        'system_prompt': system_prompt,
        'context_prompt': context_prompt,
        'max_tokens': max_new_tokens,
        'temperature': temperature,
    }

    response = requests.post(url_out, json=data, timeout=1000)

    response_dict = json.loads(response.text)
    return response_dict['answer']


@app.post('/answer')
async def answer(
        request: Request,
):
    """Collect context and call model
    to make an answer.
    Parameters
    ----------
    request : Request
        Must contain 'query' field.
    """

    ip_user = request.client.host
    LOGGER.info("Received POST request to /answer from IP: %s",
                ip_user)

    data = await request.json()
    query = data.get("query")
    context_framents = vectorstore.similarity_search(query, k=TOP_K_DOCUMENTS)
    context = ''
    for fragment in context_framents:
        fragment = fragment.page_content.replace(
            '\xa0\xa0', '').replace('\n\n', '\n')
        context += f"{fragment}\n\n"
    LOGGER.debug("Got context:\n%s", context)
    answer = await call_model(query=query,
                              context=context,
                              system_prompt=SYSTEM_PROMPT,
                              context_prompt=CONTEXT_PROMPT,
                              max_new_tokens=MAX_TOKENS_ANSWER,
                              temperature=TEMPERATURE,
                              url_out=MODEL_URL,
                              )
    return {'answer': answer}


@app.post('/load_document')
async def load_documents(
        request: Request,
):
    """Load new text to vector db
    and save updated db.

    Parameters
    ----------
    request : Request
        Must contain "documents" value
        with list of texts to upload.

    Example usage:
    ```
    documents = ['New data 1', 'New data 2']
    data = {'documents': str(documents)}
    response = requests.post(url, json=data, timeout=1000)
    ```
    """

    ip_user = request.client.host
    LOGGER.info("Received POST request to /load_document from IP: %s",
                ip_user)
    fragments_num = vectorstore.index_to_docstore_id.__len__()
    LOGGER.info("Database now consists of %s fragments.", str(fragments_num))

    data = await request.json()
    documents = literal_eval(data.get("documents"))

    documents_langchain = [Document(page_content=document)
                           for document in documents]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents_langchain)

    new_vectorstore = FAISS.from_documents(documents=splits,
                                           embedding=embeddings)
    vectorstore.merge_from(new_vectorstore)
    vectorstore.save_local(FAISS_INDEX_PATH)
    LOGGER.info("FAISS index was updated and saved to %s",
                str(FAISS_INDEX_PATH))

    fragments_num = vectorstore.index_to_docstore_id.__len__()
    LOGGER.info("Database was enriched up to %s fragments.",
                str(fragments_num))
    return {"answer": "Succesfully loaded."}


@app.get('/healthcheck')
async def healthcheck():
    """Check if service is available."""
    current_time = datetime.now()
    msg = (f"Hey, hey! "
           f"I am QA service and "
           f"I've been alive for {current_time - launch_time} now.\n")

    return {"message": msg}

if __name__ == '__main__':
    file_path = Path(__file__)
    default_config_path = file_path.parent / f"{file_path.stem}_config.yaml"
    kwargs = get_kwargs().parse_args()
    LOGGER = get_logger(level=int(kwargs.logger_level))

    with open(default_config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    LOGGER.info("Embedding model is loading...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    LOGGER.info("Loading FAISS DB...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embeddings)

    launch_time = datetime.now()

    uvicorn.run(app, host="0.0.0.0",
                port=config["api_port"], log_level="info", reload=config["reload"], use_colors=True)
