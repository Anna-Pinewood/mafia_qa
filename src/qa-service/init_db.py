"""Creates vector db from document, if there is no VDB already."""

from logging import getLogger
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from script_utils import get_kwargs, get_logger


if __name__ == "__main__":
    kwargs = get_kwargs().parse_args()
    LOGGER = get_logger(level=int(kwargs.logger_level))

    ROOT_DIR = Path(__file__).parent.parent.parent
    PATH_TO_DOCUMENTS = ROOT_DIR / "documents"
    PATH_TO_INDEX = ROOT_DIR / "faiss-index"

    LOGGER.info("Embedding model is loading...")
    embeddings_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    documents_paths = list(PATH_TO_DOCUMENTS.glob("*.docx"))
    LOGGER.info("%s documents will be loaded", str(len(documents_paths)))
    doc_list = []
    for document_path in tqdm(documents_paths):
        loader = Docx2txtLoader(str(document_path))
        data = loader.load()
        doc_list.extend(data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc_list)

    LOGGER.info("Filling up FAISS db...")
    vectorstore = FAISS.from_documents(documents=splits,
                                       embedding=embeddings)
    vectorstore.save_local(PATH_TO_INDEX)
    LOGGER.info("FAISS index was saved to %s", str(PATH_TO_INDEX))
