from langchain.docstore.document import Document
import re
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
import tempfile

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)


def clean_string(text: str) -> str:
    """
    This function takes in a string and performs a series of text cleaning operations.

    Args:
        text (str): The text to be cleaned. This is expected to be a string.

    Returns:
        cleaned_text (str): The cleaned text after all the cleaning operations have been performed.
    """
    # Replacement of newline characters:
    text = text.replace("\n", " ")

    # Stripping and reducing multiple spaces to single:
    cleaned_text = re.sub(r"\s+", " ", text.strip())

    # Removing backslashes:
    cleaned_text = cleaned_text.replace("\\", " ")

    # Replacing hash characters:
    cleaned_text = cleaned_text.replace("#", " ")

    # Eliminating consecutive non-alphanumeric characters:
    # This regex identifies consecutive non-alphanumeric characters (i.e., not a word character [a-zA-Z0-9_] and not a whitespace) in the string
    # and replaces each group of such characters with a single occurrence of that character.
    # For example, "!!! hello !!!" would become "! hello !".
    cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)

    return cleaned_text


def get_uploaded_documents_from_pdf(
    file,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    mode="paged",
    type_split="simple",
    clean=True,
    **kwargs,
) -> list[Document]:
    name = file.name
    with tempfile.NamedTemporaryFile(delete=False) as temp_f:
        temp_f.write(file.getvalue())
        loader = UnstructuredPDFLoader(temp_f.name, mode=mode, **kwargs)
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = loader.load()
        if clean:
            for doc in docs:
                doc.page_content = clean_string(doc.page_content)
                doc.metadata["file"] = name
        docs = text_splitter.split_documents(docs)
        temp_f.close()

    return docs


def get_local_documents_from_pdf(
    file,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    mode="paged",
    type_split="simple",
    clean=True,
    **kwargs,
) -> list[Document]:
    loader = UnstructuredPDFLoader(file, mode=mode, **kwargs)
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if type_split == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    docs = loader.load()
    if clean:
        for doc in docs:
            doc.page_content = clean_string(doc.page_content)
            doc.metadata["file"] = file.split("/")[-1]
    docs = text_splitter.split_documents(docs)

    return docs


def load_file(path):
    loader = UnstructuredFileLoader(path)
    doc = loader.load()
    text = " ".join([d.page_content for d in doc])

    return (text, doc)
