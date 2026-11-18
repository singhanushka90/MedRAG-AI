from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Medical PDF directory
PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "medical_pdfs"


def load_pdf(pdf_path: Path):
    """
    Load a single PDF and return LangChain Document objects.
    """

    loader = PyMuPDFLoader(str(pdf_path))

    documents = loader.load()

    # Add useful metadata to every page
    for document in documents:
        document.metadata["source"] = pdf_path.name
        document.metadata["document_id"] = pdf_path.stem

    return documents


def load_all_pdfs():
    """
    Load all PDF files from the medical_pdfs directory.
    """

    all_documents = []

    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIRECTORY}"
        )

    for pdf_path in pdf_files:
        documents = load_pdf(pdf_path)
        all_documents.extend(documents)

        print(
            f"Loaded {pdf_path.name}: "
            f"{len(documents)} pages"
        )

    print(f"\nTotal pages loaded: {len(all_documents)}")

    return all_documents