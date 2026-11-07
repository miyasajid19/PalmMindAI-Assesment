import pytest

from langchain_core.documents import Document

from app.rag.ingestion.pdf_sampler import PDFSampler


def create_documents(page_count: int) -> list[Document]:
    return [
        Document(
            page_content=f"Content of page {page_number}",
            metadata={
                "page": page_number,
                "source": "test.pdf",
            },
        )
        for page_number in range(1, page_count + 1)
    ]


def test_sample_large_document():
    documents = create_documents(13)

    sampler = PDFSampler(max_pages=5)
    sampled = sampler.sample(documents)

    assert len(sampled) == 5

    sampled_pages = [
        document.metadata["page"]
        for document in sampled
    ]

    assert sampled_pages == [1, 4, 7, 10, 13]


def test_sample_small_document():
    documents = create_documents(3)

    sampler = PDFSampler(max_pages=5)
    sampled = sampler.sample(documents)

    assert len(sampled) == 3
    assert sampled == documents


def test_sample_one_page():
    documents = create_documents(10)

    sampler = PDFSampler(max_pages=1)
    sampled = sampler.sample(documents)

    assert len(sampled) == 1
    assert sampled[0].metadata["page"] == 1


def test_sample_two_pages():
    documents = create_documents(10)

    sampler = PDFSampler(max_pages=2)
    sampled = sampler.sample(documents)

    sampled_pages = [
        document.metadata["page"]
        for document in sampled
    ]

    assert sampled_pages == [1, 10]


def test_empty_document_list():
    sampler = PDFSampler(max_pages=5)

    assert sampler.sample([]) == []


def test_sampling_is_deterministic():
    documents = create_documents(20)

    sampler = PDFSampler(max_pages=5)

    first_result = sampler.sample(documents)
    second_result = sampler.sample(documents)

    first_pages = [doc.metadata["page"] for doc in first_result]
    second_pages = [doc.metadata["page"] for doc in second_result]

    assert first_pages == second_pages


def test_sample_with_page_numbers():
    documents = create_documents(13)

    sampler = PDFSampler(max_pages=5)
    sampled = sampler.sample_with_page_numbers(documents)

    page_numbers = [page for page, _ in sampled]

    assert page_numbers == [1, 4, 7, 10, 13]


def test_sample_with_page_numbers_uses_original_index_for_missing_metadata():
    documents = create_documents(3)
    documents[1].metadata.pop("page")

    sampler = PDFSampler(max_pages=5)
    sampled = sampler.sample_with_page_numbers(documents)

    assert [page for page, _ in sampled] == [1, 2, 3]


def test_invalid_max_pages():
    with pytest.raises(ValueError):
        PDFSampler(max_pages=0)

    with pytest.raises(TypeError):
        PDFSampler(max_pages=2.5)


def test_invalid_documents():
    sampler = PDFSampler(max_pages=5)

    with pytest.raises(TypeError):
        sampler.sample(["not a document"])
