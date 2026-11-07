from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import MagicMock

import pytest

from app.services.document_matching import DocumentHashMatcher

@pytest.fixture
def repository():
    return MagicMock()


@pytest.fixture
def matcher(repository):
    return DocumentHashMatcher(repository)


def make_version(
    file_hash=None,
    content_hash=None,
    status="active",
):
    return SimpleNamespace(
        id=uuid4(),
        document_id=uuid4(),
        version_number=1,
        file_hash=file_hash,
        content_hash=content_hash,
        status=status,
        original_filename="employee_policy.pdf",
    )


def test_exact_file_hash_match(matcher, repository):

    version = make_version(
        file_hash="file_hash_123",
        content_hash="content_hash_123",
    )

    repository.get_by_hash.return_value = [version]

    result = matcher.match(
        file_hash="file_hash_123",
        content_hash="different_content_hash",
    )

    assert result.is_duplicate is True
    assert result.match_count == 1
    assert result.candidates[0].matched_by == "file_hash"


def test_exact_content_hash_match(matcher, repository):

    version = make_version(
        file_hash="different_file_hash",
        content_hash="content_hash_123",
    )

    repository.get_by_hash.return_value = [version]

    result = matcher.match(
        file_hash="different_file_hash_2",
        content_hash="content_hash_123",
    )

    assert result.is_duplicate is True
    assert result.match_count == 1
    assert result.candidates[0].matched_by == "content_hash"


def test_no_duplicate_found(matcher, repository):

    repository.get_by_hash.return_value = []

    result = matcher.match(
        file_hash="new_file_hash",
        content_hash="new_content_hash",
    )

    assert result.is_duplicate is False
    assert result.match_count == 0
    assert result.candidates == []


def test_archived_document_version_is_still_a_duplicate(
    matcher,
    repository,
):

    version = make_version(
        file_hash="archived_hash",
        status="superseded",
    )

    repository.get_by_hash.return_value = [version]

    result = matcher.match(
        file_hash="archived_hash",
        content_hash=None,
    )

    assert result.is_duplicate is True
    assert result.candidates[0].status == "superseded"


def test_both_hashes_match_same_version(
    matcher,
    repository,
):

    version = make_version(
        file_hash="same_file_hash",
        content_hash="same_content_hash",
    )

    repository.get_by_hash.return_value = [version]

    result = matcher.match(
        file_hash="same_file_hash",
        content_hash="same_content_hash",
    )

    assert result.is_duplicate is True
    assert result.match_count == 1
    assert result.candidates[0].matched_by == "file_hash"


def test_hash_is_required(matcher):

    with pytest.raises(ValueError):
        matcher.match(
            file_hash=None,
            content_hash=None,
        )