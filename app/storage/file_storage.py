from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from botocore.exceptions import ClientError

from app.storage.b2_client import (
    get_b2_client,
    get_b2_bucket_name,
    get_b2_storage_prefix,
)


class FileStorage:
    """
    Backblaze B2 file storage interface.

    Stores original uploaded documents in a private B2 bucket.
    """

    def __init__(self):
        self.client = get_b2_client()
        self.bucket = get_b2_bucket_name()
        self.prefix = get_b2_storage_prefix()

    def build_document_version_key(
        self,
        document_id: UUID,
        version_id: UUID,
        file_extension: str = ".pdf",
    ) -> str:
        """Generate a shared-corpus object key for a document version."""

        if not isinstance(document_id, UUID):
            raise TypeError("document_id must be a UUID.")
        if not isinstance(version_id, UUID):
            raise TypeError("version_id must be a UUID.")
        if not isinstance(file_extension, str):
            raise TypeError("file_extension must be a string.")

        extension = file_extension.strip().lower()
        if extension not in {".pdf", ".txt"}:
            raise ValueError("file_extension must be .pdf or .txt.")

        return (
            f"{self.prefix}"
            f"documents/{document_id}/"
            f"versions/{version_id}{extension}"
        )

    def upload_file(
        self,
        local_path: str | Path,
        storage_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        """
        Upload a local file to B2.

        Returns basic information about the uploaded object.
        """

        source = Path(local_path).expanduser()

        if not source.exists():
            raise FileNotFoundError(f"File does not exist: {source}")

        if not source.is_file():
            raise ValueError(f"Not a file: {source}")

        if not storage_key or not storage_key.strip():
            raise ValueError("Storage key cannot be empty.")

        storage_key = storage_key.lstrip("/")

        if not content_type:
            content_type, _ = mimetypes.guess_type(source.name)

        extra_args = {
            "ContentType": content_type or "application/octet-stream",
        }

        if metadata:
            extra_args["Metadata"] = metadata

        self.client.upload_file(
            Filename=str(source),
            Bucket=self.bucket,
            Key=storage_key,
            ExtraArgs=extra_args,
        )

        # Verify that the uploaded object exists and has the
        # same size as the local file.
        uploaded = self.client.head_object(
            Bucket=self.bucket,
            Key=storage_key,
        )

        local_size = source.stat().st_size
        uploaded_size = uploaded["ContentLength"]

        if local_size != uploaded_size:
            raise IOError(
                "Upload verification failed: "
                f"local size={local_size}, "
                f"remote size={uploaded_size}"
            )

        return {
            "storage_key": storage_key,
            "bucket": self.bucket,
            "size": uploaded_size,
            "content_type": uploaded.get("ContentType"),
        }

    def download_file(
        self,
        storage_key: str,
        destination: str | Path,
    ) -> Path:
        """
        Download an object from B2 to a local path.
        """

        if not storage_key:
            raise ValueError("Storage key cannot be empty.")

        destination = Path(destination).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        self.client.download_file(
            self.bucket,
            storage_key,
            str(destination),
        )

        return destination

    def download_to_fileobj(
        self,
        storage_key: str,
        file_object: BinaryIO,
    ) -> None:
        """
        Download an object directly into a binary file-like object.
        """

        self.client.download_fileobj(
            self.bucket,
            storage_key,
            file_object,
        )

    def exists(self, storage_key: str) -> bool:
        """
        Check whether an object exists in B2.

        Missing objects return False.
        Permission and connection errors are propagated.
        """

        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=storage_key,
            )
            return True

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")

            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False

            raise

    def get_metadata(self, storage_key: str) -> dict | None:
        """
        Retrieve object metadata without downloading its contents.
        """

        try:
            response = self.client.head_object(
                Bucket=self.bucket,
                Key=storage_key,
            )

            return {
                "size": response["ContentLength"],
                "content_type": response.get("ContentType"),
                "last_modified": response.get("LastModified"),
                "metadata": response.get("Metadata", {}),
            }

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")

            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return None

            raise

    def delete_file(self, storage_key: str) -> None:
        """
        Delete an object from B2.

        This method deliberately does not ask for interactive
        confirmation. The calling service controls deletion policy.
        """

        if not storage_key:
            raise ValueError("Storage key cannot be empty.")

        self.client.delete_object(
            Bucket=self.bucket,
            Key=storage_key,
        )

    def generate_download_url(
        self,
        storage_key: str,
        expiration_seconds: int = 900,
    ) -> str:
        """
        Generate a temporary presigned download URL.

        Default expiration: 15 minutes.
        """

        if expiration_seconds <= 0:
            raise ValueError("Expiration must be greater than zero.")

        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": self.bucket,
                "Key": storage_key,
            },
            ExpiresIn=expiration_seconds,
        )