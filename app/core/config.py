import os
from dotenv import load_dotenv

load_dotenv()


def get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"true", "1", "yes", "on"}


class Settings:
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")

    OCR_ENGINE = os.getenv("OCR_ENGINE", "fastocr")
    OCR_LANGS = os.getenv("OCR_LANGS")

    RENDER_DPI = int(os.getenv("RENDER_DPI", "300"))
    MIN_NATIVE_CHARS = int(os.getenv("MIN_NATIVE_CHARS", "50"))

    OCR_ALL_PAGES = get_bool("OCR_ALL_PAGES")
    OCR_MERGE_WITH_NATIVE = get_bool("OCR_MERGE_WITH_NATIVE")

    FASTOCR_API_KEY = os.getenv("FASTOCR_API_KEY")
    FASTOCR_BASE_URL = os.getenv("FASTOCR_BASE_URL", "https://api.fastocr.org")
    FASTOCR_TIMEOUT_SECONDS = float(os.getenv("FASTOCR_TIMEOUT_SECONDS", "300"))

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_API_ENDPOINT = os.getenv("QDRANT_API_ENDPOINT")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

    JINA_API_KEY = os.getenv("JINA_API_KEY")
    JINA_API_BASE_URL = os.getenv("JINA_API_BASE_URL", "https://api.jina.ai/v1")
    JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
    JINA_EMBEDDING_DIMENSIONS = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "2048"))
    JINA_EMBEDDING_BATCH_SIZE = int(os.getenv("JINA_EMBEDDING_BATCH_SIZE", "64"))
    JINA_API_TIMEOUT_SECONDS = float(os.getenv("JINA_API_TIMEOUT_SECONDS", "90"))
    JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3.5")
    RERANKER_SCORE_THRESHOLD = float(os.getenv("RERANKER_SCORE_THRESHOLD", "0.30"))

    AIVEN_POSTGRES_URI = os.getenv("AIVEN_POSTGRES_URI")
    if AIVEN_POSTGRES_URI and AIVEN_POSTGRES_URI.startswith("postgres://"):
        AIVEN_POSTGRES_URI = AIVEN_POSTGRES_URI.replace(
            "postgres://", "postgresql+psycopg2://", 1
        )

    AIVEN_REDIS_URI = os.getenv("AIVEN_REDIS_URI")
    CONVERSATION_HISTORY_TTL_SECONDS = int(os.getenv("CONVERSATION_HISTORY_TTL_SECONDS", "1800"))
    CONVERSATION_HISTORY_MAX_TURNS = int(os.getenv("CONVERSATION_HISTORY_MAX_TURNS", "10"))

    B2_KEY_ID = os.getenv("B2_KEY_ID")
    B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
    B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")
    B2_REGION = os.getenv("B2_REGION")
    B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL")
    B2_STORAGE_PREFIX = os.getenv("B2_STORAGE_PREFIX", "voice-ai/")

    QDRANT_SUMMARY_COLLECTION_NAME = os.getenv(
        "QDRANT_SUMMARY_COLLECTION_NAME",
        f"{QDRANT_COLLECTION_NAME}_summaries" if QDRANT_COLLECTION_NAME else None,
    )



settings = Settings()
