"""Google Cloud Translate provider implementation."""
from __future__ import annotations

from typing import Optional

from synthetictext.providers.base import BaseTranslationProvider
from synthetictext.utils import retry_with_backoff


class GoogleTranslateProvider(BaseTranslationProvider):
    """Translation provider backed by Google Cloud Translation API v3.

    Requires ``pip install synthetictext[google-translate]``.

    Args:
        project_id: Google Cloud project ID.
        credentials_path: Path to a service-account JSON key file.
            If *None*, uses Application Default Credentials.
    """

    def __init__(
        self,
        project_id: str,
        credentials_path: Optional[str] = None,
    ) -> None:
        try:
            from google.cloud import translate as gc_translate
        except ImportError as exc:
            raise ImportError(
                "The google-cloud-translate package is required. "
                "Install it with: pip install synthetictext[google-translate]"
            ) from exc

        if credentials_path:
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(credentials_path)
            self._client = gc_translate.TranslationServiceClient(credentials=creds)
        else:
            self._client = gc_translate.TranslationServiceClient()

        self._parent = f"projects/{project_id}/locations/global"

    @retry_with_backoff(max_retries=3)
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        response = self._client.translate_text(
            request={
                "parent": self._parent,
                "contents": [text],
                "mime_type": "text/plain",
                "target_language_code": target_lang,
                "source_language_code": source_lang,
            }
        )
        return response.translations[0].translated_text
