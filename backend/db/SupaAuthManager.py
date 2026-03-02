"""
Usage
-----
Initialise once at app start-up:

    SupaAuthManager().init(url, apiKey)

Dependency Injection:

    client = SupaAuthManager().get_client()
"""

import logging
import os
import threading

from supabase_auth import SyncGoTrueClient

logger = logging.getLogger(__name__)


class SupaAuthManager:
    """
    Singleton manager for the Supabase SyncGoTrueClient.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._client = None
            self._client_lock = threading.Lock()
            self._initialized = True
            logger.info("SupaAuthManager initialised")


    def init(self, url, apiKey):
        """
        Create the SyncGoTrueClient. Call once at app start-up.

        Subsequent calls are no-ops unless close() has been called first.
        """
        if self._client is not None:
            logger.debug("SupaAuthManager.init() called but client already exists – skipping.")
            return

        with self._client_lock:
            if self._client is not None:
                return

            logger.info("Creating SyncGoTrueClient at %s", url)
            try:
                self._client = SyncGoTrueClient(
                    url=url,
                    headers={"apiKey": apiKey},
                )
                logger.info("SyncGoTrueClient ready")
            except Exception as exc:
                logger.error("Failed to create SyncGoTrueClient: %s", exc)
                raise

    def get_client(self):
        """
        Return the shared SyncGoTrueClient.
        """
        if self._client is None:
            raise RuntimeError(
                "SupaAuthManager client has not been initialised. "
                "Ensure `SupaAuthManager().init(url, apiKey)` is called during app start-up."
            )
        return self._client

    def close(self):
        """
        Reset the client.
        A subsequent call to init() will create a fresh client.
        """
        with self._client_lock:
            self._client = None
            logger.info("SyncGoTrueClient closed and reset")