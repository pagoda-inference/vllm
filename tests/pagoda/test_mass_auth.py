# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MASS API Bearer token authentication (Issue #5).

Verifies that:
1. Bearer token is injected into Authorization header
2. No auth header when token is not configured
3. Environment variable fallback works
4. mTLS SSL context is created when certs are provided
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.pagoda.mass_client import MassApiClient, TenantConfig


def _make_defaults() -> TenantConfig:
    return TenantConfig(
        tenant_id="__default__",
        priority="normal",
        qps_limit=10.0,
        concurrent_limit=5,
        kv_cache_block_quota=100,
    )


class TestMassApiAuth:
    """Tests for MASS API authentication mechanisms."""

    @pytest.mark.asyncio
    async def test_bearer_token_injected_in_request_header(self):
        """Bearer token from config is sent as Authorization header."""
        client = MassApiClient(
            mass_api_url="http://mass:8080/api",
            defaults=_make_defaults(),
            api_key="test-secret-token",
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "priority": "high",
            "qps_limit": 50.0,
            "concurrent_limit": 20,
            "kv_cache_block_quota": 500,
        })

        mock_session = AsyncMock()
        mock_session.get = MagicMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(
            return_value=mock_resp,
        )
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_session,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client._fetch_from_mass("tenant-abc")

        # Verify Authorization header was passed
        call_args = mock_session.get.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer test-secret-token"
        assert result is not None
        assert result.priority == "high"

    @pytest.mark.asyncio
    async def test_no_token_no_auth_header(self):
        """No auth header when api_key is not configured."""
        # Clear env var to ensure no fallback
        with patch.dict(os.environ, {}, clear=True):
            client = MassApiClient(
                mass_api_url="http://mass:8080/api",
                defaults=_make_defaults(),
                api_key=None,
            )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "priority": "normal",
            "qps_limit": 10.0,
            "concurrent_limit": 5,
            "kv_cache_block_quota": 100,
        })

        mock_session = AsyncMock()
        mock_session.get = MagicMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(
            return_value=mock_resp,
        )
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_session,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client._fetch_from_mass("tenant-abc")

        call_args = mock_session.get.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_env_var_fallback_for_api_key(self):
        """PAGODA_MASS_API_KEY env var is used when api_key param is None."""
        with patch.dict(os.environ, {"PAGODA_MASS_API_KEY": "env-token-xyz"}):
            client = MassApiClient(
                mass_api_url="http://mass:8080/api",
                defaults=_make_defaults(),
                api_key=None,  # not passed directly
            )

        # The client should have picked up the env var
        assert client._api_key == "env-token-xyz"

    @pytest.mark.asyncio
    async def test_explicit_api_key_overrides_env_var(self):
        """Explicit api_key parameter takes precedence over env var."""
        with patch.dict(os.environ, {"PAGODA_MASS_API_KEY": "env-token"}):
            client = MassApiClient(
                mass_api_url="http://mass:8080/api",
                defaults=_make_defaults(),
                api_key="explicit-token",
            )

        assert client._api_key == "explicit-token"

    def test_mtls_ssl_context_created_with_certs(self, tmp_path):
        """mTLS SSL context is created when cert and key paths are provided."""
        # Create dummy cert/key files (content doesn't matter for this test)
        cert_file = tmp_path / "client.crt"
        key_file = tmp_path / "client.key"
        cert_file.write_text("dummy-cert")
        key_file.write_text("dummy-key")

        with patch("ssl.SSLContext.load_cert_chain") as mock_load:
            client = MassApiClient(
                mass_api_url="http://mass:8080/api",
                defaults=_make_defaults(),
                client_cert_path=str(cert_file),
                client_key_path=str(key_file),
            )

        assert client._ssl_context is not None
        mock_load.assert_called_once_with(
            certfile=str(cert_file), keyfile=str(key_file),
        )

    def test_no_ssl_context_without_certs(self):
        """No SSL context when cert paths are not provided."""
        with patch.dict(os.environ, {}, clear=True):
            client = MassApiClient(
                mass_api_url="http://mass:8080/api",
                defaults=_make_defaults(),
            )

        assert client._ssl_context is None

    @pytest.mark.asyncio
    async def test_bearer_token_sent_on_every_retry(self):
        """Bearer token is included in headers on retry attempts too."""
        client = MassApiClient(
            mass_api_url="http://mass:8080/api",
            defaults=_make_defaults(),
            api_key="retry-token",
        )

        fail_resp = AsyncMock()
        fail_resp.status = 500

        ok_resp = AsyncMock()
        ok_resp.status = 200
        ok_resp.json = AsyncMock(return_value={
            "priority": "normal",
            "qps_limit": 10.0,
            "concurrent_limit": 5,
            "kv_cache_block_quota": 100,
        })

        mock_session = AsyncMock()
        call_count = 0

        async def mock_get_ctx(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Verify token on every call
            headers = kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer retry-token", \
                f"Missing auth header on attempt {call_count}"
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(
                return_value=fail_resp if call_count == 1 else ok_resp,
            )
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session.get = mock_get_ctx

        with patch("aiohttp.ClientSession") as mock_cls, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_session,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client._fetch_from_mass("t1", max_retries=1)

        assert result is not None
        assert call_count == 2  # one fail + one success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
