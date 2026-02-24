# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt template management with tenant/model/default priority."""

from __future__ import annotations

import string

from vllm.logger import init_logger
from vllm.pagoda.config import PagodaConfig

logger = init_logger(__name__)


class PromptTemplateManager:
    """Apply system prompt templates to chat completion requests.

    Priority order:
    1. Tenant-specific prompt_template (from config)
    2. Default prompt_template (from config defaults)
    3. No modification (passthrough)

    Template variables:
    - ${tenant_id} — resolved tenant identifier
    - ${model} — model name from request
    - ${date} — current date (YYYY-MM-DD)
    """

    def __init__(self, config: PagodaConfig) -> None:
        self._config = config

    def apply(
        self,
        request: object,
        tenant_id: str,
    ) -> object:
        """Apply prompt template to a ChatCompletionRequest.

        Injects or prepends a system message based on the resolved template.
        Modifies the request in-place and returns it.
        """
        messages = getattr(request, "messages", None)
        if messages is None:
            return request

        # Resolve template
        tc = self._config.get_tenant_config(tenant_id)
        template = tc.prompt_template
        if template is None:
            defaults = self._config.get_defaults()
            template = defaults.prompt_template
        if template is None:
            return request  # No template configured

        # Substitute variables
        model_name = getattr(request, "model", "unknown")
        try:
            import datetime

            content = string.Template(template).safe_substitute(
                tenant_id=tenant_id,
                model=model_name,
                date=datetime.date.today().isoformat(),
            )
        except (ValueError, KeyError):
            content = template

        # Inject system message
        system_msg = {"role": "system", "content": content}

        if messages and isinstance(messages, list):
            # Check if first message is already a system message
            first = messages[0] if messages else None
            if isinstance(first, dict) and first.get("role") == "system":
                # Prepend our template before existing system message
                messages.insert(0, system_msg)
            elif hasattr(first, "role") and first.role == "system":
                # Pydantic model style
                messages.insert(0, system_msg)
            else:
                messages.insert(0, system_msg)

        return request
