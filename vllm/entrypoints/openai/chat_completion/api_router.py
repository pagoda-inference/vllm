# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from collections.abc import AsyncGenerator

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Chat Completions API"
        )

    # --- Pagoda: apply prompt template ---
    prompt_mgr = getattr(raw_request.app.state, "pagoda_prompt_template_manager", None)
    if prompt_mgr is not None:
        tenant_id = raw_request.scope.get("state", {}).get(
            "pagoda_tenant_id", "__default__"
        )
        prompt_mgr.apply(request, tenant_id)

    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        generator = handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    # --- Pagoda: tool call post-processing ---
    postprocessor = getattr(
        raw_request.app.state, "pagoda_tool_call_postprocessor", None
    )
    tenant_id = raw_request.scope.get("state", {}).get(
        "pagoda_tenant_id", "__default__"
    )

    if isinstance(generator, ChatCompletionResponse):
        if postprocessor is not None:
            generator = _apply_tool_call_fix(
                generator, postprocessor, request.model, tenant_id
            )
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    # Streaming — wrap generator with tool call accumulator
    if postprocessor is not None:
        generator = _wrap_streaming_tool_calls(
            generator, postprocessor, request.model, tenant_id
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/render",
    dependencies=[Depends(validate_json_request)],
    response_model=list,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Render chat completion request and return conversation and engine
    prompts without generating."""
    handler = chat(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Chat Completions API"
        )

    try:
        result = await handler.render_chat_request(request)
    except Exception as e:
        result = handler.create_error_response(e)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result)


# --- Pagoda helper functions ---

def _apply_tool_call_fix(
    response: ChatCompletionResponse,
    postprocessor: "ToolCallPostProcessor",
    model_name: str,
    tenant_id: str,
) -> ChatCompletionResponse:
    """Repair tool call arguments JSON in a non-streaming response."""
    from vllm.pagoda.tool_call_postprocessor import ToolCallPostProcessor

    for choice in response.choices:
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            args_str = getattr(func, "arguments", None)
            if not args_str or not isinstance(args_str, str):
                continue
            result = postprocessor.fix(args_str, model_name, tenant_id)
            if result.success and result.repaired and result.tool_calls:
                import json
                func.arguments = json.dumps(
                    result.tool_calls[0]
                    if len(result.tool_calls) == 1
                    else result.tool_calls
                )
    return response


async def _wrap_streaming_tool_calls(
    original_generator: AsyncGenerator,
    postprocessor: "ToolCallPostProcessor",
    model_name: str,
    tenant_id: str,
) -> AsyncGenerator:
    """Wrap a streaming generator with tool call accumulation and repair."""
    from vllm.pagoda.tool_call_postprocessor import StreamingToolCallAccumulator

    accumulator = StreamingToolCallAccumulator(
        model_name=model_name,
        tenant_id=tenant_id,
        postprocessor=postprocessor,
    )

    async for chunk in original_generator:
        result = accumulator.feed_chunk(chunk)
        if result is not None:
            yield result

    # Finalize any pending tool calls at end of stream
    pending = accumulator.finalize_if_pending()
    if pending and pending.success:
        # Already emitted via feed_chunk's final chunk handling
        pass


def attach_router(app: FastAPI):
    app.include_router(router)
