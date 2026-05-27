"""
Tests for LIT-2594 / GitHub issue #25335:
`cache_control_injection_points` was silently dropped on the /responses API path.

The Responses-API to Chat-Completion transform constructed a fresh
`GenericChatCompletionMessage` containing only `role` and `content`, discarding
any top-level `cache_control` that the `AnthropicCacheControlHook` had placed
on the input item for the string-content case. The downstream
`anthropic_messages_pt` already reads top-level `cache_control` for system /
user / assistant string-content messages, so the fix is to preserve that key
through the transform.

These tests pin:
  1. String-content user/system messages carry top-level cache_control through.
  2. List-content (already-working) path is not regressed.
  3. The full hook + transform pipeline (matching the GitHub issue #25335 repro
     shape) ends up with cache_control on the Anthropic-ready messages.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _msg_to_dict(m):
    if isinstance(m, dict):
        return m
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return dict(m)


class TestResponsesApiCacheControlPropagation:
    """Cover the dropped-cache_control bug in the Responses to Chat transform."""

    def test_string_content_top_level_cache_control_is_preserved(self):
        input_item = {
            "role": "user",
            "content": "Hello",
            "cache_control": {"type": "ephemeral"},
        }
        msgs = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item
        )
        assert len(msgs) == 1
        m = _msg_to_dict(msgs[0])
        assert m["role"] == "user"
        assert m["content"] == "Hello"
        assert m.get("cache_control") == {"type": "ephemeral"}, (
            "top-level cache_control on the Responses-API input item must be "
            "propagated onto the chat-completion message (LIT-2594 / #25335)"
        )

    def test_string_content_without_cache_control_is_unchanged(self):
        input_item = {"role": "user", "content": "Hello"}
        msgs = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item
        )
        m = _msg_to_dict(msgs[0])
        assert "cache_control" not in m

    def test_list_content_inner_cache_control_still_propagates(self):
        input_item = {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are helpful.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        msgs = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item
        )
        m = _msg_to_dict(msgs[0])
        assert m["role"] == "system"
        assert isinstance(m["content"], list)
        assert m["content"][0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in m

    def test_list_and_top_level_cache_control_both_survive(self):
        input_item = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "cache_control": {"type": "ephemeral"},
        }
        msgs = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item
        )
        m = _msg_to_dict(msgs[0])
        assert m.get("cache_control") == {"type": "ephemeral"}
        assert m["content"][0].get("cache_control") == {"type": "ephemeral"}

    def test_explicit_null_cache_control_is_not_propagated(self):
        input_item = {
            "role": "user",
            "content": "Hello",
            "cache_control": None,
        }
        msgs = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item
        )
        m = _msg_to_dict(msgs[0])
        assert "cache_control" not in m

    def test_full_hook_plus_transform_string_content_pipeline(self):
        input_items = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "please cache me"},
        ]
        injection_points = [
            {"location": "message", "role": "system", "index": 0},
            {"location": "message", "role": "user", "index": -1},
        ]
        hook = AnthropicCacheControlHook()
        _model, processed_input, remaining = hook.get_chat_completion_prompt(
            model="claude-sonnet-4-6",
            messages=copy.deepcopy(input_items),
            non_default_params={"cache_control_injection_points": injection_points},
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        assert "cache_control_injection_points" not in remaining
        assert processed_input[0].get("cache_control") == {"type": "ephemeral"}
        assert processed_input[-1].get("cache_control") == {"type": "ephemeral"}

        chat_messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=processed_input,
            responses_api_request={},
        )
        chat_dicts = [_msg_to_dict(m) for m in chat_messages]
        sys_msg = next(m for m in chat_dicts if m.get("role") == "system")
        usr_msg = next(m for m in reversed(chat_dicts) if m.get("role") == "user")
        assert sys_msg.get("cache_control") == {"type": "ephemeral"}
        assert usr_msg.get("cache_control") == {"type": "ephemeral"}

    def test_none_content_input_item_still_skipped_after_fix(self):
        input_item = {
            "role": "user",
            "content": None,
            "cache_control": {"type": "ephemeral"},
        }
        msgs = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item
        )
        assert msgs == []
