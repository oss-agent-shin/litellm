import base64
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponseText,
)
from litellm.types.responses.main import DecodedResponseId
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    SpecialEnums,
    Usage,
)


class ResponsesAPIRequestUtils:
    """Helper utils for constructing ResponseAPI requests"""

    @staticmethod
    def _check_valid_arg(
        supported_params: Optional[List[str]],
        non_default_params: Dict,
        drop_params: Optional[bool],
        custom_llm_provider: Optional[str],
        model: str,
    ):
        if supported_params is None:
            return
        unsupported_params = {}
        for k in non_default_params.keys():
            if k not in supported_params:
                unsupported_params[k] = non_default_params[k]
        if unsupported_params:
            if litellm.drop_params is True or (
                drop_params is not None and drop_params is True
            ):
                pass
            else:
                raise litellm.UnsupportedParamsError(
                    status_code=500,
                    message=f"{custom_llm_provider} does not support parameters: {unsupported_params}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
                )

    @staticmethod
    def get_optional_params_responses_api(
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        allowed_openai_params: Optional[List[str]] = None,
    ) -> Dict:
        """
        Get optional parameters for the responses API.

        Args:
            params: Dictionary of all parameters
            model: The model name
            responses_api_provider_config: The provider configuration for responses API

        Returns:
            A dictionary of supported parameters for the responses API
        """
        from litellm.utils import _apply_openai_param_overrides

        # Remove None values and internal parameters
        # Get supported parameters for the model
        supported_params = responses_api_provider_config.get_supported_openai_params(
            model
        )

        non_default_params = cast(Dict, response_api_optional_params)
        # Check for unsupported parameters
        ResponsesAPIRequestUtils._check_valid_arg(
            supported_params=supported_params + (allowed_openai_params or []),
            non_default_params=non_default_params,
            drop_params=litellm.drop_params,
            custom_llm_provider=responses_api_provider_config.custom_llm_provider,
            model=model,
        )

        # Map parameters to provider-specific format
        mapped_params = responses_api_provider_config.map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        # add any allowed_openai_params to the mapped_params
        mapped_params = _apply_openai_param_overrides(
            optional_params=mapped_params,
            non_default_params=non_default_params,
            allowed_openai_params=allowed_openai_params or [],
        )

        return mapped_params

    @staticmethod
    def get_requested_response_api_optional_param(
        params: Dict[str, Any],
    ) -> ResponsesAPIOptionalRequestParams:
        """
        Filter parameters to only include those defined in ResponsesAPIOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ResponsesAPIOptionalRequestParams instance with only the valid parameters
        """
        from litellm.utils import PreProcessNonDefaultParams

        valid_keys = get_type_hints(ResponsesAPIOptionalRequestParams).keys()
        custom_llm_provider = params.pop("custom_llm_provider", None)
        special_params = params.pop("kwargs", {})

        additional_drop_params = params.pop("additional_drop_params", None)
        non_default_params = (
            PreProcessNonDefaultParams.base_pre_process_non_default_params(
                passed_params=params,
                special_params=special_params,
                custom_llm_provider=custom_llm_provider,
                additional_drop_params=additional_drop_params,
                default_param_values={k: None for k in valid_keys},
                additional_endpoint_specific_params=["input"],
            )
        )

        # decode previous_response_id if it's a litellm encoded id
        if "previous_response_id" in non_default_params:
            decoded_previous_response_id = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
                non_default_params["previous_response_id"]
            )
            non_default_params["previous_response_id"] = decoded_previous_response_id

        if "metadata" in non_default_params:
            from litellm.utils import add_openai_metadata

            converted_metadata = add_openai_metadata(non_default_params["metadata"])
            if converted_metadata is not None:
                non_default_params["metadata"] = converted_metadata
            else:
                non_default_params.pop("metadata", None)

        return cast(ResponsesAPIOptionalRequestParams, non_default_params)

    # PLACEHOLDER
