import { ModelGroup } from "../llm_calls/fetch_models";
import { EndpointType, getEndpointType } from "./mode_endpoint_mapping";

/**
 * Determines the appropriate endpoint type based on the selected model
 *
 * @param selectedModel - The model identifier string
 * @param modelInfo - Array of model information
 * @returns The appropriate endpoint type
 */
export const determineEndpointType = (selectedModel: string, modelInfo: ModelGroup[]): EndpointType => {
  // Find the model information for the selected model
  const selectedModelInfo = modelInfo.find((option) => option.model_group === selectedModel);

  // If model info is found and it has a mode, determine the endpoint type
  if (selectedModelInfo?.mode) {
    return getEndpointType(selectedModelInfo.mode);
  }

  // Default to chat endpoint if no match is found
  return EndpointType.CHAT;
};

/**
 * Returns true if a model is compatible with the given endpoint type.
 *
 * Mirrors the inline filter used by the Playground model dropdown so that
 * any caller (e.g. an auto-select helper) picks a model the dropdown would
 * actually display. A model with no `mode` set is treated as compatible
 * with every endpoint type (the dropdown also shows them unconditionally).
 */
export const isModelCompatibleWithEndpoint = (option: ModelGroup, endpointType: string): boolean => {
  if (!option.mode) {
    // No mode -> the dropdown shows it for every endpoint.
    return true;
  }
  const optionEndpoint = getEndpointType(option.mode);
  if (
    endpointType === EndpointType.RESPONSES ||
    endpointType === EndpointType.ANTHROPIC_MESSAGES ||
    endpointType === EndpointType.INTERACTIONS
  ) {
    return optionEndpoint === endpointType || optionEndpoint === EndpointType.CHAT;
  }
  if (endpointType === EndpointType.IMAGE_EDITS) {
    return optionEndpoint === endpointType || optionEndpoint === EndpointType.IMAGE;
  }
  return optionEndpoint === endpointType;
};

/**
 * Pick a default model from the supplied list for the given endpoint type.
 *
 * Used by the Playground to auto-populate the model dropdown when the
 * available model list changes (e.g. the user switched to a "Virtual Key"
 * whose `models` restrict what the proxy returns from /model_group/info).
 *
 * Preference order:
 *   1. The first model whose mode is compatible with `endpointType`.
 *   2. The first model in the list (covers the case where every entry has
 *      no `mode` set, which the dropdown also shows for every endpoint).
 *
 * Returns `undefined` only when the list is empty.
 */
export const pickDefaultModelForEndpoint = (
  models: ModelGroup[],
  endpointType: string,
): string | undefined => {
  if (!models.length) return undefined;
  const compatible = models.find((option) => isModelCompatibleWithEndpoint(option, endpointType));
  return (compatible ?? models[0]).model_group;
};
