import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelGroup } from "../llm_calls/fetch_models";
import {
  determineEndpointType,
  isModelCompatibleWithEndpoint,
  pickDefaultModelForEndpoint,
} from "./EndpointUtils";
import { EndpointType } from "./mode_endpoint_mapping";

// Mock the getEndpointType function
vi.mock("./mode_endpoint_mapping", () => ({
  EndpointType: {
    IMAGE: "image",
    VIDEO: "video",
    CHAT: "chat",
    RESPONSES: "responses",
    IMAGE_EDITS: "image_edits",
    ANTHROPIC_MESSAGES: "anthropic_messages",
    EMBEDDINGS: "embeddings",
    SPEECH: "speech",
    TRANSCRIPTION: "transcription",
    A2A_AGENTS: "a2a_agents",
  },
  getEndpointType: vi.fn(),
  ModelMode: {
    AUDIO_SPEECH: "audio_speech",
    AUDIO_TRANSCRIPTION: "audio_transcription",
    IMAGE_GENERATION: "image_generation",
    VIDEO_GENERATION: "video_generation",
    CHAT: "chat",
    RESPONSES: "responses",
    IMAGE_EDITS: "image_edits",
    ANTHROPIC_MESSAGES: "anthropic_messages",
    EMBEDDING: "embedding",
  },
}));

// Import the mocked function
import { getEndpointType } from "./mode_endpoint_mapping";

describe("determineEndpointType", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return the correct endpoint type when model is found and has a valid mode", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        mode: "chat",
      },
      {
        model_group: "dall-e-3",
        mode: "image_generation",
      },
    ];

    // Mock getEndpointType to return IMAGE for image_generation mode
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.IMAGE);

    const result = determineEndpointType("dall-e-3", mockModelInfo);

    expect(getEndpointType).toHaveBeenCalledWith("image_generation");
    expect(result).toBe(EndpointType.IMAGE);
  });

  it("should return CHAT endpoint type when model is found but has no mode", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        // No mode property
      },
    ];

    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should return CHAT endpoint type when model is not found in modelInfo", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        mode: "chat",
      },
    ];

    const result = determineEndpointType("non-existent-model", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should return CHAT endpoint type when modelInfo array is empty", () => {
    const mockModelInfo: ModelGroup[] = [];

    const result = determineEndpointType("any-model", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle different mode types correctly", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "tts-model",
        mode: "audio_speech",
      },
      {
        model_group: "whisper-model",
        mode: "audio_transcription",
      },
      {
        model_group: "embedding-model",
        mode: "embedding",
      },
      {
        model_group: "video-model",
        mode: "video_generation",
      },
    ];

    // Test speech mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.SPEECH);
    const speechResult = determineEndpointType("tts-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("audio_speech");
    expect(speechResult).toBe(EndpointType.SPEECH);

    // Reset mock for next test
    vi.clearAllMocks();

    // Test transcription mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.TRANSCRIPTION);
    const transcriptionResult = determineEndpointType("whisper-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("audio_transcription");
    expect(transcriptionResult).toBe(EndpointType.TRANSCRIPTION);

    // Reset mock for next test
    vi.clearAllMocks();

    // Test embedding mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.EMBEDDINGS);
    const embeddingResult = determineEndpointType("embedding-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("embedding");
    expect(embeddingResult).toBe(EndpointType.EMBEDDINGS);

    // Reset mock for next test
    vi.clearAllMocks();

    // Test video mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.VIDEO);
    const videoResult = determineEndpointType("video-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("video_generation");
    expect(videoResult).toBe(EndpointType.VIDEO);
  });

  it("should prioritize the first matching model when there are duplicates", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        mode: "chat",
      },
      {
        model_group: "gpt-3.5-turbo",
        mode: "image_generation", // Different mode for same model name
      },
    ];

    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);

    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).toHaveBeenCalledWith("chat");
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle models with undefined mode property explicitly set", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "test-model",
        mode: undefined,
      },
    ];

    const result = determineEndpointType("test-model", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle models with empty string mode", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "test-model",
        mode: "",
      },
    ];

    const result = determineEndpointType("test-model", mockModelInfo);

    // Empty string is falsy, so getEndpointType should not be called
    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle case-sensitive model group matching", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "GPT-3.5-TURBO",
        mode: "chat",
      },
    ];

    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);

    // Test with different case - should not match
    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });
});

describe("isModelCompatibleWithEndpoint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("treats a model with no mode as compatible with every endpoint type", () => {
    const option: ModelGroup = { model_group: "no-mode-model" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.CHAT)).toBe(true);
    expect(isModelCompatibleWithEndpoint(option, EndpointType.IMAGE)).toBe(true);
    expect(isModelCompatibleWithEndpoint(option, EndpointType.EMBEDDINGS)).toBe(true);
    expect(getEndpointType).not.toHaveBeenCalled();
  });

  it("treats a model with an empty-string mode as compatible (falsy mode short-circuit)", () => {
    const option: ModelGroup = { model_group: "blank-mode-model", mode: "" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.CHAT)).toBe(true);
    expect(getEndpointType).not.toHaveBeenCalled();
  });

  it("returns true when the mapped endpoint matches the requested one", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);
    const option: ModelGroup = { model_group: "gpt-4", mode: "chat" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.CHAT)).toBe(true);
  });

  it("returns false when the mapped endpoint does not match", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.IMAGE);
    const option: ModelGroup = { model_group: "dall-e-3", mode: "image_generation" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.CHAT)).toBe(false);
  });

  it("allows chat models on responses/anthropic_messages/interactions endpoints", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);
    const chatModel: ModelGroup = { model_group: "gpt-4", mode: "chat" };
    expect(isModelCompatibleWithEndpoint(chatModel, EndpointType.RESPONSES)).toBe(true);
    expect(isModelCompatibleWithEndpoint(chatModel, EndpointType.ANTHROPIC_MESSAGES)).toBe(true);
    expect(isModelCompatibleWithEndpoint(chatModel, "interactions")).toBe(true);
  });

  it("allows responses-mode models on the responses endpoint", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.RESPONSES);
    const option: ModelGroup = { model_group: "o1", mode: "responses" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.RESPONSES)).toBe(true);
  });

  it("allows image models on the image_edits endpoint as well as on image", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.IMAGE);
    const option: ModelGroup = { model_group: "dall-e-3", mode: "image_generation" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.IMAGE_EDITS)).toBe(true);
    expect(isModelCompatibleWithEndpoint(option, EndpointType.IMAGE)).toBe(true);
  });

  it("rejects unrelated modes on the image_edits endpoint", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);
    const option: ModelGroup = { model_group: "gpt-4", mode: "chat" };
    expect(isModelCompatibleWithEndpoint(option, EndpointType.IMAGE_EDITS)).toBe(false);
  });
});

describe("pickDefaultModelForEndpoint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns undefined when the model list is empty", () => {
    expect(pickDefaultModelForEndpoint([], EndpointType.CHAT)).toBeUndefined();
    expect(getEndpointType).not.toHaveBeenCalled();
  });

  it("picks the first model that is compatible with the endpoint", () => {
    vi.mocked(getEndpointType).mockImplementation((mode: string) =>
      mode === "chat" ? EndpointType.CHAT : EndpointType.IMAGE,
    );
    const models: ModelGroup[] = [
      { model_group: "dall-e-3", mode: "image_generation" },
      { model_group: "gpt-4o-mini", mode: "chat" },
    ];
    expect(pickDefaultModelForEndpoint(models, EndpointType.CHAT)).toBe("gpt-4o-mini");
  });

  it("falls back to the first model when none have a compatible mode", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.IMAGE);
    const models: ModelGroup[] = [
      { model_group: "dall-e-3", mode: "image_generation" },
      { model_group: "stable-diffusion", mode: "image_generation" },
    ];
    expect(pickDefaultModelForEndpoint(models, EndpointType.CHAT)).toBe("dall-e-3");
  });

  it("prefers a mode-less model over a mismatched one", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);
    const models: ModelGroup[] = [
      { model_group: "mystery-model" },
      { model_group: "gpt-4", mode: "chat" },
    ];
    expect(pickDefaultModelForEndpoint(models, EndpointType.EMBEDDINGS)).toBe("mystery-model");
  });

  it("returns the only model when the list has exactly one compatible entry", () => {
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.EMBEDDINGS);
    const models: ModelGroup[] = [{ model_group: "text-embedding-3-small", mode: "embedding" }];
    expect(pickDefaultModelForEndpoint(models, EndpointType.EMBEDDINGS)).toBe(
      "text-embedding-3-small",
    );
  });
});
