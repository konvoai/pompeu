import { type LanguageModel } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createVertex } from "@ai-sdk/google-vertex";
import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createXai } from "@ai-sdk/xai";

const openaiProvider = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const vertexProvider = createVertex({
  googleAuthOptions: {
    credentials: {
      project_id: process.env.GOOGLE_CLOUD_PROJECT_ID,
      private_key: process.env.VERTEX_CERT,
      client_email: process.env.GOOGLE_CLOUD_CLIENT_EMAIL,
    },
  },
  project: process.env.GOOGLE_CLOUD_PROJECT_ID,
  location: "europe-west1",
});

const amazonBedrockProvider = createAmazonBedrock({
  apiKey: process.env.AWS_BEDROCK_API_KEY,
  region: "eu-central-1",
});

const anthropicProvider = createAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const xaiProvider = createXai({
  apiKey: process.env.XAI_API_KEY,
});

/* VERTEX */
const vertexGemini25Flash001 = vertexProvider.languageModel(
  "gemini-2.5-flash-lite"
);
/* OPENAI */
const openaiGpt5 = openaiProvider.languageModel("gpt-5");
const openaiGpt5Nano = openaiProvider.languageModel("gpt-5-nano");
const openaiGpt5Mini = openaiProvider.languageModel("gpt-5-mini");
const openaiGpt4o = openaiProvider.languageModel("gpt-4o");
const openaiGpt4oMini = openaiProvider.languageModel("gpt-4o-mini");
const openaiO3 = openaiProvider.languageModel("o3-2025-04-16");
const openaiGpt51 = openaiProvider.languageModel("gpt-5.1");

/* ANTHROPIC */
const anthropicClaude41Opus = anthropicProvider.languageModel(
  "claude-opus-4-1-20250805"
);
const anthropicClaude4Sonnet = anthropicProvider.languageModel(
  "claude-sonnet-4-20250514"
);
const anthropicClaude45Sonnet = anthropicProvider.languageModel(
  "claude-sonnet-4-5-20250929"
);
const anthropicClaude45Haiku = anthropicProvider.languageModel(
  "claude-haiku-4-5-20251001"
);

/* BEDROCK */
const amazonBedrockClaude45Sonnet = amazonBedrockProvider.languageModel(
  "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
);
const amazonBedrockClaude4Sonnet = amazonBedrockProvider.languageModel(
  "eu.anthropic.claude-sonnet-4-20250514-v1:0"
);
const amazonBedrockClaude35Haiku = amazonBedrockProvider.languageModel(
  "eu.anthropic.claude-3-5-haiku-20241022-v1:0"
);
const gptOss120b = amazonBedrockProvider.languageModel(
  "openai.gpt-oss-120b-1:0"
);

/* XAI */
const grok4Fastest = xaiProvider.languageModel("grok-4-fast-non-reasoning");
const grok4Fast = xaiProvider.languageModel("grok-4-fast-reasoning");
const grok4NotSoFast = xaiProvider.languageModel("grok-4");

// the ai that judges the other AI's
export const judger = anthropicClaude45Sonnet;

export const modelsToJudge = {
  vertexGemini25Flash001: vertexGemini25Flash001,
  openaiGpt5: openaiGpt5,
  openaiGpt5Nano: openaiGpt5Nano,
  openaiGpt5Mini: openaiGpt5Mini,
  openaiGpt4o: openaiGpt4o,
  openaiGpt4oMini: openaiGpt4oMini,
  gpt51: openaiGpt51,
  openaiO3: openaiO3,
  anthropicClaude41Opus: anthropicClaude41Opus,
  anthropicClaude4Sonnet: anthropicClaude4Sonnet,
  anthropicClaude45Sonnet: anthropicClaude45Sonnet,
  anthropicClaude45Haiku: anthropicClaude45Haiku,
  // amazonBedrockClaude45Sonnet: amazonBedrockClaude45Sonnet,
  // amazonBedrockClaude4Sonnet: amazonBedrockClaude4Sonnet,
  // amazonBedrockClaude35Haiku: amazonBedrockClaude35Haiku,
  gptOss120b: gptOss120b,
  grok4Fastest: grok4Fastest,
  grok4Fast: grok4Fast,
  grok4NotSoFast: grok4NotSoFast,
};

export const dedent = (text: string) => {
  return text
    .split("\n")
    .map((line) => line.trim())
    .join("\n");
};

export type ConversationMesage = {
  from: "buyer" | "store";
  message: string;
};

export type ExportedConversation = {
  goal: string;
  modelName: string;
  model: LanguageModel;
  conversation: ConversationMesage[];
  startTime: number;
  endTime: number;
};
