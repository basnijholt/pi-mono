import type { MessageCreateParamsStreaming } from "@anthropic-ai/sdk/resources/messages.js";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import type {
	Api,
	AssistantMessage,
	Context,
	Model,
	SimpleStreamOptions,
	StreamFunction,
	StreamOptions,
} from "../types.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import {
	type AnthropicEffortLevel,
	applyAnthropicUserMetadata,
	cleanupAnthropicStreamBlocks,
	convertAnthropicMessages,
	convertAnthropicTools,
	getCacheControl,
	mapThinkingLevelToEffort,
	mergeHeaders,
	processAnthropicStreamEvents,
	supportsAdaptiveThinking,
} from "./anthropic-shared.js";
import { adjustMaxTokensForThinking, buildBaseOptions } from "./simple-options.js";

export type AnthropicVertexEffort = AnthropicEffortLevel;

export interface AnthropicVertexOptions extends StreamOptions {
	thinkingEnabled?: boolean;
	thinkingBudgetTokens?: number;
	effort?: AnthropicVertexEffort;
	interleavedThinking?: boolean;
	toolChoice?: "auto" | "any" | "none" | { type: "tool"; name: string };
	project?: string;
	region?: string;
}

export const streamAnthropicVertex: StreamFunction<"anthropic-vertex", AnthropicVertexOptions> = (
	model: Model<"anthropic-vertex">,
	context: Context,
	options?: AnthropicVertexOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: "anthropic-vertex" as Api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const client = createClient(model, options);
			const params = buildParams(model, context, options);
			options?.onPayload?.(params);
			const anthropicStream = client.messages.stream({ ...params, stream: true }, { signal: options?.signal });
			stream.push({ type: "start", partial: output });

			await processAnthropicStreamEvents({
				model,
				output,
				stream,
				events: anthropicStream,
			});

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			if (output.stopReason === "aborted" || output.stopReason === "error") {
				throw new Error("An unknown error occurred");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			cleanupAnthropicStreamBlocks(output.content);
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
};

function createClient(model: Model<"anthropic-vertex">, options?: AnthropicVertexOptions): AnthropicVertex {
	const betaFeatures = ["fine-grained-tool-streaming-2025-05-14"];
	if (options?.interleavedThinking ?? true) {
		betaFeatures.push("interleaved-thinking-2025-05-14");
	}

	const defaultHeaders = mergeHeaders(
		{
			accept: "application/json",
			"anthropic-beta": betaFeatures.join(","),
		},
		model.headers,
		options?.headers,
	);

	const project = options?.project || process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT;
	const region = options?.region || process.env.GOOGLE_CLOUD_LOCATION || "us-east5";

	if (!project) {
		throw new Error(
			"Anthropic Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.",
		);
	}

	return new AnthropicVertex({
		projectId: project,
		region,
		defaultHeaders,
	});
}

function buildParams(
	model: Model<"anthropic-vertex">,
	context: Context,
	options?: AnthropicVertexOptions,
): MessageCreateParamsStreaming {
	const { cacheControl } = getCacheControl(model.baseUrl, options?.cacheRetention);
	const params: MessageCreateParamsStreaming = {
		model: model.id,
		messages: convertAnthropicMessages(context.messages, model, cacheControl),
		max_tokens: options?.maxTokens || (model.maxTokens / 3) | 0,
		stream: true,
	};

	if (context.systemPrompt) {
		params.system = [
			{
				type: "text",
				text: sanitizeSurrogates(context.systemPrompt),
				...(cacheControl ? { cache_control: cacheControl } : {}),
			},
		];
	}

	if (options?.temperature !== undefined) {
		params.temperature = options.temperature;
	}

	if (context.tools) {
		params.tools = convertAnthropicTools(context.tools);
	}

	if (options?.thinkingEnabled && model.reasoning) {
		if (supportsAdaptiveThinking(model.id)) {
			params.thinking = { type: "adaptive" };
			if (options.effort) {
				params.output_config = { effort: options.effort };
			}
		} else {
			params.thinking = {
				type: "enabled",
				budget_tokens: options.thinkingBudgetTokens || 1024,
			};
		}
	}

	applyAnthropicUserMetadata(params, options?.metadata);

	if (options?.toolChoice) {
		if (typeof options.toolChoice === "string") {
			params.tool_choice = { type: options.toolChoice };
		} else {
			params.tool_choice = options.toolChoice;
		}
	}

	return params;
}

export const streamSimpleAnthropicVertex: StreamFunction<"anthropic-vertex", SimpleStreamOptions> = (
	model: Model<"anthropic-vertex">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	// Anthropic Vertex uses ADC, no API key needed
	const base = buildBaseOptions(model, options, undefined);
	if (!options?.reasoning) {
		return streamAnthropicVertex(model, context, {
			...base,
			thinkingEnabled: false,
		} satisfies AnthropicVertexOptions);
	}

	if (supportsAdaptiveThinking(model.id)) {
		const effort = mapThinkingLevelToEffort(options.reasoning);
		return streamAnthropicVertex(model, context, {
			...base,
			thinkingEnabled: true,
			effort,
		} satisfies AnthropicVertexOptions);
	}

	const adjusted = adjustMaxTokensForThinking(
		base.maxTokens || 0,
		model.maxTokens,
		options.reasoning,
		options.thinkingBudgets,
	);

	return streamAnthropicVertex(model, context, {
		...base,
		maxTokens: adjusted.maxTokens,
		thinkingEnabled: true,
		thinkingBudgetTokens: adjusted.thinkingBudget,
	} satisfies AnthropicVertexOptions);
};
