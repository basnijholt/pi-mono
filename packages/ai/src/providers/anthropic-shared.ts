/**
 * Shared utilities for Anthropic API providers (direct API and Vertex AI).
 */

import type {
	Tool as AnthropicTool,
	ContentBlockParam,
	MessageParam,
	RawMessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages.js";
import { calculateCost } from "../models.js";
import type {
	Api,
	AssistantMessage,
	CacheRetention,
	ImageContent,
	Message,
	Model,
	SimpleStreamOptions,
	StopReason,
	TextContent,
	ThinkingContent,
	Tool,
	ToolCall,
	ToolResultMessage,
} from "../types.js";
import type { AssistantMessageEventStream } from "../utils/event-stream.js";
import { parseStreamingJson } from "../utils/json-parse.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import { transformMessages } from "./transform-messages.js";

/**
 * Convert content blocks to Anthropic API format.
 * Used for tool result content that can contain text and images.
 */
export function convertContentBlocks(content: (TextContent | ImageContent)[]):
	| string
	| Array<
			| { type: "text"; text: string }
			| {
					type: "image";
					source: {
						type: "base64";
						media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp";
						data: string;
					};
			  }
	  > {
	// If only text blocks, return as concatenated string for simplicity
	const hasImages = content.some((c) => c.type === "image");
	if (!hasImages) {
		return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join("\n"));
	}

	// If we have images, convert to content block array
	const blocks = content.map((block) => {
		if (block.type === "text") {
			return {
				type: "text" as const,
				text: sanitizeSurrogates(block.text),
			};
		}
		return {
			type: "image" as const,
			source: {
				type: "base64" as const,
				media_type: block.mimeType as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
				data: block.data,
			},
		};
	});

	// If only images (no text), add placeholder text block
	const hasText = blocks.some((b) => b.type === "text");
	if (!hasText) {
		blocks.unshift({
			type: "text" as const,
			text: "(see attached image)",
		});
	}

	return blocks;
}

/**
 * Merge multiple header objects, later sources override earlier ones.
 */
export function mergeHeaders(...headerSources: (Record<string, string> | undefined)[]): Record<string, string> {
	const merged: Record<string, string> = {};
	for (const headers of headerSources) {
		if (headers) {
			Object.assign(merged, headers);
		}
	}
	return merged;
}

/**
 * Normalize tool call IDs to match Anthropic's required pattern and length.
 * Anthropic requires IDs matching ^[a-zA-Z0-9_-]+$ (max 64 chars).
 */
export function normalizeToolCallId(id: string): string {
	return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

/**
 * Map Anthropic stop reasons to unified StopReason type.
 */
export function mapStopReason(reason: string): StopReason {
	switch (reason) {
		case "end_turn":
			return "stop";
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		case "refusal":
			return "error";
		case "pause_turn":
			return "stop";
		case "stop_sequence":
			return "stop";
		case "sensitive": // Content flagged by safety filters
			return "error";
		default:
			throw new Error(`Unhandled stop reason: ${reason}`);
	}
}

export type AnthropicEffortLevel = "low" | "medium" | "high" | "max";

/**
 * Resolve cache retention preference.
 * Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
 */
export function resolveCacheRetention(cacheRetention?: CacheRetention): CacheRetention {
	if (cacheRetention) {
		return cacheRetention;
	}
	if (typeof process !== "undefined" && process.env.PI_CACHE_RETENTION === "long") {
		return "long";
	}
	return "short";
}

/**
 * Get cache control settings for Anthropic API requests.
 * The extended 1h TTL is only supported on api.anthropic.com (direct Anthropic API).
 * Vertex AI does not support extended TTL, so it always uses ephemeral cache.
 */
export function getCacheControl(
	baseUrl: string,
	cacheRetention?: CacheRetention,
): { retention: CacheRetention; cacheControl?: { type: "ephemeral"; ttl?: "1h" } } {
	const retention = resolveCacheRetention(cacheRetention);
	if (retention === "none") {
		return { retention };
	}
	const ttl = retention === "long" && baseUrl.includes("api.anthropic.com") ? "1h" : undefined;
	return {
		retention,
		cacheControl: { type: "ephemeral", ...(ttl && { ttl }) },
	};
}

/**
 * Check if a model supports adaptive thinking (Opus 4.6+).
 */
export function supportsAdaptiveThinking(modelId: string): boolean {
	return modelId.includes("opus-4-6") || modelId.includes("opus-4.6");
}

/**
 * Map SimpleStreamOptions reasoning level to Anthropic effort levels for adaptive thinking.
 */
export function mapThinkingLevelToEffort(level: SimpleStreamOptions["reasoning"]): AnthropicEffortLevel {
	switch (level) {
		case "minimal":
			return "low";
		case "low":
			return "low";
		case "medium":
			return "medium";
		case "high":
			return "high";
		case "xhigh":
			return "max";
		default:
			return "high";
	}
}

interface ConvertAnthropicMessagesOptions {
	mapToolCallName?: (name: string) => string;
}

/**
 * Convert internal message format to Anthropic message parameters.
 */
export function convertAnthropicMessages<TApi extends "anthropic-messages" | "anthropic-vertex">(
	messages: Message[],
	model: Model<TApi>,
	cacheControl?: { type: "ephemeral"; ttl?: "1h" },
	options?: ConvertAnthropicMessagesOptions,
): MessageParam[] {
	const params: MessageParam[] = [];

	// Transform messages for cross-provider compatibility
	const transformedMessages = transformMessages(messages, model, normalizeToolCallId);

	for (let i = 0; i < transformedMessages.length; i++) {
		const msg = transformedMessages[i];

		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim().length > 0) {
					params.push({
						role: "user",
						content: sanitizeSurrogates(msg.content),
					});
				}
			} else {
				const blocks: ContentBlockParam[] = msg.content.map((item) => {
					if (item.type === "text") {
						return {
							type: "text",
							text: sanitizeSurrogates(item.text),
						};
					}
					return {
						type: "image",
						source: {
							type: "base64",
							media_type: item.mimeType as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
							data: item.data,
						},
					};
				});
				let filteredBlocks = !model.input.includes("image") ? blocks.filter((b) => b.type !== "image") : blocks;
				filteredBlocks = filteredBlocks.filter((b) => {
					if (b.type === "text") {
						return b.text.trim().length > 0;
					}
					return true;
				});
				if (filteredBlocks.length === 0) continue;
				params.push({
					role: "user",
					content: filteredBlocks,
				});
			}
		} else if (msg.role === "assistant") {
			const blocks: ContentBlockParam[] = [];

			for (const block of msg.content) {
				if (block.type === "text") {
					if (block.text.trim().length === 0) continue;
					blocks.push({
						type: "text",
						text: sanitizeSurrogates(block.text),
					});
				} else if (block.type === "thinking") {
					if (block.thinking.trim().length === 0) continue;
					// If thinking signature is missing/empty (e.g., from aborted stream),
					// convert to plain text block without <thinking> tags to avoid API rejection.
					if (!block.thinkingSignature || block.thinkingSignature.trim().length === 0) {
						blocks.push({
							type: "text",
							text: sanitizeSurrogates(block.thinking),
						});
					} else {
						blocks.push({
							type: "thinking",
							thinking: sanitizeSurrogates(block.thinking),
							signature: block.thinkingSignature,
						});
					}
				} else if (block.type === "toolCall") {
					blocks.push({
						type: "tool_use",
						id: block.id,
						name: options?.mapToolCallName ? options.mapToolCallName(block.name) : block.name,
						input: block.arguments ?? {},
					});
				}
			}
			if (blocks.length === 0) continue;
			params.push({
				role: "assistant",
				content: blocks,
			});
		} else if (msg.role === "toolResult") {
			// Collect all consecutive toolResult messages.
			const toolResults: ContentBlockParam[] = [];

			toolResults.push({
				type: "tool_result",
				tool_use_id: msg.toolCallId,
				content: convertContentBlocks(msg.content),
				is_error: msg.isError,
			});

			let j = i + 1;
			while (j < transformedMessages.length && transformedMessages[j].role === "toolResult") {
				const nextMsg = transformedMessages[j] as ToolResultMessage;
				toolResults.push({
					type: "tool_result",
					tool_use_id: nextMsg.toolCallId,
					content: convertContentBlocks(nextMsg.content),
					is_error: nextMsg.isError,
				});
				j++;
			}

			i = j - 1;

			params.push({
				role: "user",
				content: toolResults,
			});
		}
	}

	// Add cache_control to the last user message to cache conversation history.
	if (cacheControl && params.length > 0) {
		const lastMessage = params[params.length - 1];
		if (lastMessage.role === "user") {
			if (Array.isArray(lastMessage.content)) {
				const lastBlock = lastMessage.content[lastMessage.content.length - 1];
				if (
					lastBlock &&
					(lastBlock.type === "text" || lastBlock.type === "image" || lastBlock.type === "tool_result")
				) {
					(lastBlock as { cache_control?: { type: "ephemeral"; ttl?: "1h" } }).cache_control = cacheControl;
				}
			} else if (typeof lastMessage.content === "string") {
				lastMessage.content = [{ type: "text", text: lastMessage.content, cache_control: cacheControl }];
			}
		}
	}

	return params;
}

/**
 * Convert tool definitions to Anthropic tool schema.
 */
export function convertAnthropicTools(
	tools: Tool[] | undefined,
	mapToolName?: (name: string) => string,
): AnthropicTool[] {
	if (!tools) {
		return [];
	}

	return tools.map((tool) => {
		const jsonSchema = tool.parameters as { properties?: Record<string, unknown>; required?: string[] };
		return {
			name: mapToolName ? mapToolName(tool.name) : tool.name,
			description: tool.description,
			input_schema: {
				type: "object" as const,
				properties: jsonSchema.properties || {},
				required: jsonSchema.required || [],
			},
		};
	});
}

type AnthropicStreamingBlock = (ThinkingContent | TextContent | (ToolCall & { partialJson: string })) & {
	index: number;
};

interface ProcessAnthropicStreamOptions<TApi extends Api> {
	model: Model<TApi>;
	output: AssistantMessage;
	stream: AssistantMessageEventStream;
	events: AsyncIterable<RawMessageStreamEvent>;
	mapIncomingToolName?: (name: string) => string;
}

/**
 * Process Anthropic streaming events and emit normalized stream events.
 */
export async function processAnthropicStreamEvents<TApi extends Api>(
	options: ProcessAnthropicStreamOptions<TApi>,
): Promise<void> {
	const blocks = options.output.content as AnthropicStreamingBlock[];

	for await (const event of options.events) {
		if (event.type === "message_start") {
			options.output.usage.input = event.message.usage.input_tokens || 0;
			options.output.usage.output = event.message.usage.output_tokens || 0;
			options.output.usage.cacheRead = event.message.usage.cache_read_input_tokens || 0;
			options.output.usage.cacheWrite = event.message.usage.cache_creation_input_tokens || 0;
			options.output.usage.totalTokens =
				options.output.usage.input +
				options.output.usage.output +
				options.output.usage.cacheRead +
				options.output.usage.cacheWrite;
			calculateCost(options.model, options.output.usage);
		} else if (event.type === "content_block_start") {
			if (event.content_block.type === "text") {
				const block: AnthropicStreamingBlock = {
					type: "text",
					text: "",
					index: event.index,
				};
				options.output.content.push(block);
				options.stream.push({
					type: "text_start",
					contentIndex: options.output.content.length - 1,
					partial: options.output,
				});
			} else if (event.content_block.type === "thinking") {
				const block: AnthropicStreamingBlock = {
					type: "thinking",
					thinking: "",
					thinkingSignature: "",
					index: event.index,
				};
				options.output.content.push(block);
				options.stream.push({
					type: "thinking_start",
					contentIndex: options.output.content.length - 1,
					partial: options.output,
				});
			} else if (event.content_block.type === "tool_use") {
				const block: AnthropicStreamingBlock = {
					type: "toolCall",
					id: event.content_block.id,
					name: options.mapIncomingToolName
						? options.mapIncomingToolName(event.content_block.name)
						: event.content_block.name,
					arguments: (event.content_block.input as Record<string, any>) ?? {},
					partialJson: "",
					index: event.index,
				};
				options.output.content.push(block);
				options.stream.push({
					type: "toolcall_start",
					contentIndex: options.output.content.length - 1,
					partial: options.output,
				});
			}
		} else if (event.type === "content_block_delta") {
			if (event.delta.type === "text_delta") {
				const index = blocks.findIndex((b) => b.index === event.index);
				const block = blocks[index];
				if (block && block.type === "text") {
					block.text += event.delta.text;
					options.stream.push({
						type: "text_delta",
						contentIndex: index,
						delta: event.delta.text,
						partial: options.output,
					});
				}
			} else if (event.delta.type === "thinking_delta") {
				const index = blocks.findIndex((b) => b.index === event.index);
				const block = blocks[index];
				if (block && block.type === "thinking") {
					block.thinking += event.delta.thinking;
					options.stream.push({
						type: "thinking_delta",
						contentIndex: index,
						delta: event.delta.thinking,
						partial: options.output,
					});
				}
			} else if (event.delta.type === "input_json_delta") {
				const index = blocks.findIndex((b) => b.index === event.index);
				const block = blocks[index];
				if (block && block.type === "toolCall") {
					block.partialJson += event.delta.partial_json;
					block.arguments = parseStreamingJson(block.partialJson);
					options.stream.push({
						type: "toolcall_delta",
						contentIndex: index,
						delta: event.delta.partial_json,
						partial: options.output,
					});
				}
			} else if (event.delta.type === "signature_delta") {
				const index = blocks.findIndex((b) => b.index === event.index);
				const block = blocks[index];
				if (block && block.type === "thinking") {
					block.thinkingSignature = block.thinkingSignature || "";
					block.thinkingSignature += event.delta.signature;
				}
			}
		} else if (event.type === "content_block_stop") {
			const index = blocks.findIndex((b) => b.index === event.index);
			const block = blocks[index];
			if (block) {
				delete (block as { index?: number }).index;
				if (block.type === "text") {
					options.stream.push({
						type: "text_end",
						contentIndex: index,
						content: block.text,
						partial: options.output,
					});
				} else if (block.type === "thinking") {
					options.stream.push({
						type: "thinking_end",
						contentIndex: index,
						content: block.thinking,
						partial: options.output,
					});
				} else if (block.type === "toolCall") {
					block.arguments = parseStreamingJson(block.partialJson);
					delete (block as { partialJson?: string }).partialJson;
					options.stream.push({
						type: "toolcall_end",
						contentIndex: index,
						toolCall: block,
						partial: options.output,
					});
				}
			}
		} else if (event.type === "message_delta") {
			if (event.delta.stop_reason) {
				options.output.stopReason = mapStopReason(event.delta.stop_reason);
			}
			if (event.usage.input_tokens != null) {
				options.output.usage.input = event.usage.input_tokens;
			}
			if (event.usage.output_tokens != null) {
				options.output.usage.output = event.usage.output_tokens;
			}
			if (event.usage.cache_read_input_tokens != null) {
				options.output.usage.cacheRead = event.usage.cache_read_input_tokens;
			}
			if (event.usage.cache_creation_input_tokens != null) {
				options.output.usage.cacheWrite = event.usage.cache_creation_input_tokens;
			}
			options.output.usage.totalTokens =
				options.output.usage.input +
				options.output.usage.output +
				options.output.usage.cacheRead +
				options.output.usage.cacheWrite;
			calculateCost(options.model, options.output.usage);
		}
	}
}

/**
 * Remove temporary fields used while streaming.
 */
export function cleanupAnthropicStreamBlocks(content: AssistantMessage["content"]): void {
	for (const block of content) {
		delete (block as { index?: number }).index;
		delete (block as { partialJson?: string }).partialJson;
	}
}

/**
 * Map generic stream metadata to Anthropic-specific metadata.
 */
export function applyAnthropicUserMetadata(
	params: { metadata?: { user_id?: string | null } },
	metadata?: Record<string, unknown>,
): void {
	const userId = metadata?.user_id;
	if (typeof userId === "string") {
		params.metadata = { user_id: userId };
	}
}
