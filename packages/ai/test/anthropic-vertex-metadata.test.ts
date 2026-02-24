import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { streamAnthropicVertex } from "../src/providers/anthropic-vertex.js";
import type { Context } from "../src/types.js";

describe("Anthropic Vertex metadata forwarding", () => {
	it("forwards metadata.user_id to provider payload", async () => {
		const model = getModel("anthropic-vertex", "claude-sonnet-4-5@20250929");
		const context: Context = {
			messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
		};

		const controller = new AbortController();
		controller.abort();

		let capturedPayload: unknown;
		const s = streamAnthropicVertex(model, context, {
			project: "test-project",
			region: "us-east5",
			signal: controller.signal,
			metadata: {
				user_id: "user-123",
				ignored: "value",
			},
			onPayload: (payload) => {
				capturedPayload = payload;
			},
		});

		for await (const event of s) {
			if (event.type === "error" || event.type === "done") {
				break;
			}
		}

		expect(capturedPayload).toBeDefined();
		const payloadMetadata = (capturedPayload as { metadata?: { user_id?: string } }).metadata;
		expect(payloadMetadata).toEqual({ user_id: "user-123" });
	});
});
