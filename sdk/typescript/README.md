# AgentField TypeScript SDK

The TypeScript SDK provides an idiomatic Node.js interface for building and running AgentField agents. It mirrors the Python SDK APIs, including AI, memory, discovery, and MCP tooling.

## Installing
```bash
npm install @agentfield/sdk
```

## Rate limiting
AI calls are wrapped with a stateless rate limiter that matches the Python SDK: exponential backoff, container-scoped jitter, Retry-After support, and a circuit breaker.

Configure per-agent via `aiConfig`:
```ts
import { Agent } from '@agentfield/sdk';

const agent = new Agent({
  nodeId: 'demo',
  aiConfig: {
    model: 'gpt-4o',
    enableRateLimitRetry: true,           // default: true
    rateLimitMaxRetries: 20,              // max retry attempts
    rateLimitBaseDelay: 1.0,              // seconds
    rateLimitMaxDelay: 300.0,             // seconds cap
    rateLimitJitterFactor: 0.25,          // ±25% jitter
    rateLimitCircuitBreakerThreshold: 10, // consecutive failures before opening
    rateLimitCircuitBreakerTimeout: 300   // seconds before closing breaker
  }
});
```

To disable retries, set `enableRateLimitRetry: false`.

You can also use the limiter directly:
```ts
import { StatelessRateLimiter } from '@agentfield/sdk';

const limiter = new StatelessRateLimiter({ maxRetries: 3, baseDelay: 0.5 });
const result = await limiter.executeWithRetry(() => makeAiCall());
```
