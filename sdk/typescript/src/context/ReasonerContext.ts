import type express from 'express';
import { ExecutionContext } from './ExecutionContext.js';
import type { AIClient, AIRequestOptions, AIStream } from '../ai/AIClient.js';
import type { MemoryInterface } from '../memory/MemoryInterface.js';
import type { Agent } from '../agent/Agent.js';
import type { WorkflowReporter } from '../workflow/WorkflowReporter.js';
import type { DiscoveryOptions } from '../types/agent.js';
import type { DidInterface } from '../did/DidInterface.js';

export class ReasonerContext<TInput = any> {
  readonly input: TInput;
  readonly executionId: string;
  readonly runId?: string;
  readonly sessionId?: string;
  readonly actorId?: string;
  readonly workflowId?: string;
  readonly parentExecutionId?: string;
  readonly callerDid?: string;
  readonly targetDid?: string;
  readonly agentNodeDid?: string;
  readonly req: express.Request;
  readonly res: express.Response;
  readonly agent: Agent;
  readonly aiClient: AIClient;
  readonly memory: MemoryInterface;
  readonly workflow: WorkflowReporter;
  readonly did: DidInterface;

  constructor(params: {
    input: TInput;
    executionId: string;
    runId?: string;
    sessionId?: string;
    actorId?: string;
    workflowId?: string;
    parentExecutionId?: string;
    callerDid?: string;
    targetDid?: string;
    agentNodeDid?: string;
    req: express.Request;
    res: express.Response;
    agent: Agent;
    aiClient: AIClient;
    memory: MemoryInterface;
    workflow: WorkflowReporter;
    did: DidInterface;
  }) {
    this.input = params.input;
    this.executionId = params.executionId;
    this.runId = params.runId;
    this.sessionId = params.sessionId;
    this.actorId = params.actorId;
    this.workflowId = params.workflowId;
    this.parentExecutionId = params.parentExecutionId;
    this.callerDid = params.callerDid;
    this.targetDid = params.targetDid;
    this.agentNodeDid = params.agentNodeDid;
    this.req = params.req;
    this.res = params.res;
    this.agent = params.agent;
    this.aiClient = params.aiClient;
    this.memory = params.memory;
    this.workflow = params.workflow;
    this.did = params.did;
  }

  ai(prompt: string, options?: AIRequestOptions) {
    return this.aiClient.generate(prompt, options);
  }

  aiStream(prompt: string, options?: AIRequestOptions): Promise<AIStream> {
    return this.aiClient.stream(prompt, options);
  }

  call(target: string, input: any) {
    return this.agent.call(target, input);
  }

  discover(options?: DiscoveryOptions) {
    return this.agent.discover(options);
  }
}

export function getCurrentContext<TInput = any>(): ReasonerContext<TInput> | undefined {
  const execution = ExecutionContext.getCurrent();
  if (!execution) return undefined;
  const { metadata, input, agent, req, res } = execution;
  return new ReasonerContext<TInput>({
    input,
    executionId: metadata.executionId,
    runId: metadata.runId,
    sessionId: metadata.sessionId,
    actorId: metadata.actorId,
    workflowId: metadata.workflowId,
    parentExecutionId: metadata.parentExecutionId,
    callerDid: metadata.callerDid,
    targetDid: metadata.targetDid,
    agentNodeDid: metadata.agentNodeDid,
    req,
    res,
    agent,
    aiClient: agent.getAIClient(),
    memory: agent.getMemoryInterface(metadata),
    workflow: agent.getWorkflowReporter(metadata),
    did: agent.getDidInterface(metadata, input)
  });
}
