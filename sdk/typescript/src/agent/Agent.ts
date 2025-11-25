import express from 'express';
import type http from 'node:http';
import { randomUUID } from 'node:crypto';
import type { AgentConfig, HealthStatus } from '../types/agent.js';
import { ReasonerRegistry } from './ReasonerRegistry.js';
import { SkillRegistry } from './SkillRegistry.js';
import { AgentRouter } from '../router/AgentRouter.js';
import type { ReasonerHandler, ReasonerOptions } from '../types/reasoner.js';
import type { SkillHandler, SkillOptions } from '../types/skill.js';
import { ExecutionContext } from '../context/ExecutionContext.js';
import { ReasonerContext } from '../context/ReasonerContext.js';
import { SkillContext } from '../context/SkillContext.js';
import { AIClient } from '../ai/AIClient.js';
import { AgentFieldClient } from '../client/AgentFieldClient.js';
import { MemoryClient } from '../memory/MemoryClient.js';
import { MemoryEventClient } from '../memory/MemoryEventClient.js';
import { MemoryInterface, type MemoryChangeEvent, type MemoryWatchHandler } from '../memory/MemoryInterface.js';
import { matchesPattern } from '../utils/pattern.js';

export class Agent {
  readonly config: AgentConfig;
  readonly app: express.Express;
  readonly reasoners = new ReasonerRegistry();
  readonly skills = new SkillRegistry();
  private server?: http.Server;
  private heartbeatTimer?: NodeJS.Timeout;
  private readonly aiClient: AIClient;
  private readonly agentFieldClient: AgentFieldClient;
  private readonly memoryClient: MemoryClient;
  private readonly memoryEventClient: MemoryEventClient;
  private readonly memoryWatchers: Array<{ pattern: string; handler: MemoryWatchHandler }> = [];

  constructor(config: AgentConfig) {
    this.config = {
      port: 8001,
      agentFieldUrl: 'http://localhost:8080',
      host: '0.0.0.0',
      ...config
    };

    this.app = express();
    this.app.use(express.json());

    this.aiClient = new AIClient(this.config.aiConfig);
    this.agentFieldClient = new AgentFieldClient(this.config);
    this.memoryClient = new MemoryClient(this.config.agentFieldUrl!);
    this.memoryEventClient = new MemoryEventClient(this.config.agentFieldUrl!);
    this.memoryEventClient.onEvent((event) => this.dispatchMemoryEvent(event));

    this.registerDefaultRoutes();
  }

  reasoner<TInput = any, TOutput = any>(
    name: string,
    handler: ReasonerHandler<TInput, TOutput>,
    options?: ReasonerOptions
  ) {
    this.reasoners.register(name, handler, options);
    return this;
  }

  skill<TInput = any, TOutput = any>(
    name: string,
    handler: SkillHandler<TInput, TOutput>,
    options?: SkillOptions
  ) {
    this.skills.register(name, handler, options);
    return this;
  }

  includeRouter(router: AgentRouter) {
    this.reasoners.includeRouter(router);
    this.skills.includeRouter(router);
  }

  watchMemory(pattern: string | string[], handler: MemoryWatchHandler) {
    const patterns = Array.isArray(pattern) ? pattern : [pattern];
    patterns.forEach((p) => this.memoryWatchers.push({ pattern: p, handler }));
    this.memoryEventClient.start();
  }

  getAIClient() {
    return this.aiClient;
  }

  getMemoryInterface(metadata?: { workflowId?: string; sessionId?: string }) {
    const defaultScopeId = metadata?.workflowId ?? metadata?.sessionId;
    return new MemoryInterface({
      client: this.memoryClient,
      eventClient: this.memoryEventClient,
      defaultScope: this.config.memoryConfig?.defaultScope ?? 'workflow',
      defaultScopeId
    });
  }

  async serve(): Promise<void> {
    await this.registerWithControlPlane();
    const port = this.config.port ?? 8001;
    const host = this.config.host ?? '0.0.0.0';
    await new Promise<void>((resolve, reject) => {
      this.server = this.app
        .listen(port, host, () => resolve())
        .on('error', reject);
    });
    this.memoryEventClient.start();
    this.startHeartbeat();
  }

  async shutdown(): Promise<void> {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }
    await new Promise<void>((resolve, reject) => {
      this.server?.close((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
    this.memoryEventClient.stop();
  }

  async call(target: string, input: any) {
    const { agentId, name } = this.parseTarget(target);
    if (!agentId || agentId === this.config.nodeId) {
      const local = this.reasoners.get(name);
      if (!local) throw new Error(`Reasoner not found: ${name}`);
      const metadata = ExecutionContext.getCurrent()?.metadata ?? {
        executionId: randomUUID()
      };
      const dummyReq = {} as express.Request;
      const dummyRes = {} as express.Response;
      const execCtx = new ExecutionContext({
        input,
        metadata: {
          ...metadata,
          executionId: metadata.executionId ?? randomUUID()
        },
        req: dummyReq,
        res: dummyRes,
        agent: this
      });
      return ExecutionContext.run(execCtx, () =>
        local.handler(
          new ReasonerContext({
            input,
            executionId: execCtx.metadata.executionId,
            runId: execCtx.metadata.runId,
            sessionId: execCtx.metadata.sessionId,
            actorId: execCtx.metadata.actorId,
            workflowId: execCtx.metadata.workflowId,
            parentExecutionId: execCtx.metadata.parentExecutionId,
            callerDid: execCtx.metadata.callerDid,
            targetDid: execCtx.metadata.targetDid,
            agentNodeDid: execCtx.metadata.agentNodeDid,
            req: dummyReq,
            res: dummyRes,
            agent: this,
            aiClient: this.aiClient,
            memory: this.getMemoryInterface(execCtx.metadata)
          })
        )
      );
    }

    return this.agentFieldClient.execute(target, input);
  }

  private registerDefaultRoutes() {
    this.app.get('/health', (_req, res) => {
      res.json(this.health());
    });

    this.app.get('/status', (_req, res) => {
      res.json({
        ...this.health(),
        reasoners: this.reasoners.all().map((r) => r.name),
        skills: this.skills.all().map((s) => s.name)
      });
    });

    this.app.get('/reasoners', (_req, res) => {
      res.json(this.reasoners.all().map((r) => r.name));
    });

    this.app.get('/skills', (_req, res) => {
      res.json(this.skills.all().map((s) => s.name));
    });

    this.app.post('/api/v1/reasoners/*', (req, res) => this.executeReasoner(req, res, (req.params as any)[0]));
    this.app.post('/reasoners/:name', (req, res) => this.executeReasoner(req, res, req.params.name));

    this.app.post('/api/v1/skills/*', (req, res) => this.executeSkill(req, res, (req.params as any)[0]));
    this.app.post('/skills/:name', (req, res) => this.executeSkill(req, res, req.params.name));
  }

  private async executeReasoner(req: express.Request, res: express.Response, name: string) {
    const reasoner = this.reasoners.get(name);
    if (!reasoner) {
      res.status(404).json({ error: `Reasoner not found: ${name}` });
      return;
    }

    const metadata = this.buildMetadata(req);
    const execCtx = new ExecutionContext({ input: req.body, metadata, req, res, agent: this });

    return ExecutionContext.run(execCtx, async () => {
      try {
        const ctx = new ReasonerContext({
          input: req.body,
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
          agent: this,
          aiClient: this.aiClient,
          memory: this.getMemoryInterface(metadata)
        });

        const result = await reasoner.handler(ctx);
        res.json(result);
      } catch (err: any) {
        res.status(500).json({ error: err?.message ?? 'Execution failed' });
      }
    });
  }

  private async executeSkill(req: express.Request, res: express.Response, name: string) {
    const skill = this.skills.get(name);
    if (!skill) {
      res.status(404).json({ error: `Skill not found: ${name}` });
      return;
    }

    const metadata = this.buildMetadata(req);
    const execCtx = new ExecutionContext({ input: req.body, metadata, req, res, agent: this });

    return ExecutionContext.run(execCtx, () => {
      try {
        const ctx = new SkillContext({
          input: req.body,
          executionId: metadata.executionId,
          sessionId: metadata.sessionId,
          workflowId: metadata.workflowId,
          req,
          res,
          agent: this,
          memory: this.getMemoryInterface(metadata)
        });

        const result = skill.handler(ctx);
        res.json(result);
      } catch (err: any) {
        res.status(500).json({ error: err?.message ?? 'Execution failed' });
      }
    });
  }

  private buildMetadata(req: express.Request) {
    return {
      executionId: (req.headers['x-execution-id'] as string) ?? randomUUID(),
      runId: req.headers['x-run-id'] as string | undefined,
      sessionId: req.headers['x-session-id'] as string | undefined,
      actorId: req.headers['x-actor-id'] as string | undefined,
      workflowId: req.headers['x-workflow-id'] as string | undefined,
      parentExecutionId: req.headers['x-parent-execution-id'] as string | undefined,
      callerDid: req.headers['x-caller-did'] as string | undefined,
      targetDid: req.headers['x-target-did'] as string | undefined,
      agentNodeDid: req.headers['x-agent-did'] as string | undefined
    };
  }

  private async registerWithControlPlane() {
    try {
      const reasoners = this.reasoners.all().map((r) => ({
        id: r.name,
        input_schema: r.options?.inputSchema ?? {},
        output_schema: r.options?.outputSchema ?? {},
        memory_config: r.options?.memoryConfig ?? {
          auto_inject: [],
          memory_retention: '',
          cache_results: false
        },
        tags: r.options?.tags ?? []
      }));

      const skills = this.skills.all().map((s) => ({
        id: s.name,
        input_schema: s.options?.inputSchema ?? {},
        tags: s.options?.tags ?? []
      }));

      const port = this.config.port ?? 8001;
      const hostForUrl = this.config.publicUrl
        ? undefined
        : (this.config.host && this.config.host !== '0.0.0.0' ? this.config.host : '127.0.0.1');
      const publicUrl =
        this.config.publicUrl ?? `http://${hostForUrl ?? '127.0.0.1'}:${port}`;

      await this.agentFieldClient.register({
        nodeId: this.config.nodeId,
        version: this.config.version,
        reasoners,
        skills,
        baseUrl: publicUrl,
        publicUrl
      });
    } catch (err) {
      if (!this.config.devMode) {
        throw err;
      }
      console.warn('Control plane registration failed (devMode=true), continuing locally', err);
    }
  }

  private startHeartbeat() {
    const interval = this.config.heartbeatIntervalMs ?? 30_000;
    if (interval <= 0) return;

    const tick = async () => {
      try {
        await this.agentFieldClient.heartbeat();
      } catch (err) {
        if (!this.config.devMode) {
          console.warn('Heartbeat failed', err);
        }
      }
    };

    this.heartbeatTimer = setInterval(tick, interval);
    tick();
  }

  private health(): HealthStatus {
    return {
      status: 'ok',
      nodeId: this.config.nodeId,
      version: this.config.version
    };
  }

  private dispatchMemoryEvent(event: MemoryChangeEvent) {
    this.memoryWatchers.forEach(({ pattern, handler }) => {
      if (matchesPattern(pattern, event.key)) {
        handler(event);
      }
    });
  }

  private parseTarget(target: string): { agentId?: string; name: string } {
    if (!target.includes('.')) {
      return { name: target };
    }
    const [agentId, remainder] = target.split('.', 2);
    const name = remainder.replace(':', '/');
    return { agentId, name };
  }
}
