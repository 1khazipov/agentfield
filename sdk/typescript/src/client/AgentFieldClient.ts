import axios, { AxiosInstance } from 'axios';
import type { AgentConfig, HealthStatus } from '../types/agent.js';

export class AgentFieldClient {
  private readonly http: AxiosInstance;
  private readonly config: AgentConfig;

  constructor(config: AgentConfig) {
    const baseURL = (config.agentFieldUrl ?? 'http://localhost:8080').replace(/\/$/, '');
    this.http = axios.create({ baseURL });
    this.config = config;
  }

  async register(payload: any) {
    await this.http.post('/api/v1/nodes/register', payload);
  }

  async heartbeat(): Promise<HealthStatus> {
    const res = await this.http.get('/api/v1/nodes/heartbeat', {
      params: { nodeId: this.config.nodeId }
    });
    return res.data as HealthStatus;
  }

  async execute<T = any>(target: string, input: any): Promise<T> {
    const res = await this.http.post(`/api/v1/execute/${target}`, {
      input
    });
    return (res.data?.result as T) ?? res.data;
  }
}
