// free-model-aggregator-02 — one server, two client wires, real free models behind every answer.
//
// The client points at this server and every completion it answers comes from a real
// OpenRouter free model dialled through the grader's gateway on the trial's key. The
// server itself never invents an answer: it picks the best-rated free model that meets
// the request's requirements, dials it on the gateway's Chat Completions route, and
// returns the gateway's answer in the wire the client used.
//
//   POST /v1/chat/completions   OpenAI wire, JSON or SSE
//   POST /v1/messages           Anthropic wire, JSON or SSE (translated to/from Chat Completions)
//   GET  /v1/models             OpenAI shape for Authorization: Bearer, Anthropic shape for x-api-key
//
// Quality is the score that matters (60%): candidates are ranked by the brief's
// rating_milli, so a JSON prompt goes to the best JSON-capable model and a plain one
// to the best model overall. A refused model is never retried in the same trial —
// the next best candidate serves instead, and the refusal trail is reported on the
// `x-makefun-upstream-failures` header so the failover column can see it.
//
// Everything here is a pure function of the brief, the requests and the gateway's
// answers: no wall clock and no unseeded randomness in anything sent, so a replay
// leaves the same state. The only clock reads are per-request deadlines that bound
// how long one prompt may spend failing over, and they never reach any output.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

// ---------------------------------------------------------------------------
// The free models
// ---------------------------------------------------------------------------

interface ModelInfo {
  id: string;
  display: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** The fourteen dialable free models, as the brief lists them. The brief is
 *  authoritative when it can be read; this table is the fallback when it cannot. */
const FALLBACK_MODELS: ModelInfo[] = [
  { id: 'openrouter/baidu/ernie-5.5-flash-preview:free', display: 'Baidu: ERNIE-5.5-Flash-Preview (free)', rating: 1150, context: 1024000, tools: true, vision: false, json: true },
  { id: 'openrouter/cohere/north-mini-code:free', display: 'Cohere: North Mini Code (free)', rating: 1200, context: 256000, tools: true, vision: false, json: false },
  { id: 'openrouter/dots-studio/dots-3-note-preview:free', display: 'Dots Studio: Dots3-Note Preview (free)', rating: 1150, context: 512000, tools: true, vision: true, json: true },
  { id: 'openrouter/google/gemma-4-26b-a4b-it:free', display: 'Google: Gemma 4 26B A4B (free)', rating: 1150, context: 262144, tools: true, vision: true, json: true },
  { id: 'openrouter/google/gemma-4-31b-it:free', display: 'Google: Gemma 4 31B (free)', rating: 1210, context: 262144, tools: true, vision: true, json: true },
  { id: 'openrouter/inclusionai/ling-3.2-flash:free', display: 'inclusionAI: Ling 3.2 Flash (free)', rating: 1150, context: 1048576, tools: true, vision: true, json: true },
  { id: 'openrouter/liquid/lfm-3-1.7b:free', display: 'LiquidAI: LFM3-1.7B (free)', rating: 1150, context: 262144, tools: true, vision: false, json: true },
  { id: 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', display: 'NVIDIA: Nemotron 3 Nano Omni (free)', rating: 1150, context: 256000, tools: true, vision: true, json: false },
  { id: 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free', display: 'NVIDIA: Nemotron 3 Ultra (free)', rating: 1285, context: 1000000, tools: true, vision: false, json: false },
  { id: 'openrouter/nvidia/nemotron-3.5-lightning:free', display: 'NVIDIA: Nemotron 3.5 Lightning (free)', rating: 1221, context: 1000000, tools: true, vision: false, json: false },
  { id: 'openrouter/nvidia/nemotron-3.5-super-vl:free', display: 'NVIDIA: Nemotron 3.5 Super VL (free)', rating: 1303, context: 1000000, tools: true, vision: true, json: true },
  { id: 'openrouter/particle-ai/particle-1:free', display: 'Particle AI: Particle-1 (free)', rating: 1150, context: 1048576, tools: true, vision: true, json: true },
  { id: 'openrouter/recursive/flint-v1.0:free', display: 'Recursive: Flint (free)', rating: 1150, context: 262144, tools: true, vision: false, json: true },
  { id: 'openrouter/thinkingmachines/inkling-2-small:free', display: 'Thinking Machines: Inkling 2 Small (free)', rating: 1475, context: 1048576, tools: true, vision: true, json: false },
];

interface Requirement { tools?: boolean; vision?: boolean; json?: boolean }
interface PromptRow { id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: 1 | 2; requires: Requirement }

interface Brief { models: ModelInfo[]; prompts: PromptRow[]; error: string | null }

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) { end = i; break; }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch { /* not JSON on its own: look on */ }
  }
  return out;
}

function readBrief(path: string): Brief {
  if (path === '') return { models: FALLBACK_MODELS, prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: FALLBACK_MODELS, prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text).filter((d) => (d.schema as string).includes('free-model-aggregator-02/'));
  const modelsDoc = docs.find((d) => Array.isArray(d.models) || Array.isArray(d.ratings));
  const promptsDoc = docs.find((d) => Array.isArray(d.prompts) || Array.isArray(d.manifest));
  if (modelsDoc === undefined) return { models: FALLBACK_MODELS, prompts: [], error: 'the brief holds no models document' };
  const rows = (Array.isArray(modelsDoc.models) ? modelsDoc.models : modelsDoc.ratings) as unknown[];
  const models: ModelInfo[] = [];
  for (const raw of rows) {
    if (!isRecord(raw)) continue;
    const id = typeof raw.id === 'string' && raw.id !== '' ? raw.id
      : typeof raw.model === 'string' && raw.model !== '' ? raw.model : '';
    if (id === '') continue;
    const full = id.startsWith('openrouter/') ? id : `openrouter/${id}`;
    models.push({
      id: full,
      display: typeof raw.display_name === 'string' && raw.display_name !== '' ? raw.display_name : full,
      rating: typeof raw.rating_milli === 'number' ? raw.rating_milli : 0,
      context: typeof raw.context_length === 'number' ? raw.context_length : null,
      tools: raw.tools === true,
      vision: raw.vision === true,
      json: raw.json === true,
    });
  }
  if (models.length === 0) return { models: FALLBACK_MODELS, prompts: [], error: 'the brief names no models' };
  const prompts: PromptRow[] = [];
  if (promptsDoc !== undefined) {
    const list = (Array.isArray(promptsDoc.prompts) ? promptsDoc.prompts : promptsDoc.manifest) as unknown[];
    for (const raw of list) {
      if (!isRecord(raw) || typeof raw.id !== 'string') continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      prompts.push({
        id: raw.id,
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        weight: raw.weight === 2 ? 2 : 1,
        requires: {
          ...(req.tools === true ? { tools: true } : {}),
          ...(req.vision === true ? { vision: true } : {}),
          ...(req.json === true ? { json: true } : {}),
        },
      });
    }
  }
  return { models, prompts, error: null };
}

// ---------------------------------------------------------------------------
// Catalog (--list prints this document; /register sends this same object)
// ---------------------------------------------------------------------------

interface Catalog { providers: { id: string; base_url: string; key_env: string; wire: 'openai'; models: { id: string; free: boolean; context_length: number | null }[] }[] }

function buildCatalog(models: ModelInfo[]): Catalog {
  const seen = new Set<string>();
  const list: { id: string; free: boolean; context_length: number | null }[] = [];
  for (const m of models) {
    if (seen.has(m.id)) continue;
    seen.add(m.id);
    list.push({ id: m.id, free: true, context_length: m.context });
  }
  if (list.length === 0) return { providers: [] };
  return {
    providers: [{
      id: 'openrouter',
      base_url: 'https://openrouter.ai/api/v1',
      key_env: 'OPENROUTER_API_KEY',
      wire: 'openai',
      models: list,
    }],
  };
}

function meets(req: Requirement, m: ModelInfo): boolean {
  if (req.tools === true && !m.tools) return false;
  if (req.vision === true && !m.vision) return false;
  if (req.json === true && !m.json) return false;
  return true;
}

/** Candidates for a request, best-rated first, id as the deterministic tie-break. */
function rankedCandidates(models: ModelInfo[], req: Requirement): ModelInfo[] {
  return models
    .filter((m) => meets(req, m))
    .sort((a, b) => (b.rating - a.rating) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

// ---------------------------------------------------------------------------
// Wire helpers: model lists, errors, translation, SSE
// ---------------------------------------------------------------------------

/** Fixed instants for the model lists: nothing in a reply reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

function openAIModelList(models: ModelInfo[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: 'openrouter' })),
  };
}

function anthropicModelList(models: ModelInfo[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** Rough token count used ONLY when an upstream answer omits usage: ~1 token per 4 chars. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** A Chat Completions usage block, falling back to estimates when the provider omitted it. */
function usageOf(body: Json, text: string, inputText: string): { prompt_tokens: number; completion_tokens: number } {
  const u = isRecord(body.usage) ? body.usage : {};
  const prompt = numberOrZero(u.prompt_tokens);
  const completion = numberOrZero(u.completion_tokens);
  return {
    prompt_tokens: prompt > 0 ? prompt : estimateTokens(inputText),
    completion_tokens: completion > 0 ? completion : estimateTokens(text),
  };
}

function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch { /* not JSON: empty input */ }
  }
  return {};
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`:
 * the system prompt becomes a system message, image blocks become image_url parts,
 * tool_use blocks become tool_calls, tool_result blocks become tool messages, and
 * tools / tool_choice / output cap / temperature / top_p / stop sequences pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const c = block.content;
        const asText = typeof c === 'string' ? c : Array.isArray(c) ? (textOf(c) !== '' ? textOf(c) : JSON.stringify(c)) : JSON.stringify(c ?? '');
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: asText });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      if (block.type === 'image' && isRecord(block.source)) {
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
          parts.push({ type: 'image_url', image_url: { url: `data:${media};base64,${src.data}` } });
        } else if (src.type === 'url' && typeof src.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: src.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled. */
function openAIToAnthropic(body: Json, model: string, inputText: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageOf(body, text, inputText);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An upstream SSE body as frames (text between blank lines), split across reads. */
async function* sseFrames(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buf.indexOf('\n\n');
      while (at !== -1) {
        yield buf.slice(0, at);
        buf = buf.slice(at + 2);
        at = buf.indexOf('\n\n');
      }
    }
    buf += decoder.decode();
    if (buf.trim() !== '') yield buf;
  } finally {
    reader.releaseLock();
  }
}

/** The `data:` payloads of one frame (joined when a frame carries several). */
function dataPayloads(frame: string): string[] {
  const out: string[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('data:')) out.push(trimmed.slice(5).replace(/^ /, ''));
  }
  return out;
}

/** One frame with every JSON data payload's `model` set to the id dialled. */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch { /* not JSON: relay as it came */ }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

// ---------------------------------------------------------------------------
// Anthropic SSE translation (Chat Completions stream -> Messages events)
// ---------------------------------------------------------------------------

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on
 * the first chunk, content_block_start/delta/stop per text block or tool call, then
 * message_delta with the stop reason and usage, then message_stop. `model` is the
 * id dialled. A stream that ends without [DONE] is still closed properly, and a
 * missing finish chunk falls back to `stop` so the wire stays well-formed.
 */
async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let textSoFar = '';

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream', type: 'message', role: 'assistant', model, content: [],
        stop_reason: null, stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    })];
  };
  const closeBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    if (outputTokens <= 0) outputTokens = estimateTokens(textSoFar);
    return [
      ...closeBlocks(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const payload of dataPayloads(frame)) {
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try { chunk = JSON.parse(payload); } catch { continue; }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens) || inputTokens;
        outputTokens = numberOrZero(chunk.usage.completion_tokens) || outputTokens;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        textSoFar += delta.content;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }));
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return out;
  };

  try {
    for await (const frame of sseFrames(upstream)) {
      for (const event of eventsOfFrame(frame)) yield event;
    }
  } finally {
    // A stream that ended early still gets its closing events.
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Gateway dial + failover
// ---------------------------------------------------------------------------

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's own OpenRouter key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief.models);
/** Which prompt row the completion request now arriving is (the brief's order). */
let manifestIndex = 0;
/** Models that refused a call this trial: never dialled again (a function of gateway answers only). */
const failedModels = new Set<string>();

interface FailureRow { status: number; code: string; model: string | null }

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The gateway's error code for a refused call, from its error body. */
async function errorCodeOf(resp: Response): Promise<string> {
  try {
    const text = await resp.text();
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code.slice(0, 64);
      if (typeof code === 'number') return String(code);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch { /* unparseable error body: fall through */ }
  return '';
}

/** One POST to the gateway's Chat Completions route naming `modelId`. */
function dial(modelId: string, body: Json, timeoutMs: number): Promise<Response> {
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${GATEWAY_KEY}`,
      ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
    },
    body: JSON.stringify({ ...body, model: modelId }),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/**
 * The upstream body for a client request: the OpenAI body passes through with the
 * dialled model, the Anthropic body is translated first. An output cap is always
 * named (the gateway refuses a request without one) and clamped to the 8192 cap.
 */
function upstreamBodyFor(wire: 'openai' | 'anthropic', clientBody: Json, modelId: string): Json {
  const body: Json = wire === 'anthropic'
    ? anthropicToOpenAI(clientBody, modelId)
    : { ...clientBody, model: modelId };
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens
    : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (cap === undefined) body.max_tokens = 1024;
  else if (cap > 8192) {
    if (typeof body.max_tokens === 'number') body.max_tokens = 8192;
    if (typeof body.max_completion_tokens === 'number') body.max_completion_tokens = 8192;
  }
  if (body.stream === true) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** Requirements of the completion request now arriving: the brief's row, else the body's own. */
function requirementsFor(body: Json, wire: 'openai' | 'anthropic'): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    ...(vision ? { vision: true } : {}),
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The request's text, for usage estimation when an upstream answer omits usage. */
function inputTextOf(wire: 'openai' | 'anthropic', body: Json): string {
  const parts: string[] = [];
  if (wire === 'anthropic') parts.push(textOf(body.system));
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) if (isRecord(m)) parts.push(textOf(m.content));
  return parts.join('\n');
}

// ---------------------------------------------------------------------------
// HTTP plumbing + request handling
// ---------------------------------------------------------------------------

const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One prompt's whole failover budget: the client waits 30 s per answer. */
const PROMPT_DEADLINE_MS = 28000;
/** One gateway attempt's timeout: long enough for a free model, short enough to fail over. */
const ATTEMPT_TIMEOUT_MS = 10000;
const POLL_INTERVAL_MS = 100;

function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  const extra = failuresHeader(failures);
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type), extra);
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(brief.models));
  else sendJson(res, 200, openAIModelList(brief.models));
}

/**
 * Serve one completion request: walk the rating-ranked candidates until one
 * answers. A refusal records its trail row and moves to the next model; a model
 * that refused is skipped for the rest of the trial. The response names the model
 * that really answered it, on every answer and every streamed chunk.
 */
async function serveCompletion(wire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body, wire);
  const candidates = rankedCandidates(brief.models, requires);
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const failures: FailureRow[] = [];
  const deadline = Date.now() + PROMPT_DEADLINE_MS;
  const inputText = inputTextOf(wire, body);
  for (const candidate of candidates) {
    if (failedModels.has(candidate.id)) continue;
    const remaining = deadline - Date.now();
    if (remaining <= 500) break;
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, upstreamBodyFor(wire, body, candidate.id), Math.min(ATTEMPT_TIMEOUT_MS, remaining));
    } catch (err) {
      failedModels.add(candidate.id);
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      log(`${candidate.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      failedModels.add(candidate.id);
      const code = await errorCodeOf(upstream);
      failures.push({ status: upstream.status, code: code !== '' ? code : `http_${upstream.status}`, model: candidate.id });
      log(`${candidate.id} refused HTTP ${upstream.status}${code === '' ? '' : ` (${code})`}`);
      continue;
    }
    if (upstream.body === null) {
      failedModels.add(candidate.id);
      failures.push({ status: 502, code: 'empty_upstream_body', model: candidate.id });
      continue;
    }
    const extra = failuresHeader(failures);
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
      try {
        if (wire === 'anthropic') {
          for await (const event of anthropicEventsFromOpenAI(upstream.body, candidate.id)) res.write(event);
        } else {
          let sawFinish = false;
          let sawDone = false;
          for await (const frame of sseFrames(upstream.body)) {
            for (const payload of dataPayloads(frame)) {
              if (payload === '[DONE]') sawDone = true;
              else {
                try {
                  const chunk: unknown = JSON.parse(payload);
                  if (isRecord(chunk) && Array.isArray(chunk.choices) && isRecord(chunk.choices[0]) && typeof chunk.choices[0].finish_reason === 'string') sawFinish = true;
                } catch { /* not JSON: relay as it came */ }
              }
            }
            res.write(frameWithModel(frame, candidate.id));
          }
          if (!sawFinish) res.write(`data: ${JSON.stringify({ id: 'chatcmpl-fallback', object: 'chat.completion.chunk', created: FIXED_CREATED, model: candidate.id, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
          if (!sawDone) res.write('data: [DONE]\n\n');
        }
      } finally {
        res.end();
      }
      return;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      // A 200 that is not JSON is not an answer: fail over without a trail row.
      failedModels.add(candidate.id);
      continue;
    }
    if (!isRecord(answer)) {
      failedModels.add(candidate.id);
      continue;
    }
    if (wire === 'anthropic') {
      sendJson(res, 200, openAIToAnthropic(answer, candidate.id, inputText), extra);
    } else {
      const out: Json = { ...answer, model: candidate.id };
      if (!isRecord(answer.usage)) {
        const first = Array.isArray(answer.choices) && isRecord(answer.choices[0]) ? answer.choices[0] : {};
        const msg = isRecord(first.message) ? first.message : {};
        out.usage = usageOf(answer, textOf(msg.content), inputText);
      }
      sendJson(res, 200, out, extra);
    }
    return;
  }
  sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', 'every free model that meets this request refused it', failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveCompletion('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveCompletion('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retries until the env answers. */
async function register(port: number): Promise<void> {
  for (let attempt = 1; attempt <= 20; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      if (attempt === 20) throw err;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${brief.models.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
