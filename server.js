const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };

  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');

  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.2': 'z-ai/glm-5.2',
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'm2.7': 'minimaxai/minimax-m2.7',
  'm3': 'minimaxai/minimax-m3',
  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

// ─── Reasoning Subsystem ─────────────────────────────────────────────────────
// Pure, stateful string parser for extracting reasoning blocks across chunks.

class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        // Check for partial tag at the end
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }

        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }
    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) {
        reasoning += this.buffer;
      } else {
        content += this.buffer;
      }
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

// Normalizes structured reasoning fields and extracts content delimiters.
class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;

    // ONLY use content delimiters for models that embed reasoning in content
    if (model === 'qwen/qwen3.5-397b-a17b' || model === 'nvidia/llama-3.3-nemotron-super-49b-v1.5') {
      this.parser = new DelimiterParser('<think>', '</think>');
    }
    // Models like Gemma 4, DeepSeek, GPT-OSS use structured fields and are NOT parsed here.
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    // Priority: Structured reasoning > Content delimiters
    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;

  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    if (model === 'qwen/qwen3.5-397b-a17b' || model === 'nvidia/llama-3.3-nemotron-super-49b-v1.5') {
      parser = new DelimiterParser('<think>', '</think>');
    }

    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

// Pure function returning model-specific reasoning request payloads.
// IMPORTANT: everything returned here gets spread DIRECTLY into the top-level
// JSON body sent to NIM via axios. Do NOT wrap anything in an `extra_body` key —
// that's an openai-SDK-only convention this proxy doesn't use, and NIM's raw
// REST endpoint will just silently ignore a field called "extra_body".
function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  const effort = clientReasoningEffort;

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: true } };
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      // Unverified param — see header comment. Left as opt-in best-effort.
      if (hasTools) payload.chat_template_kwargs.force_nonempty_content = true;
      return payload;
    }

    case 'qwen/qwen3.5-397b-a17b': {
      // Model appears to default to thinking-on in its chat template. Only send
      // a field when the caller explicitly wants thinking OFF; otherwise let the
      // <think> delimiter parser handle whatever the model does natively.
      if (enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: false } };
    }

    case 'deepseek-ai/deepseek-v4-pro':
    case 'deepseek-ai/deepseek-v4-flash': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { thinking: true } };
      if (effort) payload.chat_template_kwargs.reasoning_effort = effort;
      return payload;
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort && ['low', 'medium', 'high'].includes(effort)) {
        return { reasoning_effort: effort };
      }
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'mistralai/mistral-medium-3.5-128b':
    case 'mistralai/mistral-small-4-119b-2603': {
      if (effort && ['high', 'none'].includes(effort)) {
        return { reasoning_effort: effort };
      }
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'z-ai/glm-5.2': {
      // FIX: GLM-5.2 thinks by default. `reasoning_effort` only controls
      // intensity (max vs high) once thinking is already happening — it does
      // NOT turn thinking off. The actual on/off switch is `thinking.type`.
      // Without this, GLM-5.2 was silently reasoning on every single request
      // regardless of ENABLE_THINKING_MODE.
      const payload = {
        thinking: { type: enableThinking ? 'enabled' : 'disabled' }
      };
      if (enableThinking && effort) payload.reasoning_effort = effort;
      return payload;
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: true } };
    }

    case 'stepfun-ai/step-3.7-flash': {
      if (enableThinking) return {};
      return { chat_template_kwargs: { thinking: false } };
    }

    default:
      // Default reasoning models (Kimi, MiniMax, etc.) or non-reasoning models
      return {};
  }
}

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// FIX: Extract token AFTER "Bearer " prefix, compare only the token
function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);

  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];

    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callModel(baseRequest, model, enableThinking, clientReasoningEffort, hasTools) {
  const reasoningPayload = getReasoningPayload(
    model,
    enableThinking,
    clientReasoningEffort,
    hasTools
  );

  const res = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    { ...baseRequest, model, ...reasoningPayload },
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: baseRequest.stream ? 'stream' : 'json',
      timeout: REQUEST_TIMEOUT_MS
    }
  );

  return { response: res, model };
}

// ─────────────────────────────────────────────────────────────────────────────
// FastCRW Web Search
// ─────────────────────────────────────────────────────────────────────────────

const FASTCRW_API_URL =
  process.env.FASTCRW_API_URL || 'https://api.fastcrw.com';

const FASTCRW_API_KEY = process.env.FASTCRW_API_KEY;

const WEB_SEARCH_ENABLED =
  process.env.WEB_SEARCH_ENABLED !== 'false';

const WEB_SEARCH_LIMIT = Math.min(
  Math.max(parseInt(process.env.WEB_SEARCH_LIMIT || '5', 10), 1),
  10
);

const WEB_SEARCH_TIMEOUT_MS = Math.min(
  Math.max(parseInt(process.env.WEB_SEARCH_TIMEOUT_MS || '8000', 10), 1000),
  15000
);

// Use a small/cheap model for routing.
// Ideally point this at your fastest NIM model.
const SEARCH_ROUTER_MODEL =
  process.env.SEARCH_ROUTER_MODEL ||
  'nvidia/nemotron-3-super-120b-a12b';


// ─────────────────────────────────────────────────────────────────────────────
// Cheap local classifier
// ─────────────────────────────────────────────────────────────────────────────

function obviousNoSearch(text) {
  if (!text) return true;

  const t = text.toLowerCase().trim();

  // Requests that normally don't benefit from live search.
  const noSearchPatterns = [
    /^translate\b/,
    /^rewrite\b/,
    /^rephrase\b/,
    /^summari[sz]e\b/,
    /^fix (this|the)\b/,
    /^correct (this|the)\b/,
    /^proofread\b/,
    /^write (a|an|me)\b/,
    /^create (a|an|me)\b/,
    /^generate (a|an|me)\b/,
    /^explain\b/,
    /^what does .* mean\b/,
    /^how do .* work\b/,
    /^solve\b/,
    /^calculate\b/,
    /^convert\b/,
    /^give me .* example/,
    /^make .* code/,
    /^write .* code/
  ];

  return noSearchPatterns.some(pattern => pattern.test(t));
}


function likelyNeedsSearch(text) {
  if (!text) return false;

  const t = text.toLowerCase();

  const searchSignals = [
    /\b(latest|newest|recent|current|today|tonight|tomorrow)\b/,
    /\b(right now|currently|this week|this month)\b/,
    /\b(news|breaking)\b/,
    /\b(price|pricing|cost|stock|market cap)\b/,
    /\b(weather|forecast|temperature)\b/,
    /\b(score|scores|standings|schedule|game tonight)\b/,
    /\b(availability|available now|open now)\b/,
    /\b(version|release|released|changelog|documentation|docs|api)\b/,
    /\b(2026|2027)\b/,
    /\b(who is|what happened to)\b/,
    /\b(search|look up|find online|google)\b/,
    /\b(source|sources|article|website|web page|url)\b/,
    /\b(is .* still|does .* still|has .* changed)\b/
  ];

  return searchSignals.some(pattern => pattern.test(t));
}


// ─────────────────────────────────────────────────────────────────────────────
// Extract only useful conversation context
// ─────────────────────────────────────────────────────────────────────────────

function getSearchContext(messages) {
  const usable = messages
    .filter(m =>
      (m.role === 'user' || m.role === 'assistant') &&
      typeof m.content === 'string'
    )
    .slice(-4);

  return usable
    .map(m => `${m.role}: ${m.content}`)
    .join('\n');
}


// ─────────────────────────────────────────────────────────────────────────────
// LLM search router
// ─────────────────────────────────────────────────────────────────────────────

async function shouldSearchWeb(messages) {
  if (!WEB_SEARCH_ENABLED) {
    return {
      search: false,
      query: '',
      reason: 'disabled'
    };
  }

  const latestUser = [...messages]
    .reverse()
    .find(m =>
      m.role === 'user' &&
      typeof m.content === 'string'
    );

  const latestText = latestUser?.content?.trim() || '';

  if (!latestText) {
    return {
      search: false,
      query: '',
      reason: 'no_user_message'
    };
  }

  // Fast local rejection.
  if (obviousNoSearch(latestText)) {
    return {
      search: false,
      query: '',
      reason: 'obvious_no_search'
    };
  }

  // Fast local acceptance.
  if (likelyNeedsSearch(latestText)) {
    return {
      search: true,
      query: latestText,
      reason: 'local_search_signal'
    };
  }

  // Only ambiguous cases reach the LLM router.
  const context = getSearchContext(messages);

  const routerRequest = {
    model: SEARCH_ROUTER_MODEL,

    messages: [
      {
        role: 'system',
        content: `
You are a web-search router.

Decide whether the user's request requires CURRENT or externally
verified information.

Return ONLY JSON:

{
  "search": true,
  "query": "short search query"
}

or

{
  "search": false,
  "query": ""
}

Search for:
- current/latest information
- recent news
- live prices
- current software/API documentation
- current company/person information
- current sports/weather/schedules
- specific websites/articles/sources
- facts likely to have changed

Do not search for:
- creative writing
- rewriting
- translation
- summarization
- ordinary coding help
- stable general knowledge
`
      },
      {
        role: 'user',
        content: context
      }
    ],

    temperature: 0,
    max_tokens: 120,
    stream: false
  };

  try {
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      routerRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: 5000
      }
    );

    const text =
      response.data?.choices?.[0]?.message?.content || '';

    const cleaned = text
      .replace(/```json/gi, '')
      .replace(/```/g, '')
      .trim();

    const decision = JSON.parse(cleaned);

    return {
      search: decision.search === true,
      query:
        typeof decision.query === 'string'
          ? decision.query.trim()
          : '',
      reason: 'llm_router'
    };

  } catch (err) {
    console.warn(
      '[SEARCH ROUTER] failed:',
      err.response?.data || err.message
    );

    return {
      search: false,
      query: '',
      reason: 'router_failed'
    };
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// FastCRW request
// ─────────────────────────────────────────────────────────────────────────────

async function searchWeb(query, options = {}) {
  if (!WEB_SEARCH_ENABLED || !FASTCRW_API_KEY || !query) {
    return [];
  }

  const controller = new AbortController();

  const timeout = setTimeout(() => {
    controller.abort();
  }, WEB_SEARCH_TIMEOUT_MS);

  try {
    const response = await axios.post(
      `${FASTCRW_API_URL}/v1/search`,
      {
        query: query.slice(0, 1000),
        limit: options.limit || WEB_SEARCH_LIMIT,
        lang: options.lang || 'en',
        sources: options.sources || ['web']
      },
      {
        headers: {
          Authorization: `Bearer ${FASTCRW_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: WEB_SEARCH_TIMEOUT_MS,
        signal: controller.signal
      }
    );

    const data = response.data?.data;

    const results = Array.isArray(data)
      ? data
      : Array.isArray(data?.results)
        ? data.results
        : [];

    return results
      .map(r => ({
        title: String(r.title || ''),
        url: String(r.url || ''),
        snippet: String(
          r.snippet ||
          r.description ||
          ''
        ),
        score: Number.isFinite(r.score)
          ? r.score
          : undefined
      }))
      .filter(r => r.title || r.url || r.snippet)
      .slice(0, WEB_SEARCH_LIMIT);

  } catch (err) {
    console.warn(
      '[WEB SEARCH] failed:',
      err.name === 'CanceledError'
        ? 'timeout'
        : err.response?.data || err.message
    );

    return [];

  } finally {
    clearTimeout(timeout);
  }
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.2.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    const primaryModel = MODEL_MAPPING[model];

if (!primaryModel) {
  return res.status(400).json({
    error: {
      message: `Unknown model: ${model}`,
      type: 'invalid_request_error',
      code: 400
    }
  });
}

let finalMessages = messages;
let webSearchUsed = false;

// ─── Automatic Web Search ───────────────────────────────────────────────

if (likelyNeedsSearch(latestText)) {
    return {
        search: true,
        query: latestText,
        reason: 'local_search_signal'
    };
}
    
if (WEB_SEARCH_ENABLED) {

let finalMessages = messages;
let webSearchUsed = false;

const decision = await shouldSearchWeb(messages);

console.log(
  `[SEARCH ROUTER] search=${decision.search} ` +
  `reason=${decision.reason} ` +
  `query="${decision.query}"`
);

if (decision.search && decision.query) {
  const results = await searchWeb(decision.query);

  if (results.length > 0) {
    webSearchUsed = true;

    const searchContext = results
      .map((r, i) => [
        `[SOURCE ${i + 1}]`,
        `Title: ${r.title}`,
        `URL: ${r.url}`,
        `Snippet: ${r.snippet}`
      ].join('\n'))
      .join('\n\n');

    finalMessages = [
      {
        role: 'system',
        content: `
You have access to live web-search results below.

Use them when answering questions requiring current information.

Rules:
- Treat the results as external evidence.
- Prefer them over stale internal knowledge for current facts.
- Do not claim to have opened or read a webpage unless its contents were actually provided.
- Do not invent facts that are absent from the results.
- If the sources conflict, explain the uncertainty.
- Cite sources by URL when appropriate.
`
      },

      ...messages,

      {
        role: 'user',
        content: `
LIVE WEB SEARCH RESULTS:

${searchContext}

Use these results to answer the preceding request when relevant.
`
      }
    ];
  }
}

const baseRequest = {
  messages: finalMessages,
  temperature: temperature ?? 0.7,
  max_tokens: Math.min(
    max_tokens ?? 2048,
    MAX_TOKENS_LIMIT
  ),
  stream: stream || false
};

const { response, model: usedModel } = await callModel(
  baseRequest,
  primaryModel,
  ENABLE_THINKING_MODE,
  req.body.reasoning_effort,
  !!req.body.tools
);
  
    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    // Determine if the client wants legacy inline <thinking> tags in the content stream
    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const normalizer = new StreamNormalizer(usedModel);

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            const normalizedDelta = normalizer.processDelta(delta);
            let clientContent = '';

            if (SHOW_REASONING && inlineReasoning) {
              // Legacy GoonChat behavior: bake <thinking> tags into content
              if (normalizedDelta.reasoning && !reasoningOpen) {
                clientContent += `<thinking>\n${normalizedDelta.reasoning}`;
                reasoningOpen = true;
              } else if (normalizedDelta.reasoning) {
                clientContent += normalizedDelta.reasoning;
              }

              if (normalizedDelta.content && reasoningOpen) {
                clientContent += `\n</thinking>\n\n${normalizedDelta.content}`;
                reasoningOpen = false;
              } else if (normalizedDelta.content) {
                clientContent += normalizedDelta.content;
              }
            } else {
              // Default behavior: clean content, no inline tags
              clientContent = normalizedDelta.content || '';
            }

            delta.content = clientContent;

            // FIX: keep a structured reasoning field alongside the inline
            // tags in content. GoonChat parses the inline tags;
            // clients like Pal Chat / OpenRouter-style apps look for a
            // separate `reasoning`/`reasoning_content` field to render their
            // own collapsible thinking UI. Without this, those clients just
            // see one flat content blob and never show a thinking indicator.
            if (SHOW_REASONING && normalizedDelta.reasoning) {
              delta.reasoning = normalizedDelta.reasoning;
              delta.reasoning_content = normalizedDelta.reasoning;
            } else {
              delete delta.reasoning;
              delete delta.reasoning_content;
            }
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Upstream sent malformed chunk',
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            }
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

      buffer = buffer.replace(/\r\n/g, '\n');

const lines = buffer.split('\n');
buffer = lines.pop() || '';


const MAX_LINE_SIZE = 256 * 1024;
        
        if (buffer.length > MAX_LINE_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream buffer overflow',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();

        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        const flushedDelta = normalizer.flush();
        if (flushedDelta.content || flushedDelta.reasoning) {
          let clientContent = '';
          if (SHOW_REASONING && inlineReasoning) {
            // Legacy GoonChat behavior: bake <thinking> tags into content
            if (flushedDelta.reasoning && !reasoningOpen) {
              clientContent += `<thinking>\n${flushedDelta.reasoning}`;
              reasoningOpen = true;
            } else if (flushedDelta.reasoning) {
              clientContent += flushedDelta.reasoning;
            }
            if (flushedDelta.content && reasoningOpen) {
              clientContent += `\n</thinking>\n\n${flushedDelta.content}`;
              reasoningOpen = false;
            } else if (flushedDelta.content) {
              clientContent += flushedDelta.content;
            }
          } else {
            // Default behavior: clean content, no inline tags
            clientContent = flushedDelta.content || '';
          }
          if (clientContent) {
            safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: clientContent } }] })}\n\n`);
          }
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);

        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;

        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }

        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          let content = normalizedChoice.message?.content || '';
          const reasoning = normalizedChoice.message?.reasoning || '';

          if (SHOW_REASONING && inlineReasoning && reasoning) {
            // Legacy GoonChat behavior: bake <thinking> tags into content
            content = `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
          }

          const finalMessage = { ...normalizedChoice.message, content };

          // Same fix as the streaming path: keep the structured field
          // alongside the inline tags so structured-reasoning clients
          // (Pal Chat, OpenRouter-style apps) can render their own UI.
          if (SHOW_REASONING && reasoning) {
            finalMessage.reasoning = reasoning;
            finalMessage.reasoning_content = reasoning;
          } else {
            delete finalMessage.reasoning;
            delete finalMessage.reasoning_content;
          }

          const finalChoice = {
            ...normalizedChoice,
            index: i,
            message: finalMessage
          };
          return finalChoice;
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);

  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});


//—— JudgeSearchman ——————————————————————————————————————————————————————————

async function shouldSearchWeb(messages, model) {
  const recentMessages = messages
    .slice(-6)
    .map(m => `${m.role}: ${m.content}`)
    .join('\n');
    .filter(m => m.role === 'user' || m.role === 'assistant')
}
  const decision = await shouldSearchWeb(
  getSearchableMessages(messages),
  primaryModel
);

  const routerRequest = {
    model,
    messages: [
      {
        role: 'system',
        content: `
You are a web-search routing classifier.

Determine whether answering the user's latest request would materially
benefit from CURRENT or EXTERNAL web information.

Search when the user asks for things such as:
- current/latest/recent information
- news
- today's information
- live prices, scores, weather, availability or schedules
- current software/library/API information
- current company/person information
- facts that should be verified against the web
- specific websites, articles, pages, documentation, or sources
- information that may have changed since your training


Do NOT search for:
- creative writing
- casual conversation
- rewriting
- summarization of text the user already supplied
- general stable knowledge
- coding questions that can be answered without current documentation

Return ONLY valid JSON:

{
  "search": true|false,
  "query": "best concise web search query",
  "reason": "short explanation"
}

If searching is unnecessary, query must be "".
`
      },
      {
        role: 'user',
        content: recentMessages
      }
    ],
    temperature: 0,
    max_tokens: 300,
    stream: false
  };

  try {
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      routerRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: 10000
      }
    );

    const text =
      response.data?.choices?.[0]?.message?.content || '';

    // Handle accidental markdown fences
    const cleaned = text
      .replace(/^```json\s*/i, '')
      .replace(/^```\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim();

    const decision = JSON.parse(cleaned);

    return {
      search: decision.search === true,
      query: typeof decision.query === 'string'
        ? decision.query.trim()
        : '',
      reason: decision.reason || ''
    };

  } catch (err) {
    console.warn(
      '[SEARCH ROUTER] Failed:',
      err.response?.data || err.message
    );

    // Fail closed.
    return {
      search: false,
      query: '',
      reason: 'router_failed'
    };
  }
}



const SEARCH_CACHE = new Map();

const SEARCH_CACHE_TTL_MS =
  Number(process.env.SEARCH_CACHE_TTL_MS || 60_000);

function normalizeSearchKey(query) {
  return query
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

async function cachedSearchWeb(query, options = {}) {
  const key = normalizeSearchKey(query);

  const cached = SEARCH_CACHE.get(key);

  if (cached && cached.expires > Date.now()) {
    return cached.results;
  }

  const results = await searchWeb(query, options);

  SEARCH_CACHE.set(key, {
    results,
    expires: Date.now() + SEARCH_CACHE_TTL_MS
  });

  // Prevent unlimited memory growth.
  if (SEARCH_CACHE.size > 500) {
    const oldest = SEARCH_CACHE.keys().next().value;
    SEARCH_CACHE.delete(oldest);
  }

  return results;
}

const results = await cachedSearchWeb(decision.query);

const ttl =
  queryLooksLikeNews(query) ? 30_000 :
  queryLooksLikePrice(query) ? 15_000 :
  120_000;


if (stream) {
  res.status(200);

  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');

  if (typeof res.flushHeaders === 'function') {
    res.flushHeaders();
  }
}

safeWrite(res, ': connected\n\n');
