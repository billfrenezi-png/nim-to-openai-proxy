// server.js — Robust Hybrid OpenAI ↔ NIM Proxy
// Express 5 Compatible
// Fixes: auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations

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

const MAX_TOKENS_LIMIT = 85536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

//FASTCRWSUPPORT
 const FASTCRW_API_BASE =
  process.env.FASTCRW_API_BASE || 'https://api.fastcrw.com';

const FASTCRW_API_KEY = process.env.FASTCRW_API_KEY;
const ENABLE_WEB_SEARCH = process.env.ENABLE_WEB_SEARCH === 'true';

const MAX_TOOL_ROUNDS = Math.min(
  Math.max(parseInt(process.env.MAX_TOOL_ROUNDS || '2', 10), 1),
  5
);

const FASTCRW_TIMEOUT_MS = 30000;
// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };

  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');

  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }

  if (ENABLE_WEB_SEARCH && !FASTCRW_API_KEY) {
    fatal('FASTCRW_API_KEY is required when ENABLE_WEB_SEARCH=true');
  }

  if (ENABLE_WEB_SEARCH) {
    console.log('[CONFIG] Live web search: ENABLED');
    console.log(`[CONFIG] fastCRW: ${FASTCRW_API_BASE}`);
  }
}

validateConfig();


// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'z-ai/glm-5.2',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.2': 'z-ai/glm-5.2',
  'mistral': 'thinkingmachines/inkling',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
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

// ─── Middleware ────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// FIX: Extract token AFTER "Bearer " prefix, compare only the token
// Prevents bypass when CLIENT_AUTH_KEY is empty (expected would be "Bearer " which is 7 chars)
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

// FIX: Use lightweight model listing instead of burning inference quota
// If NIM doesn't support /models, skip validation entirely rather than DDoS-ing yourself
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

// FIX: Wrap res.write in try/catch to prevent crashes on closed sockets
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function postNIM(requestBody) {
  const MAX_RETRIES = 3;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        requestBody,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout: REQUEST_TIMEOUT_MS
        }
      );
    } catch (error)
    {
      const status = error.response?.status;

      if (status !== 429 || attempt === MAX_RETRIES) {
        throw error;
      }

      const delay =
        Math.min(1000 * Math.pow(2, attempt), 15000) +
        Math.floor(Math.random() * 500);

      console.warn(
        `[NIM] 429 rate limit. Retrying in ${delay}ms (attempt ${attempt + 1}/${MAX_RETRIES})`
      );

      await sleep(delay);
    }
  }
}

// ─── Routes ────────────────────────────────────────────────────────────────
const FASTCRW_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'search_web',
      description:
        'Search the live web only when current or externally verifiable information is actually needed. Use search for: - current real-world facts - recent events or news - current public figures - current products, games, movies, shows, companies, or software - factual details that the supplied conversation/context does not contain Do NOT search for: - creative writing - roleplay - fictional continuation - brainstorming - ordinary conversation - information already present in the conversation When searching, prefer a single focused search with 2-3 results.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The web search query.'
          },
          limit: {
            type: 'integer',
            description: 'Number of results to return. Prefer 2-3 for normal questions. Maximum 5.',
            minimum: 1,
            maximum: 8
          },
          time_range: {
            type: 'string',
            enum: ['hour', 'day', 'week', 'month', 'year', 'any'],
            description: 'Optional freshness filter.'
          }
        },
        required: ['query']
      }
    }
  }
];

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.1.0' });
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
  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    if (!Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: 'messages must be an array',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    const selectedModel =
      MODEL_MAPPING[model] ||
      'nvidia/llama-3.3-nemotron-super-49b-v1.5';

    console.log(
      `[PROXY] ${model || 'default'} → ${selectedModel}`
    );

    let nimResponse;

    if (ENABLE_WEB_SEARCH) {
      nimResponse = await runWithWebSearch({
        selectedModel,
        messages,
        temperature,
        max_tokens
      });
    } else {
      const requestBody = {
        model: selectedModel,
        messages,
        temperature: temperature ?? 0.7,
        max_tokens: Math.min(
          max_tokens ?? 2048,
          MAX_TOKENS_LIMIT
        ),
        stream: stream || false,
       
      };

      const response = await postNIM(requestBody);

      nimResponse = response.data;
    }

    /*
     * Normalize NIM response into your proxy's OpenAI-compatible shape.
     */
    const openaiResponse = {
      id:
        nimResponse.id ||
        `chatcmpl-${Date.now()}`,

      object: 'chat.completion',

      created:
        nimResponse.created ||
        Math.floor(Date.now() / 1000),

      model: model || selectedModel,

      choices: (nimResponse.choices || []).map(
        (choice, i) => {
          let content =
            choice.message?.content || '';

          if (
            SHOW_REASONING &&
            choice.message?.reasoning_content
          ) {
            const safeReasoning =
              choice.message.reasoning_content
                .replace(/\n/g, '\\n');

            content =
              `<thinking>\n${safeReasoning}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,

            message: {
              role:
                choice.message?.role ||
                'assistant',

              content,

              tool_calls:
                choice.message?.tool_calls
            },

            finish_reason:
              choice.finish_reason ||
              'stop'
          };
        }
      ),

      usage:
        nimResponse.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
    };

    /*
     * Normal JSON response.
     */
    if (!stream) {
      return res.json(openaiResponse);
    }

    /*
     * Streaming compatibility.
     *
     * The tool loop itself runs internally first. Once the final
     * answer is available, expose it as normal OpenAI SSE chunks.
     */
    res.setHeader(
      'Content-Type',
      'text/event-stream'
    );

    res.setHeader(
      'Cache-Control',
      'no-cache'
    );

    res.setHeader(
      'Connection',
      'keep-alive'
    );

    const choice =
      openaiResponse.choices?.[0];

    const content =
      choice?.message?.content || '';

    const chunkId = openaiResponse.id;

    /*
     * Initial role chunk.
     */
    safeWrite(
      res,
      `data: ${JSON.stringify({
        id: chunkId,
        object: 'chat.completion.chunk',
        created: openaiResponse.created,
        model: openaiResponse.model,
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant'
            },
            finish_reason: null
          }
        ]
      })}\n\n`
    );

    /*
     * Send final content.
     */
    if (content) {
      safeWrite(
        res,
        `data: ${JSON.stringify({
          id: chunkId,
          object: 'chat.completion.chunk',
          created: openaiResponse.created,
          model: openaiResponse.model,
          choices: [
            {
              index: 0,
              delta: {
                content
              },
              finish_reason: null
            }
          ]
        })}\n\n`
      );
    }

    /*
     * Finish.
     */
    safeWrite(
      res,
      `data: ${JSON.stringify({
        id: chunkId,
        object: 'chat.completion.chunk',
        created: openaiResponse.created,
        model: openaiResponse.model,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason:
              choice?.finish_reason || 'stop'
          }
        ]
      })}\n\n`
    );

    safeWrite(res, 'data: [DONE]\n\n');

    if (!res.writableEnded) {
      res.end();
    }

  } catch (error) {
    console.error(
      '[PROXY] Fatal error:',
      error.message
    );

    if (error.response?.data) {
      console.error(
        '[PROXY] Upstream response:',
        error.response.data
      );
    }

    if (!res.headersSent) {
      return res.status(
        error.response?.status || 500
      ).json({
        error: {
          message: error.message,
          type: 'proxy_error',
          code:
            error.response?.status || 500
        }
      });
    }

    if (!res.writableEnded) {
      safeWrite(
        res,
        `data: ${JSON.stringify({
          error: {
            message: error.message,
            type: 'proxy_error'
          }
        })}\n\n`
      );

      safeWrite(
        res,
        'data: [DONE]\n\n'
      );

      res.end();
    }
  }
});

upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
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

        if (buffer.length > MAX_BUFFER_SIZE) {
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
  
  // Run validation after server starts, non-blocking
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});

// ─── FASTCRWSUPPORTB ───────────────────────────────────────────────────────────────

async function searchFastCRW(args) {
  if (!FASTCRW_API_KEY) {
    throw new Error('FASTCRW_API_KEY is not configured');
  }

  const query = String(args?.query || '').trim();

  if (!query) {
    throw new Error('search_web requires a non-empty query');
  }

  if (query.length > 1000) {
    throw new Error('Search query is too long');
  }

  const limit = Math.min(
    Math.max(Number(args?.limit) || 3, 1),
    5
  );

  const timeRangeMap = {
    hour: 'qdr:h',
    day: 'qdr:d',
    week: 'qdr:w',
    month: 'qdr:m',
    year: 'qdr:y'
  };

  const body = {
    query,
    limit,
    sources: ['web']
  };

  if (args?.time_range && timeRangeMap[args.time_range]) {
    body.tbs = timeRangeMap[args.time_range];
  }

  console.log(
    `[FASTCRW] Searching: "${query}" (limit=${limit})`
  );

  const response = await axios.post(
    `${FASTCRW_API_BASE}/v1/search`,
    body,
    {
      headers: {
        Authorization: `Bearer ${FASTCRW_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: FASTCRW_TIMEOUT_MS
    }
  );

  const results = Array.isArray(response.data?.data)
    ? response.data.data
    : [];

  return {
    success: true,
    query,
    results: results.map((r, index) => ({
      title: r.title || '',
      url: r.url || '',
      snippet: r.snippet || ''
    }))
  };
}


async function runWithWebSearch({
  selectedModel,
  messages,
  temperature,
  max_tokens
}) {
  const workingMessages = Array.isArray(messages)
    ? messages.map(m => ({ ...m }))
    : [];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const requestBody = {
      model: selectedModel,
      messages: workingMessages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(
        max_tokens ?? 2048,
        MAX_TOKENS_LIMIT
      ),
      stream: stream || false,

      tools: FASTCRW_TOOLS,
      tool_choice: 'auto',
      parallel_tool_calls: false,

      
     extra_body: ENABLE_THINKING_MODE
        ? {
            chat_template_kwargs: {
              thinking: true
            }
          }
        : undefined
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      requestBody,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: REQUEST_TIMEOUT_MS
      }
    );

    const choice = response.data?.choices?.[0];
    const assistantMessage = choice?.message;

    if (!assistantMessage) {
      throw new Error('NIM returned no assistant message');
    }

    const toolCalls = assistantMessage.tool_calls || [];

    // Model has finished answering.
    if (toolCalls.length === 0) {
      return response.data;
    }

    // Preserve the assistant tool-call message exactly.
    workingMessages.push({
      role: 'assistant',
      content: assistantMessage.content || null,
      tool_calls: toolCalls
    });

    for (const toolCall of toolCalls) {
      const toolName = toolCall.function?.name;

      if (toolName !== 'search_web') {
        workingMessages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify({
            success: false,
            error: `Unknown tool: ${toolName}`
          })
        });

        continue;
      }

      let args;

      try {
        args = JSON.parse(
          toolCall.function?.arguments || '{}'
        );
      } catch {
        workingMessages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify({
            success: false,
            error: 'Invalid JSON arguments supplied to search_web'
          })
        });

        continue;
      }

      try {
        const result = await searchFastCRW(args);

        workingMessages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(result)
        });
      } catch (error) {
        console.error(
         '[FASTCRW] Search failed:',
         error.response?.status || 'unknown',
         error.message
        );
      }
if (error.response?.data) {
  console.error(
    '[FASTCRW] Upstream response:',
    error.response.data
  );
}
        workingMessages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify({
            success: false,
            error: 'Live web search failed',
            details: error.message
          })
        });
      }
    }
  }

  throw new Error(
    `Web search tool loop exceeded ${MAX_TOOL_ROUNDS} rounds`
  );
}

