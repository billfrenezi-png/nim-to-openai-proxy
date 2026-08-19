// server.js — Robust Hybrid OpenAI ↔ NIM Proxy
// Express 5 Compatible
// Includes:
// - Model mapping
// - Model-specific reasoning controls
// - Structured reasoning normalization
// - Optional inline <thinking> output
// - FASTCRW live web search
// - OpenAI-compatible responses
// - Streaming support
// - Authentication
// - Model validation
// - Safe stream handling

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING =
  process.env.SHOW_REASONING === 'true';

const ENABLE_THINKING_MODE =
  process.env.ENABLE_THINKING_MODE === 'true';

const SKIP_VALIDATION =
  process.env.SKIP_VALIDATION === 'true';

const DISCORD_WEBHOOK_URL =
  process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 85536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024;

// ─── FASTCRW Configuration ───────────────────────────────────────────────────

const FASTCRW_API_BASE =
  process.env.FASTCRW_API_BASE ||
  'https://api.fastcrw.com';

const FASTCRW_API_KEY =
  process.env.FASTCRW_API_KEY;

const ENABLE_WEB_SEARCH =
  process.env.ENABLE_WEB_SEARCH === 'true';

// Keep this configurable, but make 1 the conservative default.
// 1 = model may search once, then MUST synthesize without another search.
// 2 = allows a second tool round if you deliberately want it.
const MAX_TOOL_ROUNDS = Math.min(
  Math.max(
    parseInt(process.env.MAX_TOOL_ROUNDS || '1', 10),
    1
  ),
  3
);

// Prevent the model from requesting a ridiculous number of searches
// in a single tool response.
const MAX_SEARCH_CALLS_PER_ROUND = 1;

// Separate token ceiling for the tool loop.
// Web-search planning/synthesis generally doesn't need the full
// normal MAX_TOKENS_LIMIT.
const FASTCRW_MAX_TOKENS = Math.min(
  Math.max(
    parseInt(
      process.env.FASTCRW_MAX_TOKENS || '8192',
      10
    ),
    256
  ),
  MAX_TOKENS_LIMIT
);

const FASTCRW_TIMEOUT_MS = 30000;

// ─── Configuration Validation ───────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => {
    console.error(`[FATAL] ${msg}`);
    process.exit(1);
  };

  if (!NIM_API_KEY) {
    fatal(
      'NIM_API_KEY is required. Get one at https://build.nvidia.com/'
    );
  }

  if (!CLIENT_AUTH_KEY) {
    console.warn(
      '[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.'
    );
  }

  if (ENABLE_WEB_SEARCH && !FASTCRW_API_KEY) {
    fatal(
      'FASTCRW_API_KEY is required when ENABLE_WEB_SEARCH=true'
    );
  }
}

validateConfig();

// ─── Model Mapping ───────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo':
    'nvidia/nemotron-3-super-120b-a12b',

  'gpt-4':
    'nvidia/nemotron-3-ultra-550b-a55b',

  'gpt-3.5':
    'qwen/qwen3.5-397b-a17b',

  'gpt-4-turbo':
    'z-ai/glm-5.2',

  'gpt-4o':
    'deepseek-ai/deepseek-v4-pro',

  'claude-3-opus':
    'openai/gpt-oss-120b',

  'claude-3-sonnet':
    'openai/gpt-oss-20b',

  'gemini-pro':
    'nvidia/llama-3.3-nemotron-super-49b-v1.5',

  'gemini-turbo':
    'meta/llama-3.3-70b-instruct',

  'gemini-turbo?':
    'abacusai/dracarys-llama-3.1-70b-instruct',

  'gpt-3.5o':
    'nvidia/nemotron-mini-4b-instruct',

  'gpt-4-flash':
    'deepseek-ai/deepseek-v4-flash',

  'glm-5.2':
    'z-ai/glm-5.2',

  'mistral':
    'thinkingmachines/inkling',

  'mistral-turbo':
    'mistralai/mistral-medium-3.5-128b',

  'mistral-pro':
    'mistralai/mistral-small-4-119b-2603',

  'mistral-nemo':
    'mistralai/mistral-nemotron',

  'mistral-fast':
    'mistralai/ministral-14b-instruct-2512',

  'google-light':
    'google/gemma-4-31b-it',

  'google-lightest':
    'google/gemma-2-2b-it',

  'google-lighter':
    'google/gemma-3n-e4b-it',

  'm3':
    'minimaxai/minimax-m3',

  'step-3.5-flash':
    'stepfun-ai/step-3.5-flash',

  'step-3.7-flash':
    'stepfun-ai/step-3.7-flash'
};

// ─── Reasoning Subsystem ────────────────────────────────────────────────────

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
      const targetTag = this.inThinking
        ? this.closeTag
        : this.openTag;

      const tagIndex =
        this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore =
          this.buffer.substring(0, tagIndex);

        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }

        this.inThinking = !this.inThinking;

        this.buffer =
          this.buffer.substring(
            tagIndex + targetTag.length
          );
      } else {
        // Preserve a possible partial delimiter
        // at the end of the current chunk.
        let partialLen = 0;

        const maxLen = Math.min(
          this.buffer.length,
          targetTag.length - 1
        );

        for (let i = maxLen; i > 0; i--) {
          if (
            targetTag.startsWith(
              this.buffer.substring(
                this.buffer.length - i
              )
            )
          ) {
            partialLen = i;
            break;
          }
        }

        const textBefore =
          this.buffer.substring(
            0,
            this.buffer.length - partialLen
          );

        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }

        this.buffer =
          this.buffer.substring(
            this.buffer.length - partialLen
          );

        break;
      }
    }

    return {
      content,
      reasoning
    };
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

    return {
      content,
      reasoning
    };
  }
}

class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;

    // These models embed reasoning inside <think> tags.
    if (
      model ===
        'qwen/qwen3.5-397b-a17b' ||
      model ===
        'nvidia/llama-3.3-nemotron-super-49b-v1.5'
    ) {
      this.parser =
        new DelimiterParser(
          '<think>',
          '</think>'
        );
    }

    // Models such as Gemma, DeepSeek and GPT-OSS
    // use structured reasoning fields.
  }

  processDelta(delta) {
    const normalizedDelta = {
      ...delta
    };

    let reasoning =
      normalizedDelta.reasoning ||
      normalizedDelta.reasoning_content ||
      '';

    let content =
      normalizedDelta.content || '';

    // Structured reasoning has priority.
    if (
      !reasoning &&
      content &&
      this.parser
    ) {
      const parsed =
        this.parser.processChunk(content);

      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) {
      normalizedDelta.content = content;
    } else {
      delete normalizedDelta.content;
    }

    if (reasoning) {
      normalizedDelta.reasoning =
        reasoning;
    } else {
      delete normalizedDelta.reasoning;
    }

    delete normalizedDelta.reasoning_content;

    return normalizedDelta;
  }

  flush() {
    if (!this.parser) {
      return {
        content: '',
        reasoning: ''
      };
    }

    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(
  choice,
  model
) {
  if (!choice) {
    return choice;
  }

  const message =
    choice.message || {};

  let reasoning =
    message.reasoning ||
    message.reasoning_content ||
    '';

  let content =
    message.content || '';

  if (!reasoning && content) {
    let parser = null;

    if (
      model ===
        'qwen/qwen3.5-397b-a17b' ||
      model ===
        'nvidia/llama-3.3-nemotron-super-49b-v1.5'
    ) {
      parser =
        new DelimiterParser(
          '<think>',
          '</think>'
        );
    }

    if (parser) {
      const parsed =
        parser.processChunk(content);

      const flushed =
        parser.flush();

      content =
        (parsed.content || '') +
        (flushed.content || '');

      reasoning =
        (parsed.reasoning || '') +
        (flushed.reasoning || '');
    }
  }

  const newMessage = {
    ...message
  };

  if (content) {
    newMessage.content = content;
  }

  if (reasoning) {
    newMessage.reasoning =
      reasoning;
  }

  delete newMessage.reasoning_content;

  return {
    ...choice,
    message: newMessage
  };
}

// ─── Model-Specific Reasoning Request Payloads ───────────────────────────────

function getReasoningPayload(
  model,
  enableThinking,
  clientReasoningEffort,
  hasTools
) {
  const effort =
    clientReasoningEffort;

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          enable_thinking: true
        }
      };
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) {
        return {};
      }

      const payload = {
        chat_template_kwargs: {
          enable_thinking: true
        }
      };

      if (hasTools) {
        payload.chat_template_kwargs.force_nonempty_content =
          true;
      }

      return payload;
    }

    case 'qwen/qwen3.5-397b-a17b': {
      // Qwen defaults to thinking.
      // Only explicitly disable it when requested.
      if (enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          enable_thinking: false
        }
      };
    }

    case 'deepseek-ai/deepseek-v4-pro':
    case 'deepseek-ai/deepseek-v4-flash': {
      if (!enableThinking) {
        return {};
      }

      const payload = {
        chat_template_kwargs: {
          thinking: true
        }
      };

      if (effort) {
        payload.chat_template_kwargs.reasoning_effort =
          effort;
      }

      return payload;
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (
        effort &&
        ['low', 'medium', 'high'].includes(effort)
      ) {
        return {
          reasoning_effort: effort
        };
      }

      if (enableThinking) {
        return {
          reasoning_effort: 'high'
        };
      }

      return {};
    }

    case 'mistralai/mistral-medium-3.5-128b':
    case 'mistralai/mistral-small-4-119b-2603': {
      if (
        effort &&
        ['high', 'none'].includes(effort)
      ) {
        return {
          reasoning_effort: effort
        };
      }

      if (enableThinking) {
        return {
          reasoning_effort: 'high'
        };
      }

      return {};
    }

    case 'z-ai/glm-5.2': {
      const payload = {
        thinking: {
          type: enableThinking
            ? 'enabled'
            : 'disabled'
        }
      };

      if (
        enableThinking &&
        effort
      ) {
        payload.reasoning_effort =
          effort;
      }

      return payload;
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          enable_thinking: true
        }
      };
    }

    case 'stepfun-ai/step-3.7-flash': {
      if (enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          thinking: false
        }
      };
    }

    default:
      return {};
  }
}

// ─── Middleware ──────────────────────────────────────────────────────────────

app.use(cors());

app.use(
  express.json({
    limit: '10mb'
  })
);

function extractBearerToken(
  authHeader
) {
  if (
    !authHeader ||
    typeof authHeader !== 'string'
  ) {
    return null;
  }

  const parts =
    authHeader.trim().split(' ');

  if (
    parts.length !== 2 ||
    parts[0] !== 'Bearer'
  ) {
    return null;
  }

  return parts[1];
}

function safeTimingEqual(a, b) {
  if (
    !a ||
    !b ||
    a.length !== b.length
  ) {
    return false;
  }

  try {
    return timingSafeEqual(
      Buffer.from(a),
      Buffer.from(b)
    );
  } catch {
    return false;
  }
}

app.use(
  (req, res, next) => {
    if (
      req.path === '/health' ||
      req.path === '/v1/models'
    ) {
      return next();
    }

    const token =
      extractBearerToken(
        req.headers.authorization
      );

    if (
      !token ||
      !CLIENT_AUTH_KEY
    ) {
      return res.status(403).json({
        error: {
          message:
            'Forbidden: Invalid or missing authentication',
          type:
            'authentication_error',
          code: 403
        }
      });
    }

    if (
      !safeTimingEqual(
        token,
        CLIENT_AUTH_KEY
      )
    ) {
      return res.status(403).json({
        error: {
          message:
            'Forbidden: Invalid authentication credentials',
          type:
            'authentication_error',
          code: 403
        }
      });
    }

    next();
  }
);

// ─── Validation ──────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log(
      '[VALIDATION] Skipped (SKIP_VALIDATION=true)'
    );
    return;
  }

  console.log(
    '[VALIDATION] Checking model availability via /v1/models...'
  );

  try {
    const response =
      await axios.get(
        `${NIM_API_BASE}/models`,
        {
          headers: {
            Authorization:
              `Bearer ${NIM_API_KEY}`,
            'Content-Type':
              'application/json'
          },
          timeout:
            VALIDATION_TIMEOUT_MS
        }
      );

    const availableModels =
      new Set(
        (response.data.data || [])
          .map(m => m.id)
      );

    const invalid = [];

    for (
      const [
        alias,
        nimId
      ] of Object.entries(
        MODEL_MAPPING
      )
    ) {
      if (
        availableModels.has(nimId)
      ) {
        console.log(
          `[VALIDATION] ✓ ${alias} → ${nimId}`
        );
      } else {
        console.warn(
          `[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`
        );

        invalid.push({
          alias,
          nimId,
          error:
            'Model not found in NIM catalog'
        });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(
        invalid
      );
    } else {
      console.log(
        '[VALIDATION] All models valid.'
      );
    }
  } catch (err) {
    console.warn(
      `[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`
    );

    console.warn(
      '[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.'
    );
  }
}

async function sendDiscordAlert(
  invalidModels
) {
  if (!DISCORD_WEBHOOK_URL) {
    return;
  }

  const embed = {
    title:
      '⚠️ NIM Proxy: Model Validation Failed',

    description:
      `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,

    color: 0xff4444,

    timestamp:
      new Date().toISOString(),

    fields:
      invalidModels.map(m => ({
        name: `\`${m.alias}\``,
        value:
          `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
        inline: true
      }))
  };

  try {
    await axios.post(
      DISCORD_WEBHOOK_URL,
      {
        embeds: [embed],
        username:
          'NIM Proxy Monitor'
      },
      {
        timeout: 5000
      }
    );

    console.log(
      '[DISCORD] Alert sent.'
    );
  } catch (err) {
    console.error(
      '[DISCORD] Failed to send alert:',
      err.message
    );
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function safeWrite(
  res,
  data
) {
  try {
    if (
      !res.writableEnded &&
      !res.destroyed &&
      res.writable
    ) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn(
      '[STREAM] Write failed:',
      err.message
    );
  }

  return false;
}

function sleep(ms) {
  return new Promise(
    resolve => setTimeout(resolve, ms)
  );
}

async function postNIM(
  requestBody,
  options = {}
) {
  const MAX_RETRIES = 3;

  for (
    let attempt = 0;
    attempt <= MAX_RETRIES;
    attempt++
  ) {
    try {
      return await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        requestBody,
        {
          responseType:
            options.responseType ||
            'json',

          headers: {
            Authorization:
              `Bearer ${NIM_API_KEY}`,
            'Content-Type':
              'application/json'
          },

          timeout:
            REQUEST_TIMEOUT_MS
        }
      );
    } catch (error) {
      const status =
        error.response?.status;

      if (
        status !== 429 ||
        attempt === MAX_RETRIES
      ) {
        throw error;
      }

      const delay =
        Math.min(
          1000 *
            Math.pow(
              2,
              attempt
            ),
          15000
        ) +
        Math.floor(
          Math.random() * 500
        );

      console.warn(
        `[NIM] 429 rate limit. Retrying in ${delay}ms (attempt ${attempt + 1}/${MAX_RETRIES})`
      );

      await sleep(delay);
    }
  }
}

// ─── FASTCRW Tool Definition ────────────────────────────────────────────────

const FASTCRW_TOOLS = [
  {
    type: 'function',

    function: {
      name: 'search_web',

      description:
        'Search the live web only when current or externally verifiable information is actually needed. Use search for current real-world facts, recent events or news, current public figures, current products, games, movies, shows, companies, software, or factual details that the supplied conversation does not contain. Do NOT search for creative writing, roleplay, fictional continuation, brainstorming, ordinary conversation, or information already present in the conversation. When searching, prefer a single focused search with 2-3 results.',

      parameters: {
        type: 'object',

        properties: {
          query: {
            type: 'string',
            description:
              'The web search query.'
          },

          limit: {
            type: 'integer',
            description:
              'Number of results to return. Prefer 2-3 for normal questions. Maximum 8.',
            minimum: 1,
            maximum: 8
          },

          time_range: {
            type: 'string',
            enum: [
              'hour',
              'day',
              'week',
              'month',
              'year',
              'any'
            ],
            description:
              'Optional freshness filter.'
          }
        },

        required: ['query']
      }
    }
  }
];

// ─── FASTCRW Search ──────────────────────────────────────────────────────────

async function fastcrwSearch(
  query,
  limit = 3,
  timeRange = 'any'
) {
  const params = {
    q: query,
    limit: Math.min(
      Math.max(limit || 3, 1),
      8
    )
  };

  if (
    timeRange &&
    timeRange !== 'any'
  ) {
    params.time_range =
      timeRange;
  }

  try {
    const response =
      await axios.get(
        `${FASTCRW_API_BASE}/search`,
        {
          params,

          headers: {
            Authorization:
              `Bearer ${FASTCRW_API_KEY}`,
            'Content-Type':
              'application/json'
          },

          timeout:
            FASTCRW_TIMEOUT_MS
        }
      );

    return response.data;
  } catch (error) {
    console.error(
      '[FASTCRW] Search failed:',
      error.response?.status ||
        'unknown',
      error.message
    );

    if (error.response?.data) {
      console.error(
        '[FASTCRW] Upstream response:',
        error.response.data
      );
    }

    throw error;
  }
}

// ─── FASTCRW Tool Loop ──────────────────────────────────────────────────────

async function runWithWebSearch({
  selectedModel,
  messages,
  temperature,
  max_tokens,
  enableThinking,
  reasoningEffort
}) {
  const workingMessages =
    Array.isArray(messages)
      ? messages.map(m => ({
          ...m
        }))
      : [];

  for (
    let round = 0;
    round < MAX_TOOL_ROUNDS;
    round++
  ) {
    const reasoningPayload =
      getReasoningPayload(
        selectedModel,
        enableThinking,
        reasoningEffort,
        true
      );

    const requestBody = {
      model: selectedModel,

      messages:
        workingMessages,

      temperature:
        temperature ?? 0.7,

      max_tokens:
        Math.min(
          max_tokens ?? 2048,
          MAX_TOKENS_LIMIT
        ),

      stream: false,

      tools: FASTCRW_TOOLS,

      tool_choice: 'auto',

      ...reasoningPayload
    };

    const response =
      await postNIM(
        requestBody
      );

    const assistantMessage =
      response.data?.choices?.[0]
        ?.message;

    if (!assistantMessage) {
      return response.data;
    }

    const toolCalls =
      assistantMessage.tool_calls ||
      [];

    if (
      !Array.isArray(toolCalls) ||
      toolCalls.length === 0
    ) {
      return response.data;
    }

    workingMessages.push(
      assistantMessage
    );

    for (
      const toolCall of toolCalls
    ) {
      if (
        toolCall.type !==
        'function'
      ) {
        continue;
      }

      const functionName =
        toolCall.function?.name;

      if (
        functionName !==
        'search_web'
      ) {
        workingMessages.push({
          role: 'tool',

          tool_call_id:
            toolCall.id,

          content:
            JSON.stringify({
              success: false,
              error:
                `Unknown tool: ${functionName}`
            })
        });

        continue;
      }

      let args = {};

      try {
        args = JSON.parse(
          toolCall.function
            ?.arguments || '{}'
        );
      } catch (error) {
        workingMessages.push({
          role: 'tool',

          tool_call_id:
            toolCall.id,

          content:
            JSON.stringify({
              success: false,
              error:
                'Invalid tool arguments'
            })
        });

        continue;
      }

      const query =
        String(
          args.query || ''
        ).trim();

      const limit =
        Number(
          args.limit || 3
        );

      const timeRange =
        args.time_range ||
        'any';

      if (!query) {
        workingMessages.push({
          role: 'tool',

          tool_call_id:
            toolCall.id,

          content:
            JSON.stringify({
              success: false,
              error:
                'Search query is empty'
            })
        });

        continue;
      }

      console.log(
        `[FASTCRW] Searching: ${query}`
      );

      try {
        const result =
          await fastcrwSearch(
            query,
            limit,
            timeRange
          );

        workingMessages.push({
          role: 'tool',

          tool_call_id:
            toolCall.id,

          content:
            JSON.stringify(
              result
            )
        });
      } catch (error) {
        workingMessages.push({
          role: 'tool',

          tool_call_id:
            toolCall.id,

          content:
            JSON.stringify({
              success: false,

              error:
                'Live web search failed',

              details:
                error.message
            })
        });
      }
    }
  }

  // If the model keeps asking for tools,
  // make one final normal completion
  // without tools.
  const finalReasoningPayload =
    getReasoningPayload(
      selectedModel,
      enableThinking,
      reasoningEffort,
      false
    );

  const finalResponse =
    await postNIM({
      model: selectedModel,

      messages:
        workingMessages,

      temperature:
        temperature ?? 0.7,

      max_tokens:
        Math.min(
          max_tokens ?? 2048,
          MAX_TOKENS_LIMIT
        ),

      stream: false,

      ...finalReasoningPayload
    });

  return finalResponse.data;
}

// ─── Fallback Completion ─────────────────────────────────────────────────────

async function callNIM({
  selectedModel,
  messages,
  temperature,
  max_tokens,
  stream,
  enableThinking,
  reasoningEffort,
  hasTools
}) {
  const reasoningPayload =
    getReasoningPayload(
      selectedModel,
      enableThinking,
      reasoningEffort,
      hasTools
    );

  const requestBody = {
    model: selectedModel,

    messages,

    temperature:
      temperature ?? 0.7,

    max_tokens:
      Math.min(
        max_tokens ?? 2048,
        MAX_TOKENS_LIMIT
      ),

    stream:
      Boolean(stream),

    ...reasoningPayload
  };

  console.log(
    `[NIM] Using model: ${selectedModel}`
  );

  return {
    response: await postNIM(
      requestBody,
      stream
        ? {
            responseType:
              'stream'
          }
        : {}
    ),

    usedModel:
      selectedModel
  };
}

// ─── Routes ──────────────────────────────────────────────────────────────────

app.get(
  '/health',
  (req, res) => {
    res.json({
      status: 'ok',
      version: '3.0.0'
    });
  }
);

app.get(
  '/v1/models',
  (req, res) => {
    res.json({
      object: 'list',

      data:
        Object.keys(
          MODEL_MAPPING
        ).map(id => ({
          id,
          object: 'model',
          created: Date.now(),
          owned_by: 'nim-proxy'
        }))
    });
  }
);

// ─── Main Chat Completion Route ──────────────────────────────────────────────

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    let upstreamStream = null;
    let streamEndedCleanly = false;

    try {
      const {
        model,
        messages,
        temperature,
        max_tokens,
        stream
      } = req.body;

      if (
        !Array.isArray(messages)
      ) {
        return res.status(400).json({
          error: {
            message:
              'messages must be an array',

            type:
              'invalid_request_error',

            code: 400
          }
        });
      }

      const selectedModel =
        MODEL_MAPPING[model];
      
      if (!selectedModel) {
        return res.status(400).json({
          error: {
            message:
              `Unknown model: ${model}`,
            type:
              'invalid_request_error',
            code: 400
          }
        });
      }

      console.log(
        `[PROXY] ${model || 'default'} → ${selectedModel}`
      );

      const enableThinking =
        ENABLE_THINKING_MODE;

      const reasoningEffort =
        req.body.reasoning_effort ||
        req.body.reasoningEffort ||
        undefined;

      const inlineReasoning =
        req.headers[
          'x-reasoning-format'
        ] === 'inline';

      // ─────────────────────────────────────────
      // FASTCRW path
      // ─────────────────────────────────────────

      if (ENABLE_WEB_SEARCH) {
        const webResult =
          await runWithWebSearch({
            selectedModel,
            messages,
            temperature,
            max_tokens,
            enableThinking,
            reasoningEffort
          });

        // FASTCRW internally uses a
        // non-streaming tool loop.
        //
        // We normalize the final response
        // and then optionally expose it
        // as OpenAI-compatible SSE below.

        const openaiResponse = {
          id:
            webResult.id ||
            `chatcmpl-${Date.now()}`,

          object:
            'chat.completion',

          created:
            webResult.created ||
            Math.floor(
              Date.now() / 1000
            ),

          model:
            model ||
            selectedModel,

          choices:
            (
              webResult.choices ||
              []
            ).map(
              (choice, i) => {
                const normalized =
                  normalizeNonStreamChoice(
                    choice,
                    selectedModel
                  );

                let content =
                  normalized.message
                    ?.content || '';

                const reasoning =
                  normalized.message
                    ?.reasoning || '';

                if (
                  SHOW_REASONING &&
                  inlineReasoning &&
                  reasoning
                ) {
                  content =
                    `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
                }

                const finalMessage = {
                  ...normalized.message,
                  content
                };

                if (
                  SHOW_REASONING &&
                  reasoning
                ) {
                  finalMessage.reasoning =
                    reasoning;

                  finalMessage.reasoning_content =
                    reasoning;
                } else {
                  delete finalMessage.reasoning;
                  delete finalMessage.reasoning_content;
                }

                return {
                  ...normalized,

                  index: i,

                  message:
                    finalMessage
                };
              }
            ),

          usage:
            webResult.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
        };

        if (!stream) {
          return res.json(
            openaiResponse
          );
        }

        // FASTCRW has already completed
        // the tool loop. We therefore expose
        // the final answer as normal SSE.

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
          openaiResponse
            .choices?.[0];

        const id =
          openaiResponse.id;

        const created =
          openaiResponse.created;

        const outputModel =
          openaiResponse.model;

        safeWrite(
          res,
          `data: ${JSON.stringify({
            id,
            object:
              'chat.completion.chunk',
            created,
            model: outputModel,
            choices: [
              {
                index: 0,
                delta: {
                  role:
                    'assistant'
                },
                finish_reason:
                  null
              }
            ]
          })}\n\n`
        );

        const content =
          choice?.message
            ?.content || '';

        if (content) {
          safeWrite(
            res,
            `data: ${JSON.stringify({
              id,
              object:
                'chat.completion.chunk',
              created,
              model: outputModel,
              choices: [
                {
                  index: 0,
                  delta: {
                    content
                  },
                  finish_reason:
                    null
                }
              ]
            })}\n\n`
          );
        }

        safeWrite(
          res,
          `data: ${JSON.stringify({
            id,
            object:
              'chat.completion.chunk',
            created,
            model: outputModel,
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason:
                  choice?.finish_reason ||
                  'stop'
              }
            ]
          })}\n\n`
        );

        safeWrite(
          res,
          'data: [DONE]\n\n'
        );

        return res.end();
      }

      // ─────────────────────────────────────────
      // Normal NIM path
      // ─────────────────────────────────────────

      const result =
        await callNIM({
          selectedModel,
          messages,
          temperature,
          max_tokens,
          stream,
          enableThinking,
          reasoningEffort,
          hasTools: false
        });

      upstreamStream =
        result.response.data;

      const usedModel =
        result.usedModel;

      // ─────────────────────────────────────────
      // Non-streaming
      // ─────────────────────────────────────────

      if (!stream) {
        const nimData =
          result.response.data;

        const openaiResponse = {
          id:
            nimData.id ||
            `chatcmpl-${Date.now()}`,

          object:
            'chat.completion',

          created:
            nimData.created ||
            Math.floor(
              Date.now() / 1000
            ),

          model:
            model ||
            usedModel,

          choices:
            (
              nimData.choices ||
              []
            ).map(
              (choice, i) => {
                const normalized =
                  normalizeNonStreamChoice(
                    choice,
                    usedModel
                  );

                let content =
                  normalized.message
                    ?.content || '';

                const reasoning =
                  normalized.message
                    ?.reasoning || '';

                if (
                  SHOW_REASONING &&
                  inlineReasoning &&
                  reasoning
                ) {
                  content =
                    `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
                }

                const finalMessage = {
                  ...normalized.message,
                  content
                };

                if (
                  SHOW_REASONING &&
                  reasoning
                ) {
                  finalMessage.reasoning =
                    reasoning;

                  finalMessage.reasoning_content =
                    reasoning;
                } else {
                  delete finalMessage.reasoning;
                  delete finalMessage.reasoning_content;
                }

                return {
                  ...normalized,

                  index: i,

                  message:
                    finalMessage
                };
              }
            ),

          usage:
            nimData.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
        };

        return res.json(
          openaiResponse
        );
      }

      // ─────────────────────────────────────────
      // Streaming
      // ─────────────────────────────────────────

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

      const decoder =
        new StringDecoder('utf8');

      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const normalizer =
        new StreamNormalizer(
          usedModel
        );

      const cleanup = () => {
        if (cleanedUp) {
          return;
        }

        cleanedUp = true;

        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }

        req.removeAllListeners(
          'close'
        );
      };

      const processLine = (
        line
      ) => {
        if (
          !line.startsWith(
            'data: '
          )
        ) {
          return;
        }

        if (
          line.includes(
            '[DONE]'
          )
        ) {
          if (!doneSent) {
            safeWrite(
              res,
              'data: [DONE]\n\n'
            );

            doneSent = true;
          }

          streamEndedCleanly =
            true;

          return;
        }

        try {
          const data =
            JSON.parse(
              line.slice(6)
            );

          const delta =
            data.choices?.[0]
              ?.delta;

          if (delta) {
            const normalizedDelta =
              normalizer.processDelta(
                delta
              );

            let clientContent =
              '';

            if (
              SHOW_REASONING &&
              inlineReasoning
            ) {
              if (
                normalizedDelta.reasoning &&
                !reasoningOpen
              ) {
                clientContent +=
                  `<thinking>\n${normalizedDelta.reasoning}`;

                reasoningOpen = true;
              } else if (
                normalizedDelta.reasoning
              ) {
                clientContent +=
                  normalizedDelta.reasoning;
              }

              if (
                normalizedDelta.content &&
                reasoningOpen
              ) {
                clientContent +=
                  `\n</thinking>\n\n${normalizedDelta.content}`;

                reasoningOpen = false;
              } else if (
                normalizedDelta.content
              ) {
                clientContent +=
                  normalizedDelta.content;
              }
            } else {
              clientContent =
                normalizedDelta.content ||
                '';
            }

            delta.content =
              clientContent;

            if (
              SHOW_REASONING &&
              normalizedDelta.reasoning
            ) {
              delta.reasoning =
                normalizedDelta.reasoning;

              delta.reasoning_content =
                normalizedDelta.reasoning;
            } else {
              delete delta.reasoning;
              delete delta.reasoning_content;
            }
          }

          safeWrite(
            res,
            `data: ${JSON.stringify(data)}\n\n`
          );
        } catch (parseErr) {
          console.warn(
            '[STREAM] Invalid JSON line:',
            line.slice(0, 100)
          );

          safeWrite(
            res,
            `data: ${JSON.stringify({
              error: {
                message:
                  'Upstream sent malformed chunk',
                type:
                  'stream_parse_error',
                details:
                  line.slice(0, 100)
              }
            })}\n\n`
          );
        }
      };

      upstreamStream.on(
        'data',
        chunk => {
          buffer +=
            decoder.write(chunk);

          if (
            buffer.length >
            MAX_BUFFER_SIZE
          ) {
            console.error(
              '[STREAM] Buffer overflow, destroying connection'
            );

            safeWrite(
              res,
              `data: ${JSON.stringify({
                error: {
                  message:
                    'Stream buffer overflow',
                  type:
                    'stream_error'
                }
              })}\n\n`
            );

            safeWrite(
              res,
              'data: [DONE]\n\n'
            );

            res.end();

            upstreamStream.destroy();

            cleanup();

            return;
          }

          const lines =
            buffer.split('\n');

          buffer =
            lines.pop() || '';

          for (
            const line of lines
          ) {
            processLine(line);
          }
        }
      );

      upstreamStream.on(
        'end',
        () => {
          buffer +=
            decoder.end();

          if (buffer.trim()) {
            for (
              const line of
                buffer.split('\n')
            ) {
              processLine(line);
            }
          }

          const flushed =
            normalizer.flush();

          if (
            flushed.content ||
            flushed.reasoning
          ) {
            let clientContent =
              '';

            if (
              SHOW_REASONING &&
              inlineReasoning
            ) {
              if (
                flushed.reasoning &&
                !reasoningOpen
              ) {
                clientContent +=
                  `<thinking>\n${flushed.reasoning}`;

                reasoningOpen = true;
              } else if (
                flushed.reasoning
              ) {
                clientContent +=
                  flushed.reasoning;
              }

              if (
                flushed.content &&
                reasoningOpen
              ) {
                clientContent +=
                  `\n</thinking>\n\n${flushed.content}`;

                reasoningOpen = false;
              } else if (
                flushed.content
              ) {
                clientContent +=
                  flushed.content;
              }
            } else {
              clientContent =
                flushed.content || '';
            }

            if (clientContent) {
              safeWrite(
                res,
                `data: ${JSON.stringify({
                  choices: [
                    {
                      delta: {
                        content:
                          clientContent
                      }
                    }
                  ]
                })}\n\n`
              );
            }
          }

          if (!doneSent) {
            safeWrite(
              res,
              'data: [DONE]\n\n'
            );

            doneSent = true;
          }

          streamEndedCleanly =
            true;

          if (!res.writableEnded) {
            res.end();
          }

          cleanup();
        }
      );

      upstreamStream.on(
        'error',
        err => {
          console.error(
            '[STREAM] Upstream error:',
            err.message
          );

          if (
            !res.writableEnded
          ) {
            safeWrite(
              res,
              `data: ${JSON.stringify({
                error: {
                  message:
                    'Stream interrupted by upstream error',
                  type:
                    'stream_error'
                }
              })}\n\n`
            );

            safeWrite(
              res,
              'data: [DONE]\n\n'
            );

            res.end();
          }

          cleanup();
        }
      );

      req.on(
        'close',
        () => {
          const clientGone =
            req.destroyed ||
            !res.writable;

          if (
            !streamEndedCleanly &&
            clientGone
          ) {
            console.warn(
              '[STREAM] Client disconnected prematurely'
            );
          }

          if (
            upstreamStream &&
            !upstreamStream.destroyed &&
            !streamEndedCleanly
          ) {
            upstreamStream.destroy();
          }

          cleanup();
        }
      );

    } catch (error) {
      console.error(
        '[PROXY] Fatal error:',
        error.message
      );

      if (
        error.response?.data
      ) {
        console.error(
          '[PROXY] Upstream response:',
          error.response.data
        );
      }

      if (
        !res.headersSent
      ) {
        return res
          .status(
            error.response?.status ||
              500
          )
          .json({
            error: {
              message:
                error.message,

              type:
                'proxy_error',

              code:
                error.response?.status ||
                500
            }
          });
      }

      if (
        !res.writableEnded
      ) {
        safeWrite(
          res,
          `data: ${JSON.stringify({
            error: {
              message:
                error.message,
              type:
                'proxy_error'
            }
          })}\n\n`
        );

        safeWrite(
          res,
          'data: [DONE]\n\n'
        );

        res.end();
      }

      if (
        upstreamStream &&
        !upstreamStream.destroyed
      ) {
        upstreamStream.destroy();
      }
    }
  }
);

// ─── 404 Handler ─────────────────────────────────────────────────────────────

app.use(
  (req, res) => {
    res.status(404).json({
      error: {
        message:
          `Endpoint ${req.method} ${req.path} not found`,

        type:
          'invalid_request_error',

        code: 404
      }
    });
  }
);

// ─── Startup ─────────────────────────────────────────────────────────────────

app.listen(
  PORT,
  () => {
    console.log(
      `[PROXY] Hybrid proxy running on port ${PORT}`
    );

    console.log(
      `[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`
    );

    validateModels().catch(
      err => {
        console.error(
          '[VALIDATION] Startup check failed:',
          err.message
        );
      }
    );
  }
);
