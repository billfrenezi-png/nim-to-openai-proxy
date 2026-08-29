// NIMTOPROXY.js
// Clean Hybrid OpenAI ↔ NVIDIA NIM Proxy
//
// Features:
// - OpenAI-compatible /v1/chat/completions
// - Model aliases
// - Model-specific reasoning controls
// - Reasoning normalization
// - Optional inline <thinking> output
// - FASTCRW web search
// - Conservative tool loop
// - Model-aware retry/backoff
// - Retry-After support
// - Streaming support
// - Authentication
// - Startup model validation
//
// Node.js + Express 5
// ---------------------------------------------------------------------------

'use strict';

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PORT = Number(process.env.PORT || 3000);

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const FASTCRW_API_BASE =
  process.env.FASTCRW_API_BASE ||
  'https://api.fastcrw.com';

const FASTCRW_API_KEY = process.env.FASTCRW_API_KEY;

const SHOW_REASONING =
  process.env.SHOW_REASONING === 'true';

const ENABLE_THINKING_MODE =
  process.env.ENABLE_THINKING_MODE === 'true';

const ENABLE_WEB_SEARCH =
  process.env.ENABLE_WEB_SEARCH === 'true';

const SKIP_VALIDATION =
  process.env.SKIP_VALIDATION === 'true';

const DISCORD_WEBHOOK_URL =
  process.env.DISCORD_WEBHOOK_URL;

// General request limits.
const MAX_TOKENS_LIMIT = 66536;
const DEFAULT_MAX_TOKENS = 8192;

const REQUEST_TIMEOUT_MS =
  Number(process.env.REQUEST_TIMEOUT_MS || 180000);

const FASTCRW_TIMEOUT_MS =
  Number(process.env.FASTCRW_TIMEOUT_MS || 30000);

const VALIDATION_TIMEOUT_MS =
  Number(process.env.VALIDATION_TIMEOUT_MS || 15000);

const MAX_BUFFER_SIZE =
  Number(process.env.MAX_BUFFER_SIZE || 1024 * 1024);

// ---------------------------------------------------------------------------
// FASTCRW limits
// ---------------------------------------------------------------------------

// Deliberately conservative.
//
// 1 means:
//   model -> one search -> mandatory synthesis
//
// There is intentionally no "search again because the model feels like it".
const MAX_TOOL_ROUNDS = Math.min(
  Math.max(
    Number(process.env.MAX_TOOL_ROUNDS || 1),
    1
  ),
  2
);

// Maximum actual searches during a single round.
const MAX_SEARCH_CALLS_PER_ROUND = 1;

// Maximum searches during one complete user request.
//
// Keeping this at one is the important anti-loop protection.
const MAX_SEARCHES_PER_REQUEST = 1;

const FASTCRW_MAX_TOKENS = Math.min(
  Math.max(
    Number(process.env.FASTCRW_MAX_TOKENS || 8192),
    256
  ),
  MAX_TOKENS_LIMIT
);

// ---------------------------------------------------------------------------
// Model mapping
// ---------------------------------------------------------------------------

const MODEL_MAPPING = {
  'gpt-3.5-turbo':
    'moonshotai/kimi-k3',

  'gpt-4':
    'nvidia/nemotron-3-ultra-550b-a55b',

  'gpt-3.5':
    'qwen/qwen3.5-397b-a17b',

  'gpt-4-turbo':
    'z-ai/glm-5.2',

  'gpt-4o':
    'deepseek-ai/deepseek-v4-pro-0813',

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

// ---------------------------------------------------------------------------
// Model retry profiles
// ---------------------------------------------------------------------------
//
// IMPORTANT:
//
// A retry is not free.
//
// On a busy NIM endpoint, aggressive retries can turn:
//
//   one 429
//
// into:
//
//   429 -> 429 -> 429 -> 429
//
// which makes the situation worse.
//
// GLM and MiniMax M3 therefore get particularly conservative policies.
//
// retryCount = number of retries AFTER the initial request.
//
// Example:
//   retryCount: 1
//
// means maximum 2 attempts total.
//
// ---------------------------------------------------------------------------

const DEFAULT_RETRY_POLICY = {
  retryCount: 2,

  // Initial delay for transient 5xx errors.
  serverBaseDelayMs: 2000,

  // Initial delay for 429.
  rateLimitBaseDelayMs: 4000,

  // Maximum calculated delay.
  maxDelayMs: 30000,

  // Random jitter.
  jitterMs: 750
};

const CONSERVATIVE_RETRY_POLICY = {
  // Only one retry after the initial request.
  retryCount: 1,

  // Wait longer instead of immediately hitting the endpoint again.
  serverBaseDelayMs: 4000,

  // Rate limits deserve an even longer pause.
  rateLimitBaseDelayMs: 7000,

  maxDelayMs: 45000,

  jitterMs: 1000
};

const RETRY_POLICIES = {
  // These models are intentionally conservative.
  'z-ai/glm-5.2':
    CONSERVATIVE_RETRY_POLICY,

  'minimaxai/minimax-m3':
    CONSERVATIVE_RETRY_POLICY,

  'moonshotai/kimi-k3':
    CONSERVATIVE_RETRY_POLICY
};

function getRetryPolicy(model) {
  return RETRY_POLICIES[model] || DEFAULT_RETRY_POLICY;
}

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------

function validateConfig() {
  const fatal = message => {
    console.error(`[FATAL] ${message}`);
    process.exit(1);
  };

  if (!NIM_API_KEY) {
    fatal(
      'NIM_API_KEY is required.'
    );
  }

  if (!CLIENT_AUTH_KEY) {
    console.warn(
      '[WARN] CLIENT_AUTH_KEY is not configured. Requests will be rejected.'
    );
  }

  if (ENABLE_WEB_SEARCH && !FASTCRW_API_KEY) {
    fatal(
      'FASTCRW_API_KEY is required when ENABLE_WEB_SEARCH=true.'
    );
  }
}

validateConfig();

// ---------------------------------------------------------------------------
// Reasoning delimiter parser
// ---------------------------------------------------------------------------

class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    if (!chunk) {
      return {
        content: '',
        reasoning: ''
      };
    }

    this.buffer += chunk;

    let content = '';
    let reasoning = '';

    while (true) {
      const target = this.inThinking
        ? this.closeTag
        : this.openTag;

      const index = this.buffer.indexOf(target);

      if (index !== -1) {
        const before = this.buffer.slice(0, index);

        if (this.inThinking) {
          reasoning += before;
        } else {
          content += before;
        }

        this.inThinking = !this.inThinking;

        this.buffer =
          this.buffer.slice(
            index + target.length
          );

        continue;
      }

      // Preserve a partial delimiter at the end
      // of the current chunk.
      let partialLength = 0;

      const maxLength = Math.min(
        this.buffer.length,
        target.length - 1
      );

      for (
        let length = maxLength;
        length > 0;
        length--
      ) {
        if (
          target.startsWith(
            this.buffer.slice(
              this.buffer.length - length
            )
          )
        ) {
          partialLength = length;
          break;
        }
      }

      const safeLength =
        this.buffer.length - partialLength;

      const safeText =
        this.buffer.slice(0, safeLength);

      if (this.inThinking) {
        reasoning += safeText;
      } else {
        content += safeText;
      }

      this.buffer =
        this.buffer.slice(safeLength);

      break;
    }

    return {
      content,
      reasoning
    };
  }

  flush() {
    const result = {
      content: '',
      reasoning: ''
    };

    if (this.buffer) {
      if (this.inThinking) {
        result.reasoning = this.buffer;
      } else {
        result.content = this.buffer;
      }
    }

    this.buffer = '';

    return result;
  }
}

// ---------------------------------------------------------------------------
// Reasoning helpers
// ---------------------------------------------------------------------------

const THINK_TAG_MODELS = new Set([
  'qwen/qwen3.5-397b-a17b',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5'
]);

function createThinkingParser(model) {
  if (!THINK_TAG_MODELS.has(model)) {
    return null;
  }

  return new DelimiterParser(
    '<think>',
    '</think>'
  );
}

class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = createThinkingParser(model);
  }

  processDelta(delta) {
    const normalized = {
      ...delta
    };

    let reasoning =
      normalized.reasoning ||
      normalized.reasoning_content ||
      '';

    let content =
      normalized.content || '';

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
      normalized.content = content;
    } else {
      delete normalized.content;
    }

    if (reasoning) {
      normalized.reasoning = reasoning;
    } else {
      delete normalized.reasoning;
    }

    delete normalized.reasoning_content;

    return normalized;
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

function normalizeNonStreamChoice(choice, model) {
  if (!choice) {
    return choice;
  }

  const message = choice.message || {};

  let reasoning =
    message.reasoning ||
    message.reasoning_content ||
    '';

  let content =
    message.content || '';

  if (
    !reasoning &&
    content &&
    THINK_TAG_MODELS.has(model)
  ) {
    const parser = createThinkingParser(model);

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

  const newMessage = {
    ...message
  };

  if (content) {
    newMessage.content = content;
  }

  if (reasoning) {
    newMessage.reasoning = reasoning;
  }

  delete newMessage.reasoning_content;

  return {
    ...choice,
    message: newMessage
  };
}

// ---------------------------------------------------------------------------
// Model-specific reasoning payload
// ---------------------------------------------------------------------------

function getReasoningPayload(
  model,
  enableThinking,
  reasoningEffort,
  hasTools
) {
  const effort = reasoningEffort;

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b':
    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) {
        return {};
      }

      const payload = {
        chat_template_kwargs: {
          enable_thinking: true
        }
      };

      if (
        model ===
          'nvidia/nemotron-3-ultra-550b-a55b' &&
        hasTools
      ) {
        payload.chat_template_kwargs
          .force_nonempty_content = true;
      }

      return payload;
    }

    case 'qwen/qwen3.5-397b-a17b':
      // Qwen thinks by default.
      if (enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          enable_thinking: false
        }
      };

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
        payload.chat_template_kwargs
          .reasoning_effort = effort;
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
        payload.reasoning_effort = effort;
      }

      return payload;
    }

    case 'google/gemma-4-31b-it':
      if (!enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          enable_thinking: true
        }
      };

    case 'stepfun-ai/step-3.7-flash':
      if (enableThinking) {
        return {};
      }

      return {
        chat_template_kwargs: {
          thinking: false
        }
      };

    default:
      return {};
  }
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

app.use(cors());

app.use(
  express.json({
    limit: '10mb'
  })
);

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

function extractBearerToken(header) {
  if (
    !header ||
    typeof header !== 'string'
  ) {
    return null;
  }

  const match =
    header.trim().match(
      /^Bearer\s+(.+)$/i
    );

  return match
    ? match[1]
    : null;
}

function safeTimingEqual(a, b) {
  if (
    !a ||
    !b ||
    typeof a !== 'string' ||
    typeof b !== 'string' ||
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

app.use((req, res, next) => {
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
        type: 'authentication_error',
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
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise(resolve =>
    setTimeout(resolve, ms)
  );
}

function safeWrite(res, data) {
  try {
    if (
      !res.writableEnded &&
      !res.destroyed &&
      res.writable
    ) {
      res.write(data);
      return true;
    }
  } catch (error) {
    console.warn(
      '[STREAM] Write failed:',
      error.message
    );
  }

  return false;
}

function clamp(value, min, max) {
  return Math.min(
    Math.max(value, min),
    max
  );
}

// ---------------------------------------------------------------------------
// Retry / backoff
// ---------------------------------------------------------------------------

const RETRYABLE_STATUS_CODES =
  new Set([
    429,
    502,
    503,
    504
  ]);

function parseRetryAfter(value) {
  if (!value) {
    return null;
  }

  // Retry-After: <seconds>
  const seconds =
    Number(value);

  if (
    Number.isFinite(seconds) &&
    seconds >= 0
  ) {
    return seconds * 1000;
  }

  // Retry-After: HTTP-date
  const timestamp =
    Date.parse(value);

  if (!Number.isNaN(timestamp)) {
    return Math.max(
      timestamp - Date.now(),
      0
    );
  }

  return null;
}

function calculateBackoff(
  status,
  attempt,
  policy,
  retryAfterHeader
) {
  const serverError =
    status >= 500;

  const base =
    status === 429
      ? policy.rateLimitBaseDelayMs
      : policy.serverBaseDelayMs;

  // Respect Retry-After when provided.
  const retryAfter =
    parseRetryAfter(
      retryAfterHeader
    );

  if (retryAfter !== null) {
    return clamp(
      retryAfter +
        Math.floor(
          Math.random() *
          policy.jitterMs
        ),
      0,
      policy.maxDelayMs
    );
  }

  // Exponential backoff:
  //
  // attempt 0:
  //   base
  //
  // attempt 1:
  //   base * 2
  //
  // etc.
  const exponential =
    base *
    Math.pow(2, attempt);

  const jitter =
    Math.floor(
      Math.random() *
      policy.jitterMs
    );

  return clamp(
    exponential + jitter,
    0,
    policy.maxDelayMs
  );
}

function getStatusMessage(status) {
  switch (status) {
    case 429:
      return 'NIM rate limited the request';

    case 502:
      return 'NIM returned Bad Gateway';

    case 503:
      return 'NIM service unavailable';

    case 504:
      return 'NIM gateway timeout';

    default:
      return `NIM returned HTTP ${status}`;
  }
}

async function postNIM(
  requestBody,
  options = {}
) {
  const model =
    requestBody?.model;

  const policy =
    getRetryPolicy(model);

  const responseType =
    options.responseType || 'json';

  for (
    let attempt = 0;
    attempt <= policy.retryCount;
    attempt++
  ) {
    try {
      return await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        requestBody,
        {
          responseType,

          headers: {
            Authorization:
              `Bearer ${NIM_API_KEY}`,

            'Content-Type':
              'application/json'
          },

          timeout:
            options.timeout ||
            REQUEST_TIMEOUT_MS,

          // Let us handle retryable status codes
          // ourselves.
          validateStatus: status =>
            status >= 200 &&
            status < 300
        }
      );
    } catch (error) {
      const status =
        error.response?.status;

      const retryable =
        RETRYABLE_STATUS_CODES.has(
          status
        );

      const lastAttempt =
        attempt >= policy.retryCount;

      if (
        !retryable ||
        lastAttempt
      ) {
        throw error;
      }

      const retryAfter =
        error.response?.headers?.[
          'retry-after'
        ];

      const delay =
        calculateBackoff(
          status,
          attempt,
          policy,
          retryAfter
        );

      console.warn(
        `[NIM] ${getStatusMessage(status)} ` +
        `for ${model}. ` +
        `Retrying in ${delay}ms ` +
        `(retry ${attempt + 1}/${policy.retryCount})`
      );

      await sleep(delay);
    }
  }

  throw new Error(
    'NIM request failed after retry policy was exhausted.'
  );
}

// ---------------------------------------------------------------------------
// FASTCRW tool
// ---------------------------------------------------------------------------

const FASTCRW_TOOLS = [
  {
    type: 'function',

    function: {
      name: 'search_web',

      description:
        'Use this tool ONLY when the answer genuinely requires external web information. ' +
        'Do not search merely because a topic is current, interesting, or potentially changing. ' +
        'Do not search for creative writing, roleplay, fictional continuation, brainstorming, casual conversation, opinions, or questions answerable from the supplied context. ' +
        'Use ONE focused query. ' +
        'After search results are provided, answer using those results instead of searching again.',

      parameters: {
        type: 'object',

        properties: {
          query: {
            type: 'string',

            description:
              'One focused web search query for information genuinely missing from the conversation.'
          },

          limit: {
            type: 'integer',

            minimum: 1,
            maximum: 5,

            description:
              'Prefer 2-3 results.'
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
              'Use a freshness filter only when freshness matters.'
          }
        },

        required: [
          'query'
        ]
      }
    }
  }
];

// ---------------------------------------------------------------------------
// FASTCRW request
// ---------------------------------------------------------------------------

async function fastcrwSearch(
  query,
  limit = 3,
  timeRange = 'any'
) {
  const params = {
    q: query,

    limit: clamp(
      Number(limit) || 3,
      1,
      5
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

// ---------------------------------------------------------------------------
// Tool-call helpers
// ---------------------------------------------------------------------------

function makeToolResult(
  toolCallId,
  payload
) {
  return {
    role: 'tool',
    tool_call_id: toolCallId,
    content: JSON.stringify(payload)
  };
}

function normalizeSearchQuery(query) {
  return String(query || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function parseToolArguments(toolCall) {
  try {
    return JSON.parse(
      toolCall.function?.arguments ||
        '{}'
    );
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// FASTCRW tool loop
// ---------------------------------------------------------------------------
//
// The old implementation supported multiple rounds and then separately
// synthesized. That works, but it gives reasoning-heavy models more
// opportunities to decide that another search would be useful.
//
// This implementation deliberately has a much harder boundary:
//
//   request
//      ↓
//   model + tool
//      ↓
//   at most ONE search
//      ↓
//   tool-free synthesis
//
// MAX_TOOL_ROUNDS is retained as an environment option, but the global
// MAX_SEARCHES_PER_REQUEST remains the important safety limit.
// ---------------------------------------------------------------------------

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
      ? messages.map(message => ({
          ...message
        }))
      : [];

  const searchedQueries =
    new Set();

  let totalSearches = 0;

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
          max_tokens ??
            FASTCRW_MAX_TOKENS,
          FASTCRW_MAX_TOKENS
        ),

      stream: false,

      tools:
        FASTCRW_TOOLS,

      tool_choice: 'auto',

      ...reasoningPayload
    };

    // If a search has already occurred, do not expose
    // the tool again.
    if (
      totalSearches >=
      MAX_SEARCHES_PER_REQUEST
    ) {
      delete requestBody.tools;
      delete requestBody.tool_choice;
    }

    const response =
      await postNIM(
        requestBody
      );

    const assistantMessage =
      response.data
        ?.choices?.[0]
        ?.message;

    if (!assistantMessage) {
      return response.data;
    }

    const toolCalls =
      Array.isArray(
        assistantMessage.tool_calls
      )
        ? assistantMessage.tool_calls
        : [];

    // Model answered normally.
    if (toolCalls.length === 0) {
      return response.data;
    }

    // Preserve the assistant tool-call message.
    workingMessages.push(
      assistantMessage
    );

    let searchCallsThisRound = 0;

    for (const toolCall of toolCalls) {
      if (
        toolCall.type !== 'function'
      ) {
        continue;
      }

      const functionName =
        toolCall.function?.name;

      if (
        functionName !== 'search_web'
      ) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                `Unknown tool: ${functionName}`
            }
          )
        );

        continue;
      }

      // Hard global search limit.
      if (
        totalSearches >=
        MAX_SEARCHES_PER_REQUEST
      ) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                'Search limit reached for this request. Answer using the information already available.'
            }
          )
        );

        continue;
      }

      // Hard per-round limit.
      if (
        searchCallsThisRound >=
        MAX_SEARCH_CALLS_PER_ROUND
      ) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                'Only one web search is permitted in this round. Use the available search results.'
            }
          )
        );

        continue;
      }

      const args =
        parseToolArguments(
          toolCall
        );

      if (!args) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                'Invalid tool arguments.'
            }
          )
        );

        continue;
      }

      const query =
        String(
          args.query || ''
        ).trim();

      const limit =
        clamp(
          Number(args.limit) || 3,
          1,
          5
        );

      const timeRange =
        args.time_range ||
        'any';

      if (!query) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                'Search query is empty. Answer using the available context.'
            }
          )
        );

        continue;
      }

      const normalizedQuery =
        normalizeSearchQuery(
          query
        );

      const searchKey =
        `${normalizedQuery}|${timeRange}`;

      // Duplicate protection.
      if (
        searchedQueries.has(
          searchKey
        )
      ) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                'This search has already been performed. Use the existing results.'
            }
          )
        );

        continue;
      }

      searchedQueries.add(
        searchKey
      );

      searchCallsThisRound++;
      totalSearches++;

      console.log(
        `[FASTCRW] Search ${totalSearches}/${MAX_SEARCHES_PER_REQUEST}: ${query}`
      );

      try {
        const result =
          await fastcrwSearch(
            query,
            limit,
            timeRange
          );

        workingMessages.push(
          makeToolResult(
            toolCall.id,
            result
          )
        );
      } catch (error) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,
              error:
                'Live web search failed.'
            }
          )
        );
      }
    }

    // -----------------------------------------------------------------------
    // Search happened.
    //
    // Do NOT expose tools again.
    // This is deliberately a separate synthesis request.
    // -----------------------------------------------------------------------

    if (
      totalSearches > 0
    ) {
      return runSynthesisRequest({
        selectedModel,
        messages:
          workingMessages,
        temperature,
        max_tokens,
        enableThinking,
        reasoningEffort
      });
    }
  }

  // Extremely defensive fallback.
  return runSynthesisRequest({
    selectedModel,
    messages:
      workingMessages,
    temperature,
    max_tokens,
    enableThinking,
    reasoningEffort
  });
}

// ---------------------------------------------------------------------------
// Tool-free synthesis
// ---------------------------------------------------------------------------

async function runSynthesisRequest({
  selectedModel,
  messages,
  temperature,
  max_tokens,
  enableThinking,
  reasoningEffort
}) {
  const reasoningPayload =
    getReasoningPayload(
      selectedModel,
      enableThinking,
      reasoningEffort,
      false
    );

  const response =
    await postNIM({
      model: selectedModel,

      messages,

      temperature:
        temperature ?? 0.7,

      max_tokens:
        Math.min(
          max_tokens ??
            FASTCRW_MAX_TOKENS,
          FASTCRW_MAX_TOKENS
        ),

      stream: false,

      ...reasoningPayload
    });

  return response.data;
}

// ---------------------------------------------------------------------------
// Standard NIM completion
// ---------------------------------------------------------------------------

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
        max_tokens ??
          DEFAULT_MAX_TOKENS,
        MAX_TOKENS_LIMIT
      ),

    stream:
      Boolean(stream),

    ...reasoningPayload
  };

  console.log(
    `[NIM] Using model: ${selectedModel}`
  );

  const response =
    await postNIM(
      requestBody,
      stream
        ? {
            responseType: 'stream'
          }
        : {}
    );

  return {
    response,
    usedModel:
      selectedModel
  };
}

// ---------------------------------------------------------------------------
// OpenAI response normalization
// ---------------------------------------------------------------------------

function buildOpenAIResponse(
  nimData,
  requestedModel,
  actualModel
) {
  return {
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
      requestedModel ||
      actualModel,

    choices:
      (nimData.choices || [])
        .map(
          (choice, index) => ({
            ...normalizeNonStreamChoice(
              choice,
              actualModel
            ),

            index
          })
        ),

    usage:
      nimData.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
  };
}

function applyReasoningVisibility(
  message,
  inlineReasoning
) {
  const finalMessage = {
    ...message
  };

  const reasoning =
    finalMessage.reasoning || '';

  let content =
    finalMessage.content || '';

  if (
    SHOW_REASONING &&
    inlineReasoning &&
    reasoning
  ) {
    content =
      `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
  }

  finalMessage.content =
    content;

  if (SHOW_REASONING && reasoning) {
    finalMessage.reasoning =
      reasoning;

    finalMessage.reasoning_content =
      reasoning;
  } else {
    delete finalMessage.reasoning;
    delete finalMessage.reasoning_content;
  }

  return finalMessage;
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

function setSSEHeaders(res) {
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

  // Useful for reverse proxies.
  res.setHeader(
    'X-Accel-Buffering',
    'no'
  );
}

function sendSSE(res, payload) {
  return safeWrite(
    res,
    `data: ${JSON.stringify(payload)}\n\n`
  );
}

function sendDone(res) {
  return safeWrite(
    res,
    'data: [DONE]\n\n'
  );
}

// ---------------------------------------------------------------------------
// FASTCRW response -> SSE
// ---------------------------------------------------------------------------

function streamCompletedResponse(
  res,
  responseData
) {
  const id =
    responseData.id ||
    `chatcmpl-${Date.now()}`;

  const created =
    responseData.created ||
    Math.floor(
      Date.now() / 1000
    );

  const model =
    responseData.model ||
    'unknown';

  const choice =
    responseData.choices?.[0];

  sendSSE(res, {
    id,

    object:
      'chat.completion.chunk',

    created,

    model,

    choices: [
      {
        index: 0,

        delta: {
          role: 'assistant'
        },

        finish_reason: null
      }
    ]
  });

  const content =
    choice?.message?.content ||
    '';

  if (content) {
    sendSSE(res, {
      id,

      object:
        'chat.completion.chunk',

      created,

      model,

      choices: [
        {
          index: 0,

          delta: {
            content
          },

          finish_reason: null
        }
      ]
    });
  }

  sendSSE(res, {
    id,

    object:
      'chat.completion.chunk',

    created,

    model,

    choices: [
      {
        index: 0,

        delta: {},

        finish_reason:
          choice?.finish_reason ||
          'stop'
      }
    ]
  });

  sendDone(res);

  if (!res.writableEnded) {
    res.end();
  }
}

// ---------------------------------------------------------------------------
// Streaming NIM response
// ---------------------------------------------------------------------------

function handleNIMStream({
  req,
  res,
  upstreamStream,
  usedModel
}) {
  setSSEHeaders(res);

  const decoder =
    new StringDecoder('utf8');

  let buffer = '';
  let doneSent = false;
  let cleanEnd = false;
  let cleanedUp = false;

  let reasoningOpen = false;

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

  const finish = () => {
    if (doneSent) {
      return;
    }

    sendDone(res);

    doneSent = true;
    cleanEnd = true;

    if (!res.writableEnded) {
      res.end();
    }

    cleanup();
  };

  const processLine = line => {
    if (
      !line.startsWith('data: ')
    ) {
      return;
    }

    const raw =
      line.slice(6).trim();

    if (
      raw === '[DONE]'
    ) {
      finish();
      return;
    }

    try {
      const data =
        JSON.parse(raw);

      const delta =
        data.choices?.[0]?.delta;

      if (!delta) {
        sendSSE(
          res,
          data
        );

        return;
      }

      const normalized =
        normalizer.processDelta(
          delta
        );

      let clientContent =
        '';

      if (
        SHOW_REASONING &&
        normalized.reasoning
      ) {
        if (!reasoningOpen) {
          clientContent +=
            `<thinking>\n${normalized.reasoning}`;

          reasoningOpen = true;
        } else {
          clientContent +=
            normalized.reasoning;
        }
      }

      if (normalized.content) {
        if (reasoningOpen) {
          clientContent +=
            `\n</thinking>\n\n${normalized.content}`;

          reasoningOpen = false;
        } else {
          clientContent +=
            normalized.content;
        }
      }

      // Never mutate the upstream object directly.
      const outputData = {
        ...data,

        choices:
          data.choices.map(
            (choice, index) => {
              if (index !== 0) {
                return choice;
              }

              return {
                ...choice,

                delta: {
                  ...choice.delta,

                  content:
                    clientContent
                }
              };
            }
          )
      };

      if (
        SHOW_REASONING &&
        normalized.reasoning
      ) {
        outputData.choices[0]
          .delta.reasoning =
            normalized.reasoning;

        outputData.choices[0]
          .delta.reasoning_content =
            normalized.reasoning;
      } else {
        delete outputData
          .choices[0]
          .delta.reasoning;

        delete outputData
          .choices[0]
          .delta.reasoning_content;
      }

      sendSSE(
        res,
        outputData
      );
    } catch (error) {
      console.warn(
        '[STREAM] Invalid JSON chunk:',
        raw.slice(0, 100)
      );

      sendSSE(res, {
        error: {
          message:
            'Upstream sent malformed chunk',

          type:
            'stream_parse_error',

          details:
            raw.slice(0, 100)
        }
      });
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
          '[STREAM] Buffer overflow.'
        );

        sendSSE(res, {
          error: {
            message:
              'Stream buffer overflow',

            type:
              'stream_error'
          }
        });

        sendDone(res);

        doneSent = true;

        if (!res.writableEnded) {
          res.end();
        }

        upstreamStream.destroy();
        cleanup();

        return;
      }

      const lines =
        buffer.split('\n');

      buffer =
        lines.pop() || '';

      for (const line of lines) {
        processLine(
          line.replace(/\r$/, '')
        );
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
          processLine(
            line.replace(/\r$/, '')
          );
        }
      }

      if (doneSent) {
        return;
      }

      const flushed =
        normalizer.flush();

      let finalContent =
        '';

      if (
        SHOW_REASONING &&
        flushed.reasoning
      ) {
        if (!reasoningOpen) {
          finalContent +=
            `<thinking>\n${flushed.reasoning}`;

          reasoningOpen = true;
        } else {
          finalContent +=
            flushed.reasoning;
        }
      }

      if (flushed.content) {
        if (reasoningOpen) {
          finalContent +=
            `\n</thinking>\n\n${flushed.content}`;

          reasoningOpen = false;
        } else {
          finalContent +=
            flushed.content;
        }
      }

      if (finalContent) {
        sendSSE(res, {
          choices: [
            {
              delta: {
                content:
                  finalContent
              }
            }
          ]
        });
      }

      finish();
    }
  );

  upstreamStream.on(
    'error',
    error => {
      console.error(
        '[STREAM] Upstream error:',
        error.message
      );

      if (!res.writableEnded) {
        sendSSE(res, {
          error: {
            message:
              'Stream interrupted by upstream error',

            type:
              'stream_error'
          }
        });

        if (!doneSent) {
          sendDone(res);
          doneSent = true;
        }

        res.end();
      }

      cleanup();
    }
  );

  req.on(
    'close',
    () => {
      if (
        !cleanEnd
      ) {
        console.warn(
          '[STREAM] Client disconnected.'
        );
      }

      if (
        upstreamStream &&
        !upstreamStream.destroyed &&
        !cleanEnd
      ) {
        upstreamStream.destroy();
      }

      cleanup();
    }
  );
}

// ---------------------------------------------------------------------------
// Model validation
// ---------------------------------------------------------------------------

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log(
      '[VALIDATION] Skipped.'
    );

    return;
  }

  console.log(
    '[VALIDATION] Checking NIM /v1/models...'
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

    const available =
      new Set(
        (response.data?.data || [])
          .map(model => model.id)
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
        available.has(nimId)
      ) {
        console.log(
          `[VALIDATION] ✓ ${alias} → ${nimId}`
        );
      } else {
        console.warn(
          `[VALIDATION] ✗ ${alias} → ${nimId}`
        );

        invalid.push({
          alias,
          nimId,
          error:
            'Model not found in NIM catalog'
        });
      }
    }

    if (invalid.length) {
      await sendDiscordAlert(
        invalid
      );
    } else {
      console.log(
        '[VALIDATION] All models valid.'
      );
    }
  } catch (error) {
    console.warn(
      `[VALIDATION] Model check failed: ${error.message}`
    );

    console.warn(
      '[VALIDATION] Continuing without validation.'
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
      `${invalidModels.length} model(s) failed validation.`,

    color: 0xff4444,

    timestamp:
      new Date().toISOString(),

    fields:
      invalidModels.map(model => ({
        name:
          `\`${model.alias}\``,

        value:
          `Backend: \`${model.nimId}\`\n` +
          `Error: \`${model.error}\``,

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
  } catch (error) {
    console.error(
      '[DISCORD] Failed:',
      error.message
    );
  }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

app.get(
  '/health',
  (req, res) => {
    res.json({
      status: 'ok',
      version: '4.0.0'
    });
  }
);

// ---------------------------------------------------------------------------
// Model list
// ---------------------------------------------------------------------------

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

          object:
            'model',

          created:
            Math.floor(
              Date.now() / 1000
            ),

          owned_by:
            'nim-proxy'
        }))
    });
  }
);

// ---------------------------------------------------------------------------
// Main chat completion route
// ---------------------------------------------------------------------------

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    let upstreamStream = null;

    try {
      const {
        model,
        messages,
        temperature,
        max_tokens,
        stream
      } = req.body;

      // ---------------------------------------------------------------
      // Request validation
      // ---------------------------------------------------------------

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
        `[PROXY] ${model} → ${selectedModel}`
      );

      // ---------------------------------------------------------------
      // Reasoning settings
      // ---------------------------------------------------------------

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

      // ---------------------------------------------------------------
      // FASTCRW
      // ---------------------------------------------------------------

      if (ENABLE_WEB_SEARCH) {
        const result =
          await runWithWebSearch({
            selectedModel,

            messages,

            temperature,

            max_tokens,

            enableThinking,

            reasoningEffort
          });

        const response =
          buildOpenAIResponse(
            result,
            model,
            selectedModel
          );

        // Apply reasoning visibility.
        response.choices =
          response.choices.map(
            choice => ({
              ...choice,

              message:
                applyReasoningVisibility(
                  choice.message,
                  inlineReasoning
                )
            })
          );

        if (!stream) {
          return res.json(
            response
          );
        }

        streamCompletedResponse(
          res,
          response
        );

        return;
      }

      // ---------------------------------------------------------------
      // Normal NIM request
      // ---------------------------------------------------------------

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

      // ---------------------------------------------------------------
      // Non-streaming
      // ---------------------------------------------------------------

      if (!stream) {
        const response =
          buildOpenAIResponse(
            result.response.data,
            model,
            result.usedModel
          );

        response.choices =
          response.choices.map(
            choice => ({
              ...choice,

              message:
                applyReasoningVisibility(
                  choice.message,
                  inlineReasoning
                )
            })
          );

        return res.json(
          response
        );
      }

      // ---------------------------------------------------------------
      // Streaming
      // ---------------------------------------------------------------

      return handleNIMStream({
        req,

        res,

        upstreamStream,

        usedModel:
          result.usedModel
      });

    } catch (error) {
      console.error(
        '[PROXY] Request failed:',
        error.message
      );

      if (error.response?.data) {
        console.error(
          '[PROXY] Upstream response:',
          error.response.data
        );
      }

      // ---------------------------------------------------------------
      // Normal HTTP error
      // ---------------------------------------------------------------

      if (!res.headersSent) {
        const status =
          error.response?.status ||
          500;

        return res
          .status(status)
          .json({
            error: {
              message:
                error.message,

              type:
                'proxy_error',

              code:
                status
            }
          });
      }

      // ---------------------------------------------------------------
      // Error after SSE started
      // ---------------------------------------------------------------

      if (
        !res.writableEnded
      ) {
        sendSSE(res, {
          error: {
            message:
              error.message,

            type:
              'proxy_error'
          }
        });

        sendDone(res);

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

// ---------------------------------------------------------------------------
// 404
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

app.listen(
  PORT,
  () => {
    console.log(
      `[PROXY] Hybrid NIM proxy running on port ${PORT}`
    );

    console.log(
      `[PROXY] Max tokens: ${MAX_TOKENS_LIMIT}`
    );

    console.log(
      `[PROXY] FASTCRW: ${ENABLE_WEB_SEARCH ? 'enabled' : 'disabled'}`
    );

    console.log(
      `[PROXY] Retry profiles: default + conservative GLM/M3`
    );

    validateModels()
      .catch(error => {
        console.error(
          '[VALIDATION] Startup check failed:',
          error.message
        );
      });
  }
);
