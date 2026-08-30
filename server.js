// NIMTOPROXY.js
// Clean Hybrid OpenAI <-> NVIDIA NIM Proxy
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

// ===========================================================================
// CONFIGURATION
// ===========================================================================

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

// ---------------------------------------------------------------------------
// General limits
// ---------------------------------------------------------------------------

const MAX_TOKENS_LIMIT = 66536;
const DEFAULT_MAX_TOKENS = 8192;

const REQUEST_TIMEOUT_MS = positiveNumber(
  process.env.REQUEST_TIMEOUT_MS,
  180000
);

const FASTCRW_TIMEOUT_MS = positiveNumber(
  process.env.FASTCRW_TIMEOUT_MS,
  30000
);

const VALIDATION_TIMEOUT_MS = positiveNumber(
  process.env.VALIDATION_TIMEOUT_MS,
  15000
);

const MAX_BUFFER_SIZE = positiveNumber(
  process.env.MAX_BUFFER_SIZE,
  1024 * 1024
);

// ---------------------------------------------------------------------------
// FASTCRW limits
//
// Intentionally conservative.
//
// The important part is that the model cannot continuously search:
//
//   - maximum 2 tool rounds by default
//   - maximum 2 searches for the whole request
//   - maximum 1 actual search per round
//   - duplicate / near-duplicate protection
//   - minimum delay between searches
//
// This is deliberately less trigger-happy.
// ---------------------------------------------------------------------------

const MAX_TOOL_ROUNDS = clamp(
  integerOrDefault(
    process.env.MAX_TOOL_ROUNDS,
    2
  ),
  1,
  4
);

const MAX_SEARCHES_PER_REQUEST = clamp(
  integerOrDefault(
    process.env.MAX_SEARCHES_PER_REQUEST,
    2
  ),
  1,
  8
);

const MAX_SEARCH_CALLS_PER_ROUND = clamp(
  integerOrDefault(
    process.env.MAX_SEARCH_CALLS_PER_ROUND,
    1
  ),
  1,
  2
);

const MIN_SEARCH_INTERVAL_MS = Math.max(
  integerOrDefault(
    process.env.MIN_SEARCH_INTERVAL_MS,
    2500
  ),
  0
);

const MAX_SEARCH_QUERY_LENGTH = clamp(
  integerOrDefault(
    process.env.MAX_SEARCH_QUERY_LENGTH,
    300
  ),
  32,
  1000
);

const MAX_SEARCH_RESULTS = clamp(
  integerOrDefault(
    process.env.MAX_SEARCH_RESULTS,
    5
  ),
  1,
  10
);

const MAX_QUERY_HISTORY = clamp(
  integerOrDefault(
    process.env.MAX_QUERY_HISTORY,
    8
  ),
  1,
  32
);

const FASTCRW_MAX_TOKENS = clamp(
  integerOrDefault(
    process.env.FASTCRW_MAX_TOKENS,
    8192
  ),
  256,
  MAX_TOKENS_LIMIT
);

// ===========================================================================
// MODEL MAPPING
// ===========================================================================

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

// ===========================================================================
// RETRY POLICIES
// ===========================================================================

const DEFAULT_RETRY_POLICY = Object.freeze({
  // Maximum retries AFTER the initial request.
  retryCount: 2,

  // 5xx errors.
  serverBaseDelayMs: 2500,

  // 429 errors.
  rateLimitBaseDelayMs: 8000,

  // Never wait longer than this from calculated backoff.
  maxDelayMs: 60000,

  // Randomized jitter prevents synchronized retries.
  jitterMs: 1500
});

// GLM / M3 are treated more carefully because hammering a busy
// endpoint with retries generally makes 429 behavior worse.
const CONSERVATIVE_RETRY_POLICY = Object.freeze({
  retryCount: 1,

  serverBaseDelayMs: 6000,

  rateLimitBaseDelayMs: 12000,

  maxDelayMs: 60000,

  jitterMs: 2000
});

const RETRY_POLICIES = {
  'z-ai/glm-5.2':
    CONSERVATIVE_RETRY_POLICY,

  'minimaxai/minimax-m3':
    CONSERVATIVE_RETRY_POLICY,

  'moonshotai/kimi-k3':
    CONSERVATIVE_RETRY_POLICY
};

const RETRYABLE_STATUS_CODES = new Set([
  429,
  502,
  503,
  504
]);

function getRetryPolicy(model) {
  return (
    RETRY_POLICIES[model] ||
    DEFAULT_RETRY_POLICY
  );
}

// ===========================================================================
// SMALL HELPERS
// ===========================================================================

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

function clamp(value, min, max) {
  return Math.min(
    Math.max(value, min),
    max
  );
}

function integerOrDefault(value, fallback) {
  const parsed = Number(value);

  return Number.isFinite(parsed)
    ? Math.floor(parsed)
    : fallback;
}

function positiveNumber(value, fallback) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }

  return parsed;
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

function normalizeSearchQuery(query) {
  return String(query || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function cleanSearchQuery(query) {
  return String(query || '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_SEARCH_QUERY_LENGTH);
}

function isValidSearchQuery(query) {
  if (typeof query !== 'string') {
    return false;
  }

  const normalized = query.trim();

  return (
    normalized.length >= 2 &&
    normalized.length <= MAX_SEARCH_QUERY_LENGTH
  );
}

// ===========================================================================
// CONFIG VALIDATION
// ===========================================================================

function validateConfig() {
  const fatal = message => {
    console.error(`[FATAL] ${message}`);
    process.exit(1);
  };

  if (!NIM_API_KEY) {
    fatal('NIM_API_KEY is required.');
  }

  if (!CLIENT_AUTH_KEY) {
    console.warn(
      '[WARN] CLIENT_AUTH_KEY is not configured. ' +
      'Protected requests will be rejected.'
    );
  }

  if (
    ENABLE_WEB_SEARCH &&
    !FASTCRW_API_KEY
  ) {
    fatal(
      'FASTCRW_API_KEY is required when ENABLE_WEB_SEARCH=true.'
    );
  }
}

validateConfig();

app.use(cors());

// ===========================================================================
// AUTHENTICATION
// ===========================================================================

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
    typeof a !== 'string' ||
    typeof b !== 'string' ||
    !a ||
    !b
  ) {
    return false;
  }

  const aBuffer = Buffer.from(a);
  const bBuffer = Buffer.from(b);

  if (aBuffer.length !== bBuffer.length) {
    return false;
  }

  try {
    return timingSafeEqual(
      aBuffer,
      bBuffer
    );
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  // These endpoints intentionally remain public.
  if (
    req.path === '/health' ||
    req.path === '/v1/models'
  ) {
    return next();
  }

  const token = extractBearerToken(
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

// ===========================================================================
// MIDDLEWARE
// ===========================================================================

app.use(
  express.json({
    limit: '10mb'
  })
);

// ===========================================================================
// REASONING DELIMITER PARSER
// ===========================================================================

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

      const index =
        this.buffer.indexOf(target);

      if (index !== -1) {
        const before =
          this.buffer.slice(0, index);

        if (this.inThinking) {
          reasoning += before;
        } else {
          content += before;
        }

        this.inThinking =
          !this.inThinking;

        this.buffer =
          this.buffer.slice(
            index + target.length
          );

        continue;
      }

      // Preserve a possible partial delimiter
      // at the end of the current chunk.
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
        const suffix =
          this.buffer.slice(
            this.buffer.length - length
          );

        if (target.startsWith(suffix)) {
          partialLength = length;
          break;
        }
      }

      const safeLength =
        this.buffer.length -
        partialLength;

      const safeText =
        this.buffer.slice(
          0,
          safeLength
        );

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

// ===========================================================================
// REASONING NORMALIZATION
// ===========================================================================

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
    this.parser =
      createThinkingParser(model);
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
      normalized.content ||
      '';

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
    message.content ||
    '';

  if (
    !reasoning &&
    content &&
    THINK_TAG_MODELS.has(model)
  ) {
    const parser =
      createThinkingParser(model);

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

// ===========================================================================
// MODEL-SPECIFIC REASONING PAYLOAD
// ===========================================================================

function getReasoningPayload(
  model,
  enableThinking,
  reasoningEffort,
  hasTools
) {
  const effort =
    typeof reasoningEffort === 'string'
      ? reasoningEffort
      : undefined;

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
        ['low', 'medium', 'high']
          .includes(effort)
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
        ['high', 'none']
          .includes(effort)
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
          type:
            enableThinking
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

// ===========================================================================
// RETRY / BACKOFF
// ===========================================================================

function parseRetryAfter(value) {
  if (!value) {
    return null;
  }

  // Retry-After: <seconds>
  const seconds = Number(value);

  if (
    Number.isFinite(seconds) &&
    seconds >= 0
  ) {
    return seconds * 1000;
  }

  // Retry-After: HTTP-date
  const timestamp = Date.parse(value);

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
  const base =
    status === 429
      ? policy.rateLimitBaseDelayMs
      : policy.serverBaseDelayMs;

  const retryAfter =
    parseRetryAfter(
      retryAfterHeader
    );

  // Retry-After is authoritative enough to prefer it,
  // but still add small jitter so multiple clients don't
  // wake up simultaneously.
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

  const timeout =
    options.timeout ||
    REQUEST_TIMEOUT_MS;

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

          timeout,

          // Make Axios reject non-2xx responses so
          // the retry logic can inspect them.
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

// ===========================================================================
// FASTCRW TOOL DEFINITION
// ===========================================================================

const FASTCRW_TOOLS = [
  {
    type: 'function',

    function: {
      name: 'search_web',

      description:
        'Search the web only when external information is genuinely necessary. ' +
        'Do not search for casual conversation, creative writing, roleplay, ' +
        'brainstorming, opinions, or information already available in context. ' +
        'Prefer ONE focused query. Inspect the returned results before searching again. ' +
        'Do not repeat or trivially vary an existing query. ' +
        'A strict search budget applies to the entire request.',

      parameters: {
        type: 'object',

        properties: {
          query: {
            type: 'string',

            description:
              'One focused search query. ' +
              'Avoid multiple unrelated questions.'
          },

          limit: {
            type: 'integer',

            minimum: 1,
            maximum: MAX_SEARCH_RESULTS,

            description:
              `Number of results requested. Prefer 2-3. ` +
              `Maximum ${MAX_SEARCH_RESULTS}.`
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
              'Use a freshness restriction only when freshness matters.'
          }
        },

        required: [
          'query'
        ],

        additionalProperties: false
      }
    }
  }
];

// ===========================================================================
// FASTCRW SEARCH PACING
// ===========================================================================

async function waitForSearchCooldown(
  lastSearchAt
) {
  if (!lastSearchAt) {
    return;
  }

  const elapsed =
    Date.now() - lastSearchAt;

  const remaining =
    MIN_SEARCH_INTERVAL_MS -
    elapsed;

  if (remaining > 0) {
    await sleep(remaining);
  }
}

// ===========================================================================
// FASTCRW QUERY SIMILARITY
// ===========================================================================

function tokenizeQuery(query) {
  return new Set(
    normalizeSearchQuery(query)
      .split(' ')
      .filter(Boolean)
  );
}

function querySimilarity(a, b) {
  const tokensA =
    tokenizeQuery(a);

  const tokensB =
    tokenizeQuery(b);

  if (
    !tokensA.size ||
    !tokensB.size
  ) {
    return 0;
  }

  let intersection = 0;

  for (const token of tokensA) {
    if (tokensB.has(token)) {
      intersection++;
    }
  }

  const union =
    new Set([
      ...tokensA,
      ...tokensB
    ]).size;

  return union
    ? intersection / union
    : 0;
}

function isDuplicateOrSimilarQuery(
  query,
  previousQueries
) {
  const normalized =
    normalizeSearchQuery(query);

  for (
    const previous of previousQueries
  ) {
    const previousNormalized =
      normalizeSearchQuery(previous);

    if (
      normalized ===
      previousNormalized
    ) {
      return true;
    }

    if (
      querySimilarity(
        normalized,
        previousNormalized
      ) >= 0.8
    ) {
      return true;
    }
  }

  return false;
}

// ===========================================================================
// FASTCRW REQUEST
// ===========================================================================

async function fastcrwSearch(
  query,
  limit = 3,
  timeRange = 'any'
) {
  const safeLimit = clamp(
    Number(limit) || 3,
    1,
    MAX_SEARCH_RESULTS
  );

  const params = {
    q: query,
    limit: safeLimit
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
            FASTCRW_TIMEOUT_MS,

          validateStatus: status =>
            status >= 200 &&
            status < 300
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

// ===========================================================================
// TOOL CALL HELPERS
// ===========================================================================

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

function parseToolArguments(toolCall) {
  try {
    const raw =
      toolCall.function?.arguments ||
      '{}';

    const parsed =
      JSON.parse(raw);

    if (
      !parsed ||
      typeof parsed !== 'object' ||
      Array.isArray(parsed)
    ) {
      return null;
    }

    return parsed;
  } catch {
    return null;
  }
}

// ===========================================================================
// FASTCRW SYNTHESIS
// ===========================================================================
//
// Once the search budget / round budget is reached, explicitly remove tools.
// This prevents the "final answer" request from immediately deciding that it
// needs another search.
// ===========================================================================

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

  const requestBody = {
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
  };

  console.log(
    `[FASTCRW] Final synthesis with ${selectedModel}`
  );

  const response =
    await postNIM(
      requestBody
    );

  return response.data;
}

// ===========================================================================
// FASTCRW TOOL LOOP
// ===========================================================================

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

  const searchedQueries = [];

  let totalSearches = 0;
  let lastSearchAt = 0;

  for (
    let round = 0;
    round < MAX_TOOL_ROUNDS;
    round++
  ) {
    console.log(
      `[FASTCRW] Round ${round + 1}/${MAX_TOOL_ROUNDS} ` +
      `| searches ${totalSearches}/${MAX_SEARCHES_PER_REQUEST}`
    );

    // Once the global search budget has been exhausted,
    // don't even advertise the tool anymore.
    const canSearch =
      totalSearches <
      MAX_SEARCHES_PER_REQUEST;

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

      ...getReasoningPayload(
        selectedModel,
        enableThinking,
        reasoningEffort,
        canSearch
      )
    };

    if (canSearch) {
      requestBody.tools =
        FASTCRW_TOOLS;

      requestBody.tool_choice =
        'auto';
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
      console.warn(
        '[FASTCRW] NIM returned no assistant message.'
      );

      return response.data;
    }

    const toolCalls =
      Array.isArray(
        assistantMessage.tool_calls
      )
        ? assistantMessage.tool_calls
        : [];

    // Model answered directly.
    if (toolCalls.length === 0) {
      return response.data;
    }

    workingMessages.push(
      assistantMessage
    );

    let searchesThisRound = 0;

    for (const toolCall of toolCalls) {
      const functionName =
        toolCall.function?.name;

      // Unknown tool.
      if (
        functionName !==
        'search_web'
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

      // Global limit.
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
                'The web-search budget has been exhausted. ' +
                'Use the information already gathered and answer the user.'
            }
          )
        );

        continue;
      }

      // Per-round limit.
      if (
        searchesThisRound >=
        MAX_SEARCH_CALLS_PER_ROUND
      ) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,

              error:
                'Only one focused web search is allowed in this round. ' +
                'Inspect the existing results instead.'
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
                'Invalid search tool arguments.'
            }
          )
        );

        continue;
      }

      const query =
        cleanSearchQuery(
          args.query
        );

      if (
        !isValidSearchQuery(query)
      ) {
        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,

              error:
                `Search query must contain between 2 and ` +
                `${MAX_SEARCH_QUERY_LENGTH} characters.`
            }
          )
        );

        continue;
      }

      // Duplicate protection.
      if (
        isDuplicateOrSimilarQuery(
          query,
          searchedQueries
        )
      ) {
        console.log(
          `[FASTCRW] Blocked duplicate query: "${query}"`
        );

        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,

              error:
                'A substantially identical search has already been performed. ' +
                'Use those results instead of searching again.'
            }
          )
        );

        continue;
      }

      const limit =
        clamp(
          Number(args.limit) || 3,
          1,
          MAX_SEARCH_RESULTS
        );

      const allowedTimeRanges = new Set([
        'hour',
        'day',
        'week',
        'month',
        'year',
        'any'
      ]);

      const timeRange =
        allowedTimeRanges.has(
          args.time_range
        )
          ? args.time_range
          : 'any';

      // Pace searches.
      await waitForSearchCooldown(
        lastSearchAt
      );

      // Mark the search time immediately before
      // making the request, not after it.
      lastSearchAt = Date.now();

      totalSearches++;
      searchesThisRound++;

      searchedQueries.push(query);

      if (
        searchedQueries.length >
        MAX_QUERY_HISTORY
      ) {
        searchedQueries.shift();
      }

      console.log(
        `[FASTCRW] Search ` +
        `${totalSearches}/${MAX_SEARCHES_PER_REQUEST}: ` +
        `"${query}"`
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
        console.error(
          '[FASTCRW] Search failed:',
          error.message
        );

        workingMessages.push(
          makeToolResult(
            toolCall.id,
            {
              success: false,

              error:
                'The web search failed. ' +
                'Continue using the available conversation context.'
            }
          )
        );
      }
    }

    // Let the model inspect the search result and decide
    // whether it genuinely needs another search.
    //
    // The next round is still subject to both hard limits.
    if (
      totalSearches <
        MAX_SEARCHES_PER_REQUEST &&
      round + 1 <
        MAX_TOOL_ROUNDS
    ) {
      continue;
    }

    break;
  }

  console.log(
    '[FASTCRW] Search/tool budget exhausted. ' +
    'Starting final synthesis.'
  );

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

// ===========================================================================
// STANDARD NIM COMPLETION
// ===========================================================================

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

    ...getReasoningPayload(
      selectedModel,
      enableThinking,
      reasoningEffort,
      hasTools
    )
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

// ===========================================================================
// OPENAI RESPONSE NORMALIZATION
// ===========================================================================

function buildOpenAIResponse(
  nimData,
  requestedModel,
  actualModel
) {
  return {
    id:
      nimData?.id ||
      `chatcmpl-${Date.now()}`,

    object:
      'chat.completion',

    created:
      nimData?.created ||
      Math.floor(
        Date.now() / 1000
      ),

    model:
      requestedModel ||
      actualModel,

    choices:
      Array.isArray(
        nimData?.choices
      )
        ? nimData.choices.map(
            (choice, index) => ({
              ...normalizeNonStreamChoice(
                choice,
                actualModel
              ),

              index
            })
          )
        : [],

    usage:
      nimData?.usage || {
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
    ...(message || {})
  };

  const reasoning =
    finalMessage.reasoning ||
    '';

  let content =
    finalMessage.content ||
    '';

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

  return finalMessage;
}

// ===========================================================================
// SSE HELPERS
// ===========================================================================

function setSSEHeaders(res) {
  res.setHeader(
    'Content-Type',
    'text/event-stream'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache, no-transform'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  res.setHeader(
    'X-Accel-Buffering',
    'no'
  );
}

function sendSSE(
  res,
  payload
) {
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

// ===========================================================================
// FASTCRW COMPLETED RESPONSE -> SSE
// ===========================================================================

function streamCompletedResponse(
  res,
  responseData
) {
  const id =
    responseData?.id ||
    `chatcmpl-${Date.now()}`;

  const created =
    responseData?.created ||
    Math.floor(
      Date.now() / 1000
    );

  const model =
    responseData?.model ||
    'unknown';

  const choice =
    responseData?.choices?.[0];

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

// ===========================================================================
// NIM STREAM HANDLER
// ===========================================================================

function handleNIMStream({
  req,
  res,
  upstreamStream,
  usedModel
}) {
  setSSEHeaders(res);

  const decoder =
    new StringDecoder('utf8');

  const normalizer =
    new StreamNormalizer(
      usedModel
    );

  let buffer = '';
  let doneSent = false;
  let cleanEnd = false;
  let cleanedUp = false;
  let reasoningOpen = false;

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

  const sendStreamError = (
    message,
    type = 'stream_error'
  ) => {
    if (res.writableEnded) {
      return;
    }

    sendSSE(res, {
      error: {
        message,
        type
      }
    });
  };

  const processLine = line => {
    // SSE comments / blank lines.
    if (
      !line ||
      !line.startsWith('data:')
    ) {
      return;
    }

    const raw =
      line.slice(5).trim();

    if (!raw) {
      return;
    }

    if (raw === '[DONE]') {
      finish();
      return;
    }

    try {
      const data =
        JSON.parse(raw);

      const delta =
        data?.choices?.[0]?.delta;

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

      let clientContent = '';

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

      const outputChoices =
        Array.isArray(data.choices)
          ? data.choices.map(
              (choice, index) => {
                if (index !== 0) {
                  return choice;
                }

                const outputDelta = {
                  ...(choice.delta || {})
                };

                if (clientContent) {
                  outputDelta.content =
                    clientContent;
                } else {
                  delete outputDelta.content;
                }

                if (
                  SHOW_REASONING &&
                  normalized.reasoning
                ) {
                  outputDelta.reasoning =
                    normalized.reasoning;

                  outputDelta.reasoning_content =
                    normalized.reasoning;
                } else {
                  delete outputDelta.reasoning;
                  delete outputDelta.reasoning_content;
                }

                return {
                  ...choice,
                  delta: outputDelta
                };
              }
            )
          : [];

      sendSSE(res, {
        ...data,
        choices: outputChoices
      });
    } catch (error) {
      console.warn(
        '[STREAM] Invalid JSON chunk:',
        raw.slice(0, 100)
      );

      sendStreamError(
        'Upstream sent malformed chunk',
        'stream_parse_error'
      );
    }
  };

  upstreamStream.on(
    'data',
    chunk => {
      if (doneSent) {
        return;
      }

      buffer +=
        decoder.write(chunk);

      if (
        buffer.length >
        MAX_BUFFER_SIZE
      ) {
        console.error(
          '[STREAM] Buffer overflow.'
        );

        sendStreamError(
          'Stream buffer overflow'
        );

        upstreamStream.destroy();

        finish();

        return;
      }

      const lines =
        buffer.split(/\r?\n/);

      buffer =
        lines.pop() || '';

      for (const line of lines) {
        processLine(line);

        if (doneSent) {
          break;
        }
      }
    }
  );

  upstreamStream.on(
    'end',
    () => {
      if (doneSent) {
        return;
      }

      buffer +=
        decoder.end();

      if (buffer) {
        for (
          const line of
          buffer.split(/\r?\n/)
        ) {
          processLine(line);

          if (doneSent) {
            return;
          }
        }
      }

      if (doneSent) {
        return;
      }

      const flushed =
        normalizer.flush();

      let finalContent = '';

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
              index: 0,

              delta: {
                content:
                  finalContent
              },

              finish_reason: null
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
        sendStreamError(
          'Stream interrupted by upstream error'
        );

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
      if (!cleanEnd) {
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

// ===========================================================================
// MODEL VALIDATION
// ===========================================================================

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
            VALIDATION_TIMEOUT_MS,

          validateStatus: status =>
            status >= 200 &&
            status < 300
        }
      );

    const available =
      new Set(
        Array.isArray(
          response.data?.data
        )
          ? response.data.data
              .map(model => model.id)
              .filter(Boolean)
          : []
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

    // Validation is deliberately non-fatal.
    // A temporary NIM /models failure should not
    // prevent the proxy itself from starting.
    console.warn(
      '[VALIDATION] Continuing without validation.'
    );
  }
}

// ===========================================================================
// DISCORD ALERT
// ===========================================================================

async function sendDiscordAlert(
  invalidModels
) {
  if (
    !DISCORD_WEBHOOK_URL ||
    !Array.isArray(invalidModels) ||
    !invalidModels.length
  ) {
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
        timeout: 5000,

        validateStatus: status =>
          status >= 200 &&
          status < 300
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

// ===========================================================================
// ERROR HELPERS
// ===========================================================================

function getUpstreamErrorMessage(
  error
) {
  const upstreamMessage =
    error.response?.data?.error?.message ||
    error.response?.data?.message;

  if (typeof upstreamMessage === 'string') {
    return upstreamMessage;
  }

  return error.message ||
    'Upstream request failed.';
}

function sendProxyError(
  res,
  error,
  fallbackStatus = 500
) {
  if (res.headersSent) {
    return false;
  }

  const status =
    Number(error.response?.status) ||
    fallbackStatus;

  return res
    .status(status)
    .json({
      error: {
        message:
          getUpstreamErrorMessage(
            error
          ),

        type:
          'proxy_error',

        code:
          status
      }
    });
}

// ===========================================================================
// HEALTH
// ===========================================================================

app.get(
  '/health',
  (req, res) => {
    res.json({
      status: 'ok',
      version: '4.1.0'
    });
  }
);

// ===========================================================================
// MODEL LIST
// ===========================================================================

app.get(
  '/v1/models',
  (req, res) => {
    const created =
      Math.floor(
        Date.now() / 1000
      );

    res.json({
      object: 'list',

      data:
        Object.keys(
          MODEL_MAPPING
        ).map(id => ({
          id,

          object:
            'model',

          created,

          owned_by:
            'nim-proxy'
        }))
    });
  }
);

// ===========================================================================
// MAIN CHAT COMPLETION ROUTE
// ===========================================================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    let upstreamStream = null;

    try {
      const body =
        req.body || {};

      const {
        model,
        messages,
        temperature,
        max_tokens,
        stream
      } = body;

      // ---------------------------------------------------------------------
      // Validate request
      // ---------------------------------------------------------------------

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

      if (!model) {
        return res.status(400).json({
          error: {
            message:
              'model is required',

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

      // ---------------------------------------------------------------------
      // Reasoning settings
      // ---------------------------------------------------------------------

      const enableThinking =
        ENABLE_THINKING_MODE;

      const reasoningEffort =
        body.reasoning_effort ||
        body.reasoningEffort ||
        undefined;

      const inlineReasoning =
        req.headers[
          'x-reasoning-format'
        ] === 'inline';

      const wantsStream =
        Boolean(stream);

      // ---------------------------------------------------------------------
      // FASTCRW
      //
      // Important:
      // FASTCRW itself is non-streaming internally.
      // Once the final answer is produced, it can still be emitted as
      // an OpenAI-compatible SSE response to clients requesting stream=true.
      // ---------------------------------------------------------------------

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

        if (!wantsStream) {
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

      // ---------------------------------------------------------------------
      // Normal NIM request
      // ---------------------------------------------------------------------

      const result =
        await callNIM({
          selectedModel,

          messages,

          temperature,

          max_tokens,

          stream:
            wantsStream,

          enableThinking,

          reasoningEffort,

          hasTools: false
        });

      upstreamStream =
        result.response.data;

      // ---------------------------------------------------------------------
      // Non-streaming
      // ---------------------------------------------------------------------

      if (!wantsStream) {
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

      // ---------------------------------------------------------------------
      // Streaming
      // ---------------------------------------------------------------------

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

      // ---------------------------------------------------------------------
      // Error before SSE headers were sent
      // ---------------------------------------------------------------------

      if (!res.headersSent) {
        return sendProxyError(
          res,
          error
        );
      }

      // ---------------------------------------------------------------------
      // Error after SSE started
      // ---------------------------------------------------------------------

      if (!res.writableEnded) {
        sendSSE(res, {
          error: {
            message:
              getUpstreamErrorMessage(
                error
              ),

            type:
              'proxy_error'
          }
        });

        if (!res.writableEnded) {
          sendDone(res);
        }

        if (!res.writableEnded) {
          res.end();
        }
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

// ===========================================================================
// 404
// ===========================================================================

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

// ===========================================================================
// EXPRESS ERROR HANDLER
// ===========================================================================

app.use(
  (error, req, res, next) => {
    console.error(
      '[EXPRESS] Unhandled error:',
      error.message
    );

    if (res.headersSent) {
      return next(error);
    }

    return res.status(500).json({
      error: {
        message:
          'Internal proxy error',

        type:
          'proxy_error',

        code: 500
      }
    });
  }
);

// ===========================================================================
// STARTUP
// ===========================================================================

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
      `[PROXY] FASTCRW: ` +
      `${ENABLE_WEB_SEARCH ? 'enabled' : 'disabled'}`
    );

    console.log(
      `[PROXY] FASTCRW limits: ` +
      `${MAX_TOOL_ROUNDS} rounds / ` +
      `${MAX_SEARCHES_PER_REQUEST} searches / ` +
      `${MAX_SEARCH_CALLS_PER_ROUND} per round`
    );

    console.log(
      `[PROXY] Search cooldown: ` +
      `${MIN_SEARCH_INTERVAL_MS}ms`
    );

    console.log(
      '[PROXY] Retry profiles: ' +
      'default + conservative GLM/M3/Kimi'
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
