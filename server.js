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

const SHOW_REASONING = process.env.SHOW_REASONING === 'false';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
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
  Math.max(parseInt(process.env.MAX_TOOL_ROUNDS || '3', 10), 1),
  5
);

const FASTCRW_TIMEOUT_MS = 30000;
// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => {
    console.error(`[FATAL] ${msg}`);
    process.exit(1);
  };

  if (!NIM_API_KEY) {
    fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  }

  if (!CLIENT_AUTH_KEY) {
    console.warn(
      '[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.'
    );
  }

  if (ENABLE_WEB_SEARCH && !FASTCRW_API_KEY) {
    fatal('FASTCRW_API_KEY is required when ENABLE_WEB_SEARCH=true');
  }

  if (ENABLE_WEB_SEARCH) {
    console.log('[CONFIG] Live web search: ENABLED');
    console.log(`[CONFIG] fastCRW: ${FASTCRW_API_BASE}`);
  }
}

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'zai-org/glm-5.2',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.2': 'zai-org/glm-5.2',
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
// ─── Middleware ─────────────────────────────────────────────────────────────

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

// ─── Routes ────────────────────────────────────────────────────────────────
const FASTCRW_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'search_web',
      description:
        'Search the live web for current information. Use this when the user asks about recent events, current prices, current software/library information, news, recent releases, current people/companies, or anything where up-to-date web information would improve accuracy. Do not use it for ordinary timeless questions unless web verification is useful.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The web search query.'
          },
          limit: {
            type: 'integer',
            description: 'Number of results to return. Maximum 8.',
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
        stream: false,

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
    Math.max(Number(args?.limit) || 5, 1),
    8
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
      position: r.position ?? index + 1,
      title: r.title || '',
      url: r.url || '',
      description: r.description || '',
      snippet: r.snippet || '',
      score: r.score ?? null,
      publishedDate: r.publishedDate ?? null
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
      stream: false,

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
          error.message
        );

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
