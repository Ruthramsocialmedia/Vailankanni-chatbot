// rag/semantic-search.js — INTENT-AWARE SEMANTIC ENGINE (Option A)
// Supports multiple question variations mapping to the same intent.

/* ---------------------------------------------------------
   COSINE SIMILARITY
--------------------------------------------------------- */
export function cosineSimilarity(a, b) {
  if (!a || !b || !a.length || !b.length) return 0;

  let dot = 0,
    normA = 0,
    normB = 0;

  const len = Math.min(a.length, b.length);

  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (!normA || !normB) return 0;

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* ---------------------------------------------------------
   TOKENIZER
--------------------------------------------------------- */
function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/* ---------------------------------------------------------
   BUILD TOKEN STATS (topic clustering)
--------------------------------------------------------- */
function buildTokenStats(entries) {
  const df = new Map();
  let totalDocs = 0;

  for (const e of entries) {
    const text = (e.keyword || e.question || "").toLowerCase();
    const tokens = new Set(tokenize(text));

    if (!tokens.size) continue;

    totalDocs++;

    for (const t of tokens) {
      df.set(t, (df.get(t) || 0) + 1);
    }
  }

  return { df, totalDocs };
}

/* ---------------------------------------------------------
   RARE WORD WEIGHTING (IDF)
--------------------------------------------------------- */
function idf(token, dfMap, totalDocs) {
  const df = dfMap.get(token) || 0;
  return Math.log((totalDocs + 1) / (df + 1)) + 1;
}

/* ---------------------------------------------------------
   KEYWORD SCORE (meaning-based keyword overlap)
--------------------------------------------------------- */
function keywordScore(userTokens, entryTokens, dfMap, totalDocs) {
  if (!userTokens.length || !entryTokens.length || !totalDocs) return 0;

  const entrySet = new Set(entryTokens);
  const uniqueUser = Array.from(new Set(userTokens));

  let num = 0;
  let den = 0;

  for (const t of uniqueUser) {
    const weight = idf(t, dfMap, totalDocs); // rare words get higher weight
    den += weight;

    if (entrySet.has(t)) num += weight;
  }

  if (!den) return 0;

  return num / den;
}

/* ---------------------------------------------------------
   INTENT-AWARE MATCHING
   - Multiple variations per intent collapse to single best entry
   - Best variation = selected result per intent
--------------------------------------------------------- */
export function findTopMatches(userVector, entries, userQuery = "", limit = 5) {
  if (!entries || !entries.length) return [];

  const userTokens = tokenize((userQuery || "").toLowerCase());

  const isShortQuery = userTokens.length <= 2; // Used for weighting
  const { df, totalDocs } = buildTokenStats(entries);

  // Weight balancing tuned for excellent accuracy
  const SEM_WEIGHT = isShortQuery ? 0.70 : 0.65;
  const KEY_WEIGHT = isShortQuery ? 0.30 : 0.35;

  // TEMP STORAGE: intent → best match object
  const intentScores = new Map();

  for (const entry of entries) {
    if (!entry.vector?.length) continue;

    // Semantic similarity
    const semantic = cosineSimilarity(userVector, entry.vector);

    // Keyword score
    const entryTokens = tokenize(entry.keyword || entry.question || "");
    const key = keywordScore(userTokens, entryTokens, df, totalDocs);

    // Small boost when user tokens appear in the result tokens
    const hasOverlap = userTokens.some((t) => entryTokens.includes(t));
    const topicBoost = hasOverlap ? 0.05 : 0;

    // Final combined score
    const score =
      semantic * SEM_WEIGHT +
      key * KEY_WEIGHT +
      topicBoost;

    const intent = entry.intent || entry.id;

    // Keep BEST version per intent
    if (!intentScores.has(intent) || score > intentScores.get(intent)._score) {
      intentScores.set(intent, {
        ...entry,
        _score: score,
      });
    }
  }

  // Convert Map → Array
  const list = Array.from(intentScores.values());

  // Sort highest score first
  return list.sort((a, b) => b._score - a._score).slice(0, limit);
}

/* ---------------------------------------------------------
   BEST SINGLE MATCH
--------------------------------------------------------- */
export function findBestMatch(userVector, entries, userQuery = "") {
  return findTopMatches(userVector, entries, userQuery, 1)[0] || null;
}
