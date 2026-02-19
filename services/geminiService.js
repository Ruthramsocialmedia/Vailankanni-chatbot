// services/geminiService.js — IUI Engine v2.5
// Advanced Input Understanding for Montfort Chatbot
// - Safe spelling (no noun swaps)
// - Word-split + local vocab spell-fix
// - Meaning-preserving normalization
// - School-safe fallback
// - Embeddings for semantic search

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { ENV } from "../config/env.js";

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models";
const EMBED_MODEL = "gemini-embedding-001";
const CHAT_MODEL = ENV.GEMINI_MODEL || "gemini-1.5-flash";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Vocabulary file used for word-split + local spell-fix
const VOCAB_PATH = path.join(
  __dirname,
  "..",
  "rag",
  "school-data-understood.json"
);

let VOCAB_SET = null;
let VOCAB_WORDS = null;

/* ---------------------------------------------------------
   LOAD VOCAB (for local spell-fix & word split)
--------------------------------------------------------- */
function loadVocabOnce() {
  if (VOCAB_SET) return;

  try {
    const data = JSON.parse(fs.readFileSync(VOCAB_PATH, "utf8"));

    const set = new Set();
    for (const item of data) {
      const text = `${item.keyword || ""} ${item.question || ""}`;
      text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/)
        .filter(Boolean)
        .forEach((t) => {
          if (t.length >= 3) set.add(t);
        });
    }

    VOCAB_SET = set;
    VOCAB_WORDS = Array.from(set);
    console.log(
      `[GeminiService] Loaded ${VOCAB_WORDS.length} vocabulary words.`
    );
  } catch (err) {
    console.error("[GeminiService] Failed to load vocabulary:", err);
    VOCAB_SET = new Set();
    VOCAB_WORDS = [];
  }
}

/* ---------------------------------------------------------
   HELPER: Levenshtein distance
--------------------------------------------------------- */
function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

/* ---------------------------------------------------------
   CORE GEMINI CALLER
--------------------------------------------------------- */
async function callGemini(prompt, instruction = "") {
  if (!ENV.GEMINI_API_KEY) return "";

  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${CHAT_MODEL}:generateContent?key=${ENV.GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [
                {
                  text: instruction
                    ? `${instruction}\n\nUSER: ${prompt}`
                    : prompt,
                },
              ],
            },
          ],
          generationConfig: {
            temperature: 0.1, // stable, not creative
            maxOutputTokens: 180,
          },
        }),
      }
    );

    if (!res.ok) {
      console.error("[Gemini] HTTP error:", res.status);
      return "";
    }

    const data = await res.json();
    return data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
  } catch (err) {
    console.error("[Gemini] callGemini error:", err);
    return "";
  }
}

/* ---------------------------------------------------------
   LAYER 0 — Basic pre-clean
--------------------------------------------------------- */
function preClean(text) {
  if (!text) return "";
  return text
    .replace(/[\u200B-\u200D\uFEFF]/g, "") // zero-width chars
    .replace(/[“”„]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[\u{1F300}-\u{1FAFF}]/gu, "") // emojis
    .replace(/\s+/g, " ")
    .trim();
}

/* ---------------------------------------------------------
   LAYER 1 — Word-split fix (merged words)
   Examples:
   - "waterin"    → "water in"
   - "canteenin"  → "canteen in"
   - "hostelstudents" → "hostel students"
--------------------------------------------------------- */
function splitMergedWords(text) {
  loadVocabOnce();
  if (!VOCAB_WORDS.length) return text;

  return text
    .split(/\s+/)
    .map((word) => {
      const lower = word.toLowerCase();
      if (VOCAB_SET.has(lower)) return word;

      // Try splitting into 2 vocab words
      for (let i = 3; i < lower.length - 2; i++) {
        const left = lower.slice(0, i);
        const right = lower.slice(i);

        if (VOCAB_SET.has(left) && VOCAB_SET.has(right)) {
          return `${left} ${right}`;
        }
      }

      return word;
    })
    .join(" ");
}

/* ---------------------------------------------------------
   LAYER 2 — Local vocab-based spell-fix (safe)
   Important:
   - We SKIP very short words (length < 5) like "text", "test"
     to avoid "text" → "test" mistakes.
--------------------------------------------------------- */
function localSpellFix(text) {
  loadVocabOnce();
  if (!VOCAB_WORDS.length) return text;

  return text
    .split(/\s+/)
    .map((token) => {
      const t = token.toLowerCase();

      // keep very short / non-alpha tokens as is
      if (!/^[a-z]+$/.test(t) || t.length < 5) return token;

      // already known word
      if (VOCAB_SET.has(t)) return token;

      let best = t;
      let bestDist = Infinity;

      for (const v of VOCAB_WORDS) {
        const d = levenshtein(t, v);
        if (d < bestDist) {
          bestDist = d;
          best = v;
        }
      }

      // Allow up to ~40% of length as distance
      const maxDist = Math.max(1, Math.round(t.length * 0.4));

      if (bestDist <= maxDist) {
        if (token[0] === token[0].toUpperCase()) {
          return best.charAt(0).toUpperCase() + best.slice(1);
        }
        return best;
      }

      return token;
    })
    .join(" ");
}

/* ---------------------------------------------------------
   EXPORT: SPELLING CORRECTION (IUI v2.5)
   Pipeline:
   1) preClean
   2) splitMergedWords
   3) localSpellFix
   4) LLM spelling (strict, no noun swaps)
--------------------------------------------------------- */
export async function correctSpelling(text) {
  if (!text) return "";

  // 1. basic cleaning
  let processed = preClean(text);

  // 2. fix merged words
  processed = splitMergedWords(processed);

  // 3. local vocab-based spell-fix (safe)
  processed = localSpellFix(processed);

  const originalTokens = Array.from(
    new Set(processed.split(/\s+/).map((t) => t.toLowerCase()))
  );

  const inst = `
You are a STRICT spelling corrector for school-related questions.

GOAL:
- Fix ONLY spelling mistakes.
- Optionally fix very small grammar issues (like "is they" → "are they").
- Preserve the original meaning 100%.

VERY IMPORTANT:
- DO NOT replace one noun with a DIFFERENT noun.
- DO NOT change "text books" to "test blocks" or anything similar.
- DO NOT invent new words that were not typed by the user.
- DO NOT add new concepts, places, or items.
- DO NOT remove important words.

You MAY:
Correct spelling mistakes exactly, but do not alter the meaning, replace nouns, add new words, or change the user’s intent. Fix only obvious typos, such as “texxt” → “text”, “bukks” → “books”, “conatct” → “contact”, "conatct" → "contact", "deils" → "details".
- Combine obvious pairs: "text books" → "textbooks" (same meaning).
- Fix simple English grammar if needed.

The following logic words MUST NOT CHANGE MEANING:
- "did not"
- "don't"
- "didn't"
- "not"
- "follow"
- "break"
- "obey"
- "rules"

You MUST treat these as FIXED MEANING WORDS.

These original user words must stay the SAME CONCEPT
(you may only fix spelling or spacing for them):
${originalTokens.join(", ")}

Return ONLY the corrected sentence, nothing else.
`;

  try {
    const out = await callGemini(processed, inst);
    if (!out) return processed;

    const trimmed = out.trim();

    const outWords = trimmed.split(/\s+/).length;
    const origWords = processed.split(/\s+/).length;

    // If Gemini shrinks too much, something went wrong → keep processed
    if (outWords < origWords - 3) return processed;

    return trimmed;
  } catch {
    return processed;
  }
}

/* ---------------------------------------------------------
   EXPORT: MEANING NORMALIZER
   Rewrites into a clean English question, preserving meaning.
--------------------------------------------------------- */
export async function normalizeToMeaning(text) {
  if (!text) return "";

  const cleaned = preClean(text);
  const tokens = cleaned.split(/\s+/).filter(Boolean);
  const nounCandidates = tokens.filter((t) => t.length > 3);

  // If too short → don't risk rewrite
  if (tokens.length <= 3) return cleaned;

  const inst = `
Rewrite the user's message as a clean English QUESTION.

VERY IMPORTANT:
You must NOT change the user's meaning under ANY condition.

STRICT RULES (DO NOT BREAK):
1. DO NOT change or replace ANY noun.
2. DO NOT introduce ANY new noun or verb.
3. DO NOT guess the user's meaning or intention.
4. DO NOT remove important words.
5. DO NOT change the action:
   - follow
   - break
   - obey
   - allow
   - rules
6. DO NOT change words like:
   - what
   - if
   - happen / happens
   - did / did not / didn't
7. You may ONLY:
   - Add helper words ("the", "does", "do", "is", "are")
   - Fix tiny grammar issues
   - Reorder existing words WITHOUT changing meaning

If the sentence is too unclear to safely rewrite,
RETURN the cleaned version as-is.

These words represent important concepts and MUST stay the SAME CONCEPT:
${nounCandidates.join(
    ", "
  )}, follow, rule, rules, break, obey, happen, happens, did, didn't, "did not"

Return ONLY the meaning-preserved corrected question, one sentence, no explanation.
`;

  try {
    const out = await callGemini(cleaned, inst);
    if (!out) return cleaned;

    const result = out.trim();

    // Safety check: output must not be too short
    if (result.split(/\s+/).length <= 3) return cleaned;

    return result;
  } catch {
    return cleaned;
  }
}

/* ---------------------------------------------------------
   EXPORT: SUMMARY
--------------------------------------------------------- */
export async function summarizeAnswers(text) {
  const inst = `
Summarize the following Q&A pairs into short, simple bullet points.
Do NOT add any new facts.
Return ONLY the bullet list.
`;

  const out = await callGemini(text, inst);
  return out || text;
}

/* ---------------------------------------------------------
   EXPORT: GENERAL QUESTION FALLBACK (school-safe)
--------------------------------------------------------- */
export async function answerGeneralQuestion(text) {
  const inst = `
You are a safe assistant for Montfort ICSE School.

If the question is about Montfort ICSE School but the exact information
is NOT present in the school's official data or FAQ, you MUST reply EXACTLY:

"I don’t have that information in my data. Please visit https://montforticse.in/ or contact the school office for official details."

If the question is clearly about something ELSE (not Montfort ICSE School),
you may answer normally with a brief, helpful reply.

Return ONLY the final answer text.
`;

  const out = await callGemini(text, inst);

  return (
    out ||
    "I don’t have that information in my data. Please visit https://montforticse.in/ or contact the school office for official details."
  );
}

/* ---------------------------------------------------------
   EXPORT: EMBEDDINGS
--------------------------------------------------------- */
export async function embedText(text) {
  if (!ENV.GEMINI_API_KEY) return [];

  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${ENV.GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: { parts: [{ text }] },
        }),
      }
    );

    if (!res.ok) {
      const errText = await res.text();
      console.error(`[Gemini] embedText HTTP error: ${res.status}`, errText);
      return [];
    }

    const data = await res.json();
    return data?.embedding?.values || [];
  } catch (err) {
    console.error("[Gemini] embedText error:", err);
    return [];
  }
}
