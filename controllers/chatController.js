// controllers/chatController.js — FINAL STABLE v3.0
// Includes:
// ✓ AI pano/project routing
// ✓ One-word multi-match mode
// ✓ Strong semantic scoring engine
// ✓ Safe LLM confirmation (null handling)
// ✓ Fixes for 429 + fallback text
// ✓ ZERO hallucination

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import {
  correctSpelling,
  normalizeToMeaning,
  embedText,
  answerGeneralQuestion,
} from "../services/geminiService.js";

import { findTopMatches } from "../rag/semantic-search.js";
import { aiIntentRouter } from "./aiIntentRouter.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// DEBUG CONTROL
const DEBUG = true;
const log = (...msg) => DEBUG && console.log("[DEBUG]", ...msg);

// Follow-up memory
let memory = "";

// Embedding cache
const EMB_CACHE = new Map();

// Load embeddings
const EMB_PATH = path.join(__dirname, "..", "rag", "embeddings.json");
let EMBEDDINGS = [];

/* ---------------------------------------------------------
   LOAD EMBEDDINGS
--------------------------------------------------------- */
function loadEmbeddingsOnce() {
  if (EMBEDDINGS.length > 0) return;

  if (!fs.existsSync(EMB_PATH)) {
    EMBEDDINGS = [];
    log("Embeddings file missing");
    return;
  }

  try {
    EMBEDDINGS = JSON.parse(fs.readFileSync(EMB_PATH, "utf8"));
    log(`Loaded ${EMBEDDINGS.length} embeddings`);
  } catch (err) {
    console.error("Error loading embeddings:", err);
    EMBEDDINGS = [];
  }
}

/* ---------------------------------------------------------
   SMART FALLBACK
--------------------------------------------------------- */
async function smartFallback() {
  return (
    "I don’t have that information in my data. " +
    "Please visit https://montforticse.in/ or contact the school office for official details."
  );
}

/* ---------------------------------------------------------
   LLM SAME-MEANING VALIDATION (SAFE)
   RETURNS:
   true  → same meaning
   false → different meaning
   null  → LLM failed (fallback / 429 / weird output)
--------------------------------------------------------- */
async function llmMeaningMatch(userQ, candidateQ) {
  const prompt = `
Do these questions have EXACTLY the same meaning?

Rules:
- Grammar changes DO NOT matter
- Word order DOES NOT matter
- Spelling mistakes DO NOT matter
- Synonyms count as same meaning:
  (school = campus)
  (class start time = school timing)
  (hostel food = mess food)
  (canteen = snack shop)

Reply ONLY:
"yes" or "no"

User: "${userQ}"
Reference: "${candidateQ}"
  `;

  try {
    const out = await answerGeneralQuestion(prompt);
    const ans = (out || "").trim().toLowerCase();

    log("LLM Meaning RAW:", ans);

    if (ans === "yes") return true;
    if (ans === "no") return false;

    // Detect fallback text
    if (
      ans.includes("montforticse.in") ||
      ans.startsWith("i don’t have that information")
    ) {
      log("[LLM] Fallback detected → returning null");
      return null;
    }

    return null;
  } catch (err) {
    console.error("[LLM] Error:", err);
    return null;
  }
}

/* ---------------------------------------------------------
   MAIN CHAT HANDLER
--------------------------------------------------------- */
export async function handleChat(req, res) {
  try {
    loadEmbeddingsOnce();

    let { question, panoNames = [], projectNames = [] } = req.body || {};

    if (!question || typeof question !== "string") {
      return res.json({ answer: "question is required" });
    }

    question = question.trim();
    log("USER:", question);

    /* ============================================================
       0) AI ROUTER → PANORAMA / PROJECT / SCHOOL
    ============================================================ */
    const aiIntent = aiIntentRouter(question, panoNames, projectNames);
    log("AI ROUTER:", aiIntent);

    if (aiIntent.intent === "pano") {
      return res.json({ intent: "pano", target: aiIntent.target });
    }

    if (aiIntent.intent === "project") {
      return res.json({ intent: "project", target: aiIntent.target });
    }

    /* ============================================================
       1) FOLLOW-UP MEMORY
    ============================================================ */
    let finalUser = question;

    if (memory && finalUser.length <= 4) {
      finalUser = memory + " " + finalUser;
      log("MEMORY MERGED:", finalUser);
    }

    memory = question;

    /* ============================================================
       2) SPELL FIX
    ============================================================ */
    const corrected = await correctSpelling(finalUser);
    log("SPELL:", corrected);

    /* ============================================================
       3) NORMALIZE
    ============================================================ */
    const normalized = await normalizeToMeaning(corrected);
    log("NORM:", normalized);

    /* ============================================================
       4) EMBED (CACHE)
    ============================================================ */
    let vector = EMB_CACHE.get(normalized);

    if (!vector) {
      log("GENERATING EMBEDDING…");
      vector = await embedText(normalized);
      EMB_CACHE.set(normalized, vector);
    }

    if (!vector.length)
      return res.json({ answer: await smartFallback(), via: "no-vector" });

    /* ============================================================
       5) SEMANTIC SEARCH
    ============================================================ */
    const matches = findTopMatches(vector, EMBEDDINGS, normalized, 5);
    const best = matches[0];
    const second = matches[1];

    log("BEST:", best);
    log("SECOND:", second);

    if (!best) {
      return res.json({ answer: await smartFallback(), via: "no-match" });
    }

    /* ============================================================
       6) MULTI-MATCH (ONE WORD MODE)
    ============================================================ */
    const tokenCount = normalized.split(/\s+/).length;

    if (tokenCount <= 1) {
      const list = matches
        .filter((m) => m._score >= 0.08)
        .map((m) => `• ${m.answer}`)
        .join("\n\n");

      if (list.trim()) {
        return res.json({ answer: list, via: "multi-match" });
      }
    }

    /* ============================================================
       7) SCORE VALIDATION
    ============================================================ */
    const MIN = 0.11;
    const GAP = 0.06;
    const low = best._score < MIN;
    const ambi =
      second && Math.abs(best._score - second._score) < GAP;

    log("Score:", best._score, "LOW?", low, "AMBIG?", ambi);

    /* ============================================================
       8) LLM VALIDATION (SAFE)
    ============================================================ */
    if (low || ambi) {
      log("LLM VALIDATING…");
      const ok = await llmMeaningMatch(normalized, best.question);

      // CASE 1: LLM CLEAR YES → accept
      if (ok === true) {
        return res.json({
          answer: best.answer,
          via: "llm-validated",
        });
      }

      // CASE 2: LLM CLEAR NO → fallback
      if (ok === false) {
        return res.json({
          answer: await smartFallback(),
          via: "llm-reject",
        });
      }

      // CASE 3: LLM FAILED → TRUST SCORE IF STRONG
      if (ok === null) {
        const TRUST = 0.55;
        if (!low && best._score >= TRUST) {
          return res.json({
            answer: best.answer,
            via: "semantic-llm-unavailable",
          });
        }

        // NOT SAFE → fallback
        return res.json({
          answer: await smartFallback(),
          via: "llm-unavailable",
        });
      }
    }

    /* ============================================================
       9) DIRECT SEMANTIC MATCH
    ============================================================ */
    return res.json({
      answer: best.answer,
      via: "semantic",
    });

  } catch (err) {
    console.error("ERROR:", err);
    return res.json({
      answer: await smartFallback(),
      via: "error",
    });
  }
}
