// controllers/chatController.js — FINAL PRODUCTION VERSION + DEBUG LOGS
// ChatGPT-style memory + cache + smart fallback + IUI meaning-match v3

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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 🟣 DEBUG SWITCH — turn off in production
const DEBUG = true;
const log = (...msg) => DEBUG && console.log("[DEBUG]", ...msg);

// 🧠 MEMORY (ChatGPT-like follow-up understanding)
let memory = "";

// ⚡ EMBEDDING CACHE (10x faster performance)
const EMB_CACHE = new Map();

const EMB_PATH = path.join(__dirname, "..", "rag", "embeddings.json");
let EMBEDDINGS = [];

/* -----------------------------------------------------------
   LOAD EMBEDDINGS
----------------------------------------------------------- */
function loadEmbeddingsOnce() {
  if (EMBEDDINGS.length > 0) return;

  if (!fs.existsSync(EMB_PATH)) {
    EMBEDDINGS = [];
    return;
  }

  try {
    EMBEDDINGS = JSON.parse(fs.readFileSync(EMB_PATH, "utf8"));
    log(`Loaded ${EMBEDDINGS.length} embeddings`);
  } catch {
    EMBEDDINGS = [];
  }
}

/* -----------------------------------------------------------
   SMART FALLBACK (School-safe reply)
----------------------------------------------------------- */
async function smartFallback(question) {
  const prompt = `
If this question is about Montfort ICSE School but not in dataset:
Reply EXACTLY:
"I don’t have that information in my data. Please visit https://montforticse.in/ or contact the school office for official details."

If general question (not about school):
Answer normally with a short helpful reply.

Question: "${question}"
`;
  return await answerGeneralQuestion(prompt);
}

/* -----------------------------------------------------------
   LLM MEANING MATCH (v3)
----------------------------------------------------------- */
async function llmMeaningMatch(userQ, faqQ) {
  const prompt = `
Determine if these two questions have the SAME MEANING.

You MUST treat the following as SAME meaning:
- "school" = "campus"
- "hostel food" = "mess food" = "food for hostel students"
- "canteen" = "food area" = "snacks place"
- Grammar differences do NOT change meaning
- Word order differences do NOT change meaning
- Spelling mistakes do NOT change meaning
- Missing helper words ("in", "at", "for") do NOT change meaning

If both questions are about the same topic,
reply EXACTLY: "yes"

If NOT same topic, reply EXACTLY: "no"

User question: "${userQ}"
FAQ question: "${faqQ}"

Reply ONLY "yes" or "no".
`;

  const out = await answerGeneralQuestion(prompt);
  const ans = (out || "").trim().toLowerCase();

  log("LLM Meaning Match:", ans);

  return (
    ans === "yes" ||
    ans.startsWith("yes") ||
    ans.includes("yes")
  );
}

/* -----------------------------------------------------------
   MAIN CHAT HANDLER (FINAL PIPELINE)
----------------------------------------------------------- */
export async function handleChat(req, res) {
  try {
    loadEmbeddingsOnce();

    let { question } = req.body || {};
    if (!question || typeof question !== "string") {
      return res.json({ answer: "question is required" });
    }

    question = question.trim();
    log("USER:", question);

    /* 1) MEMORY */
    let finalUser = question;

    if (memory && finalUser.length <= 4) {
      finalUser = memory + " " + finalUser;
      log("MEMORY APPLIED:", finalUser);
    }
    memory = question;

    /* 2) SPELL FIX */
    const corrected = await correctSpelling(finalUser);
    log("SPELL:", corrected);

    /* 3) NORMALIZED */
    const normalized = await normalizeToMeaning(corrected);
    log("NORM:", normalized);

    /* 4) EMBEDDINGS + CACHE */
    let vector = EMB_CACHE.get(normalized);

    if (vector) {
      log("EMBEDDING CACHE HIT");
    } else {
      log("GENERATING EMBEDDING...");
      vector = await embedText(normalized);
      EMB_CACHE.set(normalized, vector);
    }

    if (!vector.length) {
      log("NO EMBEDDING → FALLBACK");
      return res.json({
        answer: await smartFallback(normalized),
        via: "no-embedding",
      });
    }

    if (!EMBEDDINGS.length) {
      log("NO EMBEDDINGS FILE → FALLBACK");
      return res.json({
        answer: await smartFallback(normalized),
        via: "no-embeddings-file",
      });
    }

    /* 5) SEMANTIC SEARCH */
    const matches = findTopMatches(vector, EMBEDDINGS, normalized, 5);
    const best = matches[0];
    const second = matches[1];

    log("BEST MATCH:", best);
    log("SECOND MATCH:", second);

    if (!best) {
      log("NO MATCH");
      return res.json({
        answer: await smartFallback(normalized),
        via: "no-match",
      });
    }

    /* 6) SCORE CHECK */
    const MIN_SCORE = 0.10;
    const GAP = 0.05;

    const lowScore = best._score < MIN_SCORE;
    const ambiguous = second && Math.abs(best._score - second._score) < GAP;

    log("BEST SCORE:", best._score, "LOW?", lowScore, "AMBIG?", ambiguous);

    /* 7) LLM VALIDATION */
    if (lowScore || ambiguous) {
      log("RUNNING LLM MEANING VALIDATION...");
      const same = await llmMeaningMatch(normalized, best.question);

      if (same) {
        log("LLM CONFIRMED MATCH");
        return res.json({
          answer: best.answer,
          via: "llm-validated-match",
        });
      }

      log("LLM REJECTED → FALLBACK");
      return res.json({
        answer: await smartFallback(normalized),
        via: lowScore ? "low-score" : "ambiguous",
      });
    }

    /* 8) DIRECT MATCH */
    log("SEMANTIC MATCH SUCCESS");
    return res.json({
      answer: best.answer,
      via: "semantic-match",
    });

  } catch (err) {
    console.error("ERROR:", err);

    return res.json({
      answer: await smartFallback(req.body?.question || ""),
      via: "error",
    });
  }
}
