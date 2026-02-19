// rag/generate-embeddings.js — INTENT + MULTIVARIATION VERSION (Option A)

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { ENV } from "../config/env.js";
import { embedText } from "../services/geminiService.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const RAW_INPUT_PATH = path.join(__dirname, "school-data.json");  // Intent-based file
const UNDERSTOOD_OUTPUT_PATH = path.join(__dirname, "school-data-understood.json");
const EMBEDDINGS_OUTPUT_PATH = path.join(__dirname, "embeddings.json");

/* ---------------------------------------------------------
   Build embedding entry for *each question variation*
--------------------------------------------------------- */
function buildSemanticEntry(intent, question, answer, index) {
  const q = question.trim();
  const a = answer.trim();

  const semanticBlock = [
    `This is official Montfort ICSE School FAQ content.`,
    `Intent: "${intent}"`,
    `User question variation: "${q}".`,
    `Verified school answer: "${a}".`,
    `This FAQ belongs to category: academics, admission, hostel, canteen, sports, facilities, safety.`,
    `Do NOT change meaning of the answer.`,
  ];

  const tokens = `${q} ${a}`
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length >= 4);

  const keywords = [...new Set(tokens)].join(", ");

  const embeddingText =
    `INTENT: ${intent}\n` +
    `QUESTION: ${q}\n` +
    `ANSWER: ${a}\n\n` +
    `MEANING BLOCK: ${semanticBlock.join(" ")}\n\n` +
    `KEYWORDS: ${keywords}, montfort, icse, school, academics, hostel, admission, facilities`;

  return {
    id: `${intent}_q${index}`,
    intent,
    question: q,
    answer: a,
    embedding_text: embeddingText,
    keyword: keywords,
  };
}

/* ---------------------------------------------------------
   MAIN PROCESS
--------------------------------------------------------- */
async function main() {
  if (!ENV.GEMINI_API_KEY) {
    console.error("❌ GEMINI_API_KEY missing");
    process.exit(1);
  }

  console.log("📥 Loading intent-based school-data.json");
  const raw = fs.readFileSync(RAW_INPUT_PATH, "utf8");
  const items = JSON.parse(raw);  // Array of intents

  const understood = [];
  const embeddings = [];

  let counter = 0;

  for (const intentBlock of items) {
    const intent = intentBlock.intent;
    const answer = intentBlock.answer;

    if (!intent || !answer || !Array.isArray(intentBlock.questions)) {
      console.error("❌ Invalid entry (intent/questions missing):", intentBlock);
      continue;
    }

    // Loop through ALL question variations
    for (const q of intentBlock.questions) {
      const entry = buildSemanticEntry(intent, q, answer, counter);
      understood.push(entry);

      console.log(`🧠 Embedding ${counter + 1}: ${q}`);

      let vector = [];
      try {
        vector = await embedText(entry.embedding_text);
      } catch (err) {
        console.error("❌ Embedding failed", err);
      }

      embeddings.push({
        id: entry.id,
        intent: entry.intent,
        question: entry.question,
        answer: entry.answer,
        keyword: entry.keyword,
        vector,
      });

      counter++;
    }
  }

  fs.writeFileSync(UNDERSTOOD_OUTPUT_PATH, JSON.stringify(understood, null, 2));
  fs.writeFileSync(EMBEDDINGS_OUTPUT_PATH, JSON.stringify(embeddings, null, 2));

  console.log("🎉 SUCCESS! Intent-based embeddings generated.");
  console.log(`Total variations embedded: ${counter}`);
}

main();
