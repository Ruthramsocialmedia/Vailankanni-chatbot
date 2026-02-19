import { ENV } from "./config/env.js";
import { embedText, correctSpelling } from "./services/geminiService.js";

async function test() {
    console.log("Testing Gemini API...");
    console.log("API Key present:", !!ENV.GEMINI_API_KEY);

    console.log("1. Testing Chat (correctSpelling)...");
    const spell = await correctSpelling("wod is speling");
    console.log("Chat Result:", spell);

    console.log("2. Testing Embeddings (embedText)...");
    const vec = await embedText("test");
    console.log("Embedding Result Length:", vec.length);
}

test();
