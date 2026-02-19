import { ENV } from "./config/env.js";

async function listModels() {
    const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${ENV.GEMINI_API_KEY}`;
    console.log("Fetching models from:", url.replace(ENV.GEMINI_API_KEY, "HIDDEN_KEY"));

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.models) {
            console.log("\nAvailable Models:");
            data.models.forEach(m => {
                if (m.name.includes("embed")) {
                    console.log(`- ${m.name} (Supported methods: ${m.supportedGenerationMethods})`);
                }
            });
        } else {
            console.log("No models found or error:", data);
        }
    } catch (err) {
        console.error("Error fetching models:", err);
    }
}

listModels();
