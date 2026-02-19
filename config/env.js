import dotenv from "dotenv";

dotenv.config();

export const ENV = {
  GEMINI_API_KEY: process.env.GEMINI_API_KEY || "",
  GEMINI_MODEL: process.env.GEMINI_MODEL || "gemini-2.0-flash",
  PORT: process.env.PORT || 3000
};
