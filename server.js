import express from "express";
import cors from "cors";
import { ENV } from "./config/env.js";
import chatRoute from "./routers/chatRoute.js";

const app = express();

app.use(cors({ origin: "*"})); // In production, restrict origin
app.use(express.json());

app.get("/", (req, res) => {
  res.json({ status: "ok", message: "Montfort Chatbot backend running" });
});

app.use("/api", chatRoute);

app.listen(ENV.PORT, () => {
  console.log(`🚀 Backend listening on http://localhost:${ENV.PORT}`);
});
