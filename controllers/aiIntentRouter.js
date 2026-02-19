// controllers/aiIntentRouter.js — ULTRA-STABLE FINAL VERSION (UPDATED)
// Navigation-aware + Auto-correct + Fuzzy Matching

export function aiIntentRouter(question, panoNames = [], projectNames = []) {
  if (!question) return { intent: "school" };

  /* ------------------------------------------------------------
     1) NAVIGATION VERB DETECTION (must exist to open pano/project)
  ------------------------------------------------------------ */
  const navRegex = /\b(go to|go|goto|open|show|view|take me to|take me|navigate|visit|see|check|look at)\b/i;
  const userHasNavIntent = navRegex.test(question);

  /* ------------------------------------------------------------
     2) BASIC NORMALIZATION
  ------------------------------------------------------------ */
  let raw = question.toLowerCase().trim();

  // Remove navigation verbs
  raw = raw.replace(
    /\b(go to|go|goto|open|show|view|take me to|take me|navigate|visit|see|check|look at)\b/g,
    ""
  );

  // Remove filler words
  raw = raw.replace(/\b(the|a|an|please|pls|kindly|can you|could you)\b/g, "");

  // Remove punctuation + numbers
  let cleaned = raw
    .replace(/[^\w\s]/g, "")
    .replace(/\d+/g, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleaned) return { intent: "school" };

  /* ------------------------------------------------------------
     3) AUTO-CORRECT TARGET NAME (Fix spelling mistakes)
  ------------------------------------------------------------ */
  function autoCorrectTarget(name, panoList, projectList) {
    const all = [...panoList, ...projectList];
    const target = name.toLowerCase();

    let best = target;
    let bestDist = 999;

    for (const item of all) {
      const d = levenshteinDistance(item.toLowerCase(), target);
      if (d < bestDist) {
        bestDist = d;
        best = item.toLowerCase();
      }
    }

    // Allow corrections up to distance 3
    if (bestDist <= 3) return best;
    return target;
  }

  // Apply auto-correct ONLY when user intends navigation
  if (userHasNavIntent) {
    cleaned = autoCorrectTarget(cleaned, panoNames, projectNames);
  }

  /* ------------------------------------------------------------
     4) STOP HERE if no navigation verb → treat as SCHOOL query
  ------------------------------------------------------------ */
  if (!userHasNavIntent) {
    return { intent: "school" };
  }

  /* ------------------------------------------------------------
     5) EXACT PANORAMA MATCH
  ------------------------------------------------------------ */
  const panoExact = panoNames.find(
    (p) => p.toLowerCase() === cleaned
  );

  if (panoExact) {
    return { intent: "pano", target: panoExact };
  }

  /* ------------------------------------------------------------
     6) PANORAMA — STRONG FUZZY MATCH
  ------------------------------------------------------------ */
  const panoFuzzy = panoNames.find((p) => {
    const t = p.toLowerCase();

    return (
      t.includes(cleaned) ||
      cleaned.includes(t) ||
      levenshteinDistance(t, cleaned) <= 2
    );
  });

  if (panoFuzzy) {
    return { intent: "pano", target: panoFuzzy };
  }

  /* ------------------------------------------------------------
     7) EXACT PROJECT MATCH
  ------------------------------------------------------------ */
  const projExact = projectNames.find(
    (p) => p.toLowerCase() === cleaned
  );

  if (projExact) {
    return { intent: "project", target: projExact };
  }

  /* ------------------------------------------------------------
     8) PROJECT — STRONG FUZZY MATCH
  ------------------------------------------------------------ */
  const projFuzzy = projectNames.find((p) => {
    const t = p.toLowerCase();

    return (
      t.includes(cleaned) ||
      cleaned.includes(t) ||
      levenshteinDistance(t, cleaned) <= 2
    );
  });

  if (projFuzzy) {
    return { intent: "project", target: projFuzzy };
  }

  /* ------------------------------------------------------------
     9) DEFAULT → SCHOOL CHATBOT
  ------------------------------------------------------------ */
  return { intent: "school" };
}

/* ------------------------------------------------------------
   LEVENSHTEIN DISTANCE
------------------------------------------------------------ */
function levenshteinDistance(a, b) {
  if (!a || !b) return 99;

  const m = a.length;
  const n = b.length;

  const dp = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0)
  );

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
