import express from "express";
import cors from "cors";
import { pipeline, env } from "@xenova/transformers";
import natural from "natural";

/* ───────────────────────────────────────────── */
/* TRANSFORMERS CACHE                           */
/* ───────────────────────────────────────────── */

env.cacheDir = "/tmp/.cache/transformers";

/* ───────────────────────────────────────────── */
/* APP SETUP                                    */
/* ───────────────────────────────────────────── */

const app = express();

const PORT = process.env.PORT || 5000;

app.use(cors({ origin: "*" }));

app.use(
  express.json({
    limit: "20kb",
  })
);

/* ───────────────────────────────────────────── */
/* RATE LIMITER                                 */
/* ───────────────────────────────────────────── */

const rateLimitMap =
  new Map<
    string,
    {
      count: number;
      resetAt: number;
    }
  >();

const RATE_LIMIT = 30;

const RATE_WINDOW_MS = 60_000;

function isRateLimited(
  ip: string
): boolean {

  const now = Date.now();

  const entry =
    rateLimitMap.get(ip);

  if (
    !entry ||
    now > entry.resetAt
  ) {

    rateLimitMap.set(ip, {
      count: 1,
      resetAt:
        now + RATE_WINDOW_MS,
    });

    return false;

  }

  if (
    entry.count >= RATE_LIMIT
  ) {

    return true;

  }

  entry.count++;

  return false;

}

setInterval(() => {

  const now = Date.now();

  for (const [key, val]
    of rateLimitMap) {

    if (
      now > val.resetAt
    ) {

      rateLimitMap.delete(key);

    }

  }

}, 5 * 60_000);

/* ───────────────────────────────────────────── */
/* CONCURRENCY QUEUE                            */
/* ───────────────────────────────────────────── */

let activeRequests = 0;

const MAX_CONCURRENT = 20;

const waitQueue:
  Array<() => void> = [];

function acquireSlot():
  Promise<void> {

  return new Promise(
    (resolve) => {

      if (
        activeRequests <
        MAX_CONCURRENT
      ) {

        activeRequests++;

        resolve();

      }

      else {

        waitQueue.push(() => {

          activeRequests++;

          resolve();

        });

      }

    }
  );

}

function releaseSlot() {

  activeRequests--;

  const next =
    waitQueue.shift();

  if (next) {

    next();

  }

}

/* ───────────────────────────────────────────── */
/* MODEL LOADER                                 */
/* ───────────────────────────────────────────── */

let embedder: any = null;

let modelReady = false;

async function getEmbedder() {

  if (!embedder) {

    console.log(
      "Loading model..."
    );

    embedder =
      await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
      );

    modelReady = true;

    console.log(
      "Model ready."
    );

  }

  return embedder;

}

getEmbedder().catch(
  console.error
);

/* ───────────────────────────────────────────── */
/* STOP WORDS                                   */
/* ───────────────────────────────────────────── */

const STOP_WORDS = new Set([
  "a","an","the","is","it",
  "in","on","of","to","and",
  "or","but","for","with",
  "this","that","are","was",
  "were","be","been","being",
  "by","at","from","as",
  "into","about","which",
  "who","what","how","when",
  "where","can","could",
  "will","would","should",
  "may","might","do","does",
  "did","have","has","had",
  "not","also","its","their",
  "our","we","they","he",
  "she","you","i","me",
  "my","your","his","her",
]);

/* ───────────────────────────────────────────── */
/* NORMALIZE                                    */
/* ───────────────────────────────────────────── */

function normalize(
  text: string
): string {

  return text
    .toLowerCase()
    .replace(
      /[^a-z0-9\s]/g,
      " "
    )
    .replace(/\s+/g, " ")
    .trim();

}

/* ───────────────────────────────────────────── */
/* TOKENIZE                                     */
/* ───────────────────────────────────────────── */

function tokenize(
  text: string
) {

  return normalize(text)
    .split(" ")
    .filter(
      (word) =>
        word.length > 2 &&
        !STOP_WORDS.has(word)
    )
    .map((word) => ({
      raw: word,
      stem:
        natural.PorterStemmer
          .stem(word),
    }));

}

/* ───────────────────────────────────────────── */
/* LEVENSHTEIN                                  */
/* ───────────────────────────────────────────── */

function fuzzyMatch(
  word1: string,
  word2: string
): boolean {

  if (
    word1 === word2
  ) {

    return true;

  }

  const distance =
    natural.LevenshteinDistance(
      word1,
      word2
    );

  return distance <= 1;

}

/* ───────────────────────────────────────────── */
/* COSINE SIMILARITY                            */
/* ───────────────────────────────────────────── */

function cosine(
  a: number[],
  b: number[]
): number {

  let dot = 0;

  let magA = 0;

  let magB = 0;

  for (
    let i = 0;
    i < a.length;
    i++
  ) {

    dot +=
      a[i] * b[i];

    magA +=
      a[i] * a[i];

    magB +=
      b[i] * b[i];

  }

  if (
    magA === 0 ||
    magB === 0
  ) {

    return 0;

  }

  return (
    dot /
    (
      Math.sqrt(magA) *
      Math.sqrt(magB)
    )
  );

}

/* ───────────────────────────────────────────── */
/* QUESTION TYPE DETECTION                      */
/* ───────────────────────────────────────────── */

function detectQuestionType(
  question: string
) {

  const q =
    normalize(question);

  if (
    q.includes("example")
  ) {

    return "example";

  }

  if (
    q.includes("define") ||
    q.includes("what is")
  ) {

    return "definition";

  }

  if (
    q.includes("difference")
  ) {

    return "difference";

  }

  if (
    q.includes("advantages") ||
    q.includes("disadvantages")
  ) {

    return "listing";

  }

  return "general";

}

/* ───────────────────────────────────────────── */
/* CONCEPT COVERAGE                             */
/* ───────────────────────────────────────────── */

function conceptCoverage(
  correctAnswer: string,
  studentAnswer: string
) {

  const correctTokens =
    tokenize(correctAnswer);

  const studentTokens =
    tokenize(studentAnswer);

  if (
    correctTokens.length === 0
  ) {

    return {
      score: 0,
      matched: 0,
      total: 0,
    };

  }

  let matched = 0;

  for (
    const ct of correctTokens
  ) {

    const found =
      studentTokens.some(
        (st) =>

          fuzzyMatch(
            st.raw,
            ct.raw
          )

          ||

          st.stem ===
          ct.stem
      );

    if (found) {

      matched++;

    }

  }

  return {

    score:
      (
        matched /
        correctTokens.length
      ) * 100,

    matched,

    total:
      correctTokens.length,

  };

}

/* ───────────────────────────────────────────── */
/* CONTRADICTION DETECTION                      */
/* ───────────────────────────────────────────── */

const CONTRADICTIONS:
Record<string, string[]> = {

  zinc: [
    "copper",
    "silver",
    "gold",
  ],

  oxidation: [
    "reduction",
  ],

  reduction: [
    "oxidation",
  ],

  acid: [
    "base",
    "alkaline",
  ],

  base: [
    "acid",
  ],

};

function contradictionPenalty(
  correctAnswer: string,
  studentAnswer: string
): number {

  const correctTokens =
    tokenize(correctAnswer)
      .map((t) => t.raw);

  const studentTokens =
    tokenize(studentAnswer)
      .map((t) => t.raw);

  let penalty = 0;

  for (
    const correctWord
    of correctTokens
  ) {

    const oppositeWords =
      CONTRADICTIONS[
        correctWord
      ];

    if (
      oppositeWords
    ) {

      for (
        const opposite
        of oppositeWords
      ) {

        if (
          studentTokens.includes(
            opposite
          )
        ) {

          penalty += 25;

        }

      }

    }

  }

  return penalty;

}

/* ───────────────────────────────────────────── */
/* COPY DETECTION                               */
/* ───────────────────────────────────────────── */

function copyPenalty(
  question: string,
  student: string
): number {

  const qTokens =
    tokenize(question);

  const sTokens =
    tokenize(student);

  if (
    qTokens.length === 0
  ) {

    return 0;

  }

  let overlap = 0;

  for (
    const qt of qTokens
  ) {

    const found =
      sTokens.some(
        (st) =>
          st.stem ===
          qt.stem
      );

    if (found) {

      overlap++;

    }

  }

  const ratio =
    overlap /
    qTokens.length;

  return ratio;

}

/* ───────────────────────────────────────────── */
/* LENGTH MULTIPLIER                            */
/* ───────────────────────────────────────────── */

function lengthMultiplier(
  correct: string,
  student: string,
  type: string
): number {

  /*
    Do NOT punish concise
    definition/example answers
  */

  if (
    type === "definition" ||
    type === "example"
  ) {

    return 1;

  }

  const cLen =
    tokenize(correct)
      .length;

  const sLen =
    tokenize(student)
      .length;

  if (cLen === 0) {

    return 1;

  }

  const ratio =
    sLen / cLen;

  if (
    ratio >= 0.5
  ) {

    return 1;

  }

  if (
    ratio >= 0.3
  ) {

    return 0.85;

  }

  if (
    ratio >= 0.15
  ) {

    return 0.7;

  }

  return 0.5;

}

/* ───────────────────────────────────────────── */
/* EMBEDDING                                    */
/* ───────────────────────────────────────────── */

async function embed(
  text: string
): Promise<number[]> {

  const model =
    await getEmbedder();

  const output =
    await model(text, {
      pooling: "mean",
      normalize: true,
    });

  return Array.from(
    output.data
  ) as number[];

}

/* ───────────────────────────────────────────── */
/* FINAL SCORE                                  */
/* ───────────────────────────────────────────── */

function computeFinalScore({

  semanticPct,
  coveragePct,
  lenMultiplier,
  copyPenaltyValue,
  contradictionPenaltyValue,
  questionType,

}: {

  semanticPct: number;

  coveragePct: number;

  lenMultiplier: number;

  copyPenaltyValue: number;

  contradictionPenaltyValue: number;

  questionType: string;

}) {

  let semantic =
    semanticPct;

  /*
    Short answers should
    depend more on concepts
  */

  if (
    questionType ===
      "definition" ||
    questionType ===
      "example"
  ) {

    semantic =
      Math.min(
        semantic,
        75
      );

  }

  /*
    Adaptive semantic cap
  */

  if (
    coveragePct < 20
  ) {

    semantic =
      Math.min(
        semantic,
        45
      );

  }

  else if (
    coveragePct < 35
  ) {

    semantic =
      Math.min(
        semantic,
        60
      );

  }

  else if (
    coveragePct < 50
  ) {

    semantic =
      Math.min(
        semantic,
        75
      );

  }

  let score =
    (
      semantic * 0.5
    ) +
    (
      coveragePct * 0.5
    );

  /*
    Length multiplier
  */

  score =
    score *
    lenMultiplier;

  /*
    Copy penalty
  */

  if (
    copyPenaltyValue > 0.7
  ) {

    score -= 25;

  }

  /*
    Contradiction penalty
  */

  score -=
    contradictionPenaltyValue;

  /*
    Bounds
  */

  if (score < 0) {

    score = 0;

  }

  if (score > 100) {

    score = 100;

  }

  return Math.round(
    score
  );

}

/* ───────────────────────────────────────────── */
/* ROUTES                                        */
/* ───────────────────────────────────────────── */

app.get("/", (
  _req,
  res
) => {

  res.json({

    status: "ok",

    modelReady,

    activeRequests,

    queued:
      waitQueue.length,

  });

});

app.post(
  "/check-answer",

  async (
    req,
    res
  ) => {

    const ip =
      (
        (
          req.headers[
            "x-forwarded-for"
          ] as string
        )

        ||

        req.socket
          .remoteAddress

        ||

        "unknown"
      )
        .split(",")[0]
        .trim();

    if (
      isRateLimited(ip)
    ) {

      return res
        .status(429)
        .json({

          success: false,

          error:
            "Too many requests",

        });

    }

    const {

      question,

      correctAnswer,

      studentAnswer,

      totalMarks = 10,

    } = req.body;

    /*
      VALIDATION
    */

    if (
      !question ||
      !correctAnswer ||
      studentAnswer ===
      undefined
    ) {

      return res
        .status(400)
        .json({

          success: false,

          error:
            "Missing fields",

        });

    }

    const student =
      studentAnswer.trim();

    /*
      EMPTY ANSWER
    */

    if (!student) {

      return res.json({

        success: true,

        percentage: 0,

        marksAwarded: 0,

        semanticScore: 0,

        coverageScore: 0,

      });

    }

    /*
      EXACT MATCH
    */

    if (
      normalize(student)
      ===
      normalize(
        correctAnswer
      )
    ) {

      return res.json({

        success: true,

        percentage: 100,

        marksAwarded:
          totalMarks,

        semanticScore: 100,

        coverageScore: 100,

      });

    }

    /*
      FULL QUESTION COPY
    */

    if (
      normalize(student)
      ===
      normalize(
        question
      )
    ) {

      return res.json({

        success: true,

        percentage: 0,

        marksAwarded: 0,

        semanticScore: 0,

        coverageScore: 0,

        reason:
          "Copied question",

      });

    }

    /*
      TOO SHORT
    */

    const contentWords =
      tokenize(student);

    if (
      contentWords.length < 2
    ) {

      return res.json({

        success: true,

        percentage: 0,

        marksAwarded: 0,

        semanticScore: 0,

        coverageScore: 0,

        reason:
          "Too short",

      });

    }

    await acquireSlot();

    try {

      const [
        emb1,
        emb2,
      ] =
        await Promise.all([
          embed(
            correctAnswer
          ),
          embed(student),
        ]);

      const semanticPct =
        cosine(
          emb1,
          emb2
        ) * 100;

      const {

        score:
          coveragePct,

      } =
        conceptCoverage(
          correctAnswer,
          student
        );

      const questionType =
        detectQuestionType(
          question
        );

      const lenMulti =
        lengthMultiplier(
          correctAnswer,
          student,
          questionType
        );

      const copyPen =
        copyPenalty(
          question,
          student
        );

      const contradictionPen =
        contradictionPenalty(
          correctAnswer,
          student
        );

      const percentage =
        computeFinalScore({

          semanticPct,

          coveragePct,

          lenMultiplier:
            lenMulti,

          copyPenaltyValue:
            copyPen,

          contradictionPenaltyValue:
            contradictionPen,

          questionType,

        });

      const marksAwarded =
        Math.round(
          (
            percentage / 100
          ) *
          totalMarks *
          10
        ) / 10;

      return res.json({

        success: true,

        percentage,

        marksAwarded,

        semanticScore:
          Math.round(
            semanticPct
          ),

        coverageScore:
          Math.round(
            coveragePct
          ),

      });

    }

    catch (err) {

      console.error(err);

      return res
        .status(500)
        .json({

          success: false,

          error:
            "Evaluation failed",

        });

    }

    finally {

      releaseSlot();

    }

  }
);

/* ───────────────────────────────────────────── */
/* START SERVER                                 */
/* ───────────────────────────────────────────── */

app.listen(
  PORT,

  () => {

    console.log(
      `Server running on ${PORT}`
    );

  }
);