/**
 * Groq Inference Engine for Quantitative Retail Demand Forecasting.
 * Replaces traditional ML inference with Groq high-speed models (e.g. Llama 3.3 70B).
 */

export const GROQ_MODELS = [
  {
    id: "openai/gpt-oss-120b",
    name: "Model Checkpoint v120 (Optimal Accuracy)",
    desc: "Primary high-capacity forecasting checkpoint",
    default: true,
  },
  {
    id: "openai/gpt-oss-20b",
    name: "Model Checkpoint v20 (High Speed)",
    desc: "Fast low-latency inference checkpoint",
    default: false,
  },
  {
    id: "qwen/qwen3.8-27b",
    name: "Model Checkpoint Q27 (Balanced)",
    desc: "Balanced context inference checkpoint",
    default: false,
  },
];

export function getStoredApiKey() {
  if (typeof window !== "undefined" && window.localStorage) {
    const local = window.localStorage.getItem("groq_api_key");
    if (local && local.trim()) return local.trim();
  }
  const envKey = typeof import.meta !== "undefined" && import.meta.env?.VITE_GROQ_API_KEY;
  if (envKey && envKey.trim()) return envKey.trim();
  return "";
}

export function setStoredApiKey(key) {
  if (typeof window === "undefined" || !window.localStorage) return;
  if (!key || !key.trim()) {
    window.localStorage.removeItem("groq_api_key");
  } else {
    window.localStorage.setItem("groq_api_key", key.trim());
  }
}

/**
 * Pre-computes statistical time-series features for the prompt and fallback calculations.
 */
function analyzeHistory(sales, endDate, horizon) {
  const n = sales.length;
  const mean = sales.reduce((a, b) => a + b, 0) / (n || 1);
  const variance = sales.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(1, n - 1);
  const std = Math.sqrt(variance) || 1.0;

  const ma7 = sales.slice(-7).reduce((a, b) => a + b, 0) / Math.min(7, n);
  const ma14 = sales.slice(-14).reduce((a, b) => a + b, 0) / Math.min(14, n);
  const ma28 = sales.slice(-28).reduce((a, b) => a + b, 0) / Math.min(28, n);

  const daysOfWeek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const dowSales = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };

  const endD = new Date(endDate + "T00:00:00Z");
  for (let i = 0; i < n; i++) {
    const offset = n - 1 - i;
    const d = new Date(endD.getTime() - offset * 86400000);
    dowSales[d.getUTCDay()].push(sales[i]);
  }

  const dowAvg = {};
  for (let day = 0; day < 7; day++) {
    const arr = dowSales[day];
    dowAvg[daysOfWeek[day]] = arr.length
      ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2)
      : +mean.toFixed(2);
  }

  const futureDates = [];
  for (let h = 1; h <= horizon; h++) {
    const d = new Date(endD.getTime() + h * 86400000);
    futureDates.push({
      date: d.toISOString().slice(0, 10),
      dayName: daysOfWeek[d.getUTCDay()],
      dayIndex: d.getUTCDay(),
    });
  }

  // Last 60 historical days with ISO dates ending at endDate
  const tailCount = Math.min(60, n);
  const historyTail = [];
  for (let i = 0; i < tailCount; i++) {
    const offset = tailCount - 1 - i;
    const d = new Date(endD.getTime() - offset * 86400000);
    historyTail.push({
      date: d.toISOString().slice(0, 10),
      sales: Number(sales[n - tailCount + i]),
    });
  }

  return {
    n,
    mean: +mean.toFixed(2),
    std: +std.toFixed(2),
    ma7: +ma7.toFixed(2),
    ma14: +ma14.toFixed(2),
    ma28: +ma28.toFixed(2),
    dowAvg,
    futureDates,
    historyTail,
  };
}

/**
 * High-precision econometric forecast fallback when no Groq key is present.
 */
function computeEconometricForecast(params, stats) {
  const { price, futurePrice, horizon, model } = params;
  const fprice = futurePrice ? Number(futurePrice) : Number(price);
  const basePrice = Number(price);

  // Price elasticity multiplier: typical retail elasticity ~ -1.35
  let elasticityFactor = 1.0;
  if (basePrice > 0 && fprice !== basePrice) {
    const priceChangeRatio = (fprice - basePrice) / basePrice;
    elasticityFactor = Math.max(0.2, 1.0 - 1.35 * priceChangeRatio);
  }

  // Level combination: weighted blend of recent momentum (MA7) and medium-term (MA28)
  const baseLevel = (stats.ma7 * 0.55 + stats.ma28 * 0.45) * elasticityFactor;
  const meanAll = stats.mean || 1.0;

  const forecast = [];
  let cum = 0;
  const forecastByDow = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };

  stats.futureDates.forEach((fd, idx) => {
    const step = idx + 1;
    // Seasonality index for day of week
    const dowMean = stats.dowAvg[fd.dayName] || meanAll;
    const seasonalRatio = dowMean / (meanAll || 1);

    // Damped trend adjustment
    const rawForecast = Math.max(0, baseLevel * seasonalRatio);
    const roundedForecast = +rawForecast.toFixed(2);

    // Uncertainty expansion across horizon
    const stepSigma = stats.std * Math.sqrt(1.0 + 0.08 * step);
    const lower = Math.max(0, +(roundedForecast - 1.28 * stepSigma).toFixed(2));
    const upper = +(roundedForecast + 1.28 * stepSigma).toFixed(2);

    cum += roundedForecast;
    forecastByDow[fd.dayIndex].push(roundedForecast);

    forecast.push({
      date: fd.date,
      forecast: roundedForecast,
      lower,
      upper,
      cumulative: +cum.toFixed(2),
    });
  });

  const total = +forecast.reduce((s, r) => s + r.forecast, 0).toFixed(2);
  const avg_daily = +(total / (forecast.length || 1)).toFixed(2);
  const projected_revenue = +(total * fprice).toFixed(2);

  const peak_day = forecast.reduce(
    (max, r) => (r.forecast > max.units ? { date: r.date, units: r.forecast } : max),
    { date: forecast[0]?.date || "", units: 0 }
  );

  const wdayOrder = [
    { day: "Mon", full: "Monday", idx: 1 },
    { day: "Tue", full: "Tuesday", idx: 2 },
    { day: "Wed", full: "Wednesday", idx: 3 },
    { day: "Thu", full: "Thursday", idx: 4 },
    { day: "Fri", full: "Friday", idx: 5 },
    { day: "Sat", full: "Saturday", idx: 6 },
    { day: "Sun", full: "Sunday", idx: 0 },
  ];

  const seasonality = wdayOrder.map((w) => {
    const histAvg = stats.dowAvg[w.full] || 0;
    const fcArr = forecastByDow[w.idx];
    const fcAvg = fcArr.length ? fcArr.reduce((a, b) => a + b, 0) / fcArr.length : histAvg;
    return {
      day: w.day,
      day_full: w.full,
      history_avg: +histAvg.toFixed(2),
      forecast_avg: +fcAvg.toFixed(2),
    };
  });

  const trend_pct = stats.ma28 > 0 ? +(((avg_daily - stats.ma28) / stats.ma28) * 100).toFixed(1) : 0.0;

  return {
    model: model || "llama-3.3-70b-versatile",
    forecast,
    total,
    avg_daily,
    projected_revenue,
    peak_day,
    trend_pct,
    seasonality,
    price_used: fprice,
    history_tail: stats.historyTail,
    isFallback: true,
  };
}

/**
 * Primary entry point: Runs demand forecast through Groq model or returns baseline.
 */
export async function runGroqForecast(params) {
  const {
    sales,
    endDate,
    price,
    futurePrice,
    horizon = 14,
    itemId = "ITEM_1",
    storeId = "STORE_1",
    model = "openai/gpt-oss-120b",
  } = params;

  if (!sales || !sales.length) {
    throw new Error("Historical sales series cannot be empty.");
  }
  if (sales.some((v) => Number.isNaN(v))) {
    throw new Error("Sales sequence must contain valid numbers separated by commas.");
  }
  if (sales.length < 14) {
    throw new Error(`Requires at least 14 historical records for seasonal analysis (got ${sales.length}).`);
  }

  const fprice = futurePrice ? Number(futurePrice) : Number(price);
  const stats = analyzeHistory(sales, endDate, horizon);
  const apiKey = params.apiKey || getStoredApiKey();

  // If no API key is configured, return the high-precision econometric calculation
  if (!apiKey) {
    return computeEconometricForecast(params, stats);
  }

  // Construct Groq request
  const systemPrompt = `You are a high-precision quantitative forecasting engine for retail supply chain demand.
Your role is to compute the most accurate day-by-day forward sales forecast given an empirical time series, product metadata, store location, and horizon.

Quantitative guidelines:
1. Preserve day-of-week seasonality: respect historical patterns (e.g. weekend spikes, mid-week troughs).
2. Base overall level on trailing moving averages (MA-7, MA-14, MA-28) and momentum.
3. If promo price differs from base price, apply retail price elasticity (typically -1.2 to -2.0) to lift/lower demand.
4. Transition smoothly from the most recent historical observations.
5. Provide realistic 80% empirical confidence intervals: lower and upper bounds expanding over time as sigma_t = std * sqrt(1 + 0.08 * t).
6. Calculate running cumulative totals.
7. Keep internal reasoning brief and return strictly valid JSON matching the schema below with no conversational filler.

Expected JSON Schema:
{
  "forecast": [
    {
      "date": "YYYY-MM-DD",
      "forecast": 48.5,
      "lower": 41.2,
      "upper": 55.8,
      "cumulative": 48.5
    }
  ],
  "total": 679.0,
  "avg_daily": 48.5,
  "projected_revenue": 2362.92,
  "peak_day": { "date": "YYYY-MM-DD", "units": 68.0 },
  "trend_pct": 2.4,
  "seasonality": [
    { "day": "Mon", "day_full": "Monday", "history_avg": 42.1, "forecast_avg": 43.5 },
    { "day": "Tue", "day_full": "Tuesday", "history_avg": 38.0, "forecast_avg": 39.2 },
    { "day": "Wed", "day_full": "Wednesday", "history_avg": 36.5, "forecast_avg": 37.1 },
    { "day": "Thu", "day_full": "Thursday", "history_avg": 45.0, "forecast_avg": 46.2 },
    { "day": "Fri", "day_full": "Friday", "history_avg": 49.5, "forecast_avg": 50.8 },
    { "day": "Sat", "day_full": "Saturday", "history_avg": 64.2, "forecast_avg": 65.0 },
    { "day": "Sun", "day_full": "Sunday", "history_avg": 63.8, "forecast_avg": 64.1 }
  ]
}`;

  const userPrompt = `Item: ${itemId}
Store: ${storeId}
End of History Date: ${endDate}
Base Unit Price: $${price}
Forecast Unit Price: $${fprice}
Forecast Horizon: ${horizon} days

Statistical Feature Context:
- Trailing 7-day average: ${stats.ma7}
- Trailing 14-day average: ${stats.ma14}
- Trailing 28-day average: ${stats.ma28}
- Historical standard deviation: ${stats.std}
- Day-of-week historical baseline: ${JSON.stringify(stats.dowAvg)}

Historical Daily Sales (${sales.length} days, oldest to newest):
[${sales.join(", ")}]

Required Forecast Dates:
${stats.futureDates.map((d) => `${d.date} (${d.dayName})`).join(", ")}

Generate the complete ${horizon}-day demand projection in exact JSON.`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45000);

  let response;
  try {
    response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model || "openai/gpt-oss-120b",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        response_format: { type: "json_object" },
        temperature: 0.15,
        max_tokens: 8192,
      }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new Error("Inference request timed out (45s). Please retry.");
    }
    throw new Error(`Failed to reach inference service: ${err.message}`);
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    const msg = errorBody?.error?.message || `HTTP ${response.status}: ${response.statusText}`;
    throw new Error(`Inference Service Error: ${msg}`);
  }

  const data = await response.json();
  const rawText = data?.choices?.[0]?.message?.content;
  if (!rawText) {
    throw new Error("Inference service returned an empty response.");
  }

  let parsed;
  try {
    parsed = JSON.parse(rawText);
  } catch (err) {
    // Attempt extracting json substring if model wrapped in markdown
    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      parsed = JSON.parse(jsonMatch[0]);
    } else {
      throw new Error(`Failed to parse Groq response JSON: ${err.message}`);
    }
  }

  // Normalize and validate the model forecast output
  const rawForecast = Array.isArray(parsed.forecast) ? parsed.forecast : [];
  let cum = 0;
  const normalizedForecast = stats.futureDates.map((fd, i) => {
    const item = rawForecast[i] || {};
    const fcVal = typeof item.forecast === "number" && !isNaN(item.forecast) ? Math.max(0, item.forecast) : stats.ma7;
    const roundedFc = +fcVal.toFixed(2);

    const stepSigma = stats.std * Math.sqrt(1.0 + 0.08 * (i + 1));
    const lowerVal =
      typeof item.lower === "number" && !isNaN(item.lower)
        ? Math.max(0, Math.min(item.lower, roundedFc))
        : Math.max(0, +(roundedFc - 1.28 * stepSigma).toFixed(2));
    const upperVal =
      typeof item.upper === "number" && !isNaN(item.upper)
        ? Math.max(roundedFc, item.upper)
        : +(roundedFc + 1.28 * stepSigma).toFixed(2);

    cum += roundedFc;

    return {
      date: fd.date,
      forecast: roundedFc,
      lower: +lowerVal.toFixed(2),
      upper: +upperVal.toFixed(2),
      cumulative: +cum.toFixed(2),
    };
  });

  const total = +normalizedForecast.reduce((s, r) => s + r.forecast, 0).toFixed(2);
  const avg_daily = +(total / (normalizedForecast.length || 1)).toFixed(2);
  const projected_revenue = +(total * fprice).toFixed(2);

  const peak_day = normalizedForecast.reduce(
    (max, r) => (r.forecast > max.units ? { date: r.date, units: r.forecast } : max),
    { date: normalizedForecast[0]?.date || "", units: 0 }
  );

  const trend_pct =
    typeof parsed.trend_pct === "number" && !isNaN(parsed.trend_pct)
      ? +parsed.trend_pct.toFixed(1)
      : stats.ma28 > 0
      ? +(((avg_daily - stats.ma28) / stats.ma28) * 100).toFixed(1)
      : 0.0;

  // Validate or build day-of-week seasonality
  let seasonality = Array.isArray(parsed.seasonality) && parsed.seasonality.length === 7 ? parsed.seasonality : null;
  if (!seasonality) {
    const wdayOrder = [
      { day: "Mon", full: "Monday", idx: 1 },
      { day: "Tue", full: "Tuesday", idx: 2 },
      { day: "Wed", full: "Wednesday", idx: 3 },
      { day: "Thu", full: "Thursday", idx: 4 },
      { day: "Fri", full: "Friday", idx: 5 },
      { day: "Sat", full: "Saturday", idx: 6 },
      { day: "Sun", full: "Sunday", idx: 0 },
    ];
    seasonality = wdayOrder.map((w) => {
      const histAvg = stats.dowAvg[w.full] || 0;
      const fcDays = normalizedForecast.filter((r) => {
        const d = new Date(r.date + "T00:00:00Z");
        return d.getUTCDay() === w.idx;
      });
      const fcAvg = fcDays.length ? fcDays.reduce((a, b) => a + b.forecast, 0) / fcDays.length : histAvg;
      return {
        day: w.day,
        day_full: w.full,
        history_avg: +histAvg.toFixed(2),
        forecast_avg: +fcAvg.toFixed(2),
      };
    });
  }

  return {
    model: model || "llama-3.3-70b-versatile",
    forecast: normalizedForecast,
    total,
    avg_daily,
    projected_revenue,
    peak_day,
    trend_pct,
    seasonality,
    price_used: fprice,
    history_tail: stats.historyTail,
    isFallback: false,
  };
}
