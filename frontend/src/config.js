// API Base configuration for local dev and cloud deployment (Vercel -> Render)
// Set VITE_API_BASE in Vercel project environment variables if not using reverse proxy rewrites.
export const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
