import type { PredictionInput, PredictionResult } from "../types/prediction";
const base = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
export async function getLocations(): Promise<string[]> { const r = await fetch(`${base}/locations`); if (!r.ok) throw new Error("Could not load locations"); return r.json(); }
export async function requestPrediction(data: PredictionInput): Promise<PredictionResult> { const r = await fetch(`${base}/predict`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}); if (!r.ok) throw new Error("The estimate could not be calculated. Is the API running?"); return r.json(); }
