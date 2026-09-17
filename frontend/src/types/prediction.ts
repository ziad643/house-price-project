export type PredictionInput = { location: string; carpet_area_sqft: number; floor_num: number; bathroom: number; balcony: number; car_parking: number; furnishing: string; transaction: string; ownership: string; facing: string };
export type PredictionResult = { predicted_price: number; currency: string };
