// Glasshouse assumptions and rough crop targets. Values are textbook approximations:
// tune them to your varieties and glass.
export const GDD_BASE_C = 10; // default growth base when a crop has none
export const GLASS_TRANSMISSION = 0.7; // share of outdoor light that reaches the plants
const PAR_FRACTION = 0.45; // photosynthetic share of shortwave radiation
const UMOL_PER_J_PAR = 4.57; // µmol photons per joule of sunlight PAR

/** mol/m² of usable light per day for 1 W/m² of mean outdoor shortwave radiation. */
export const DLI_PER_WM2_DAY = (PAR_FRACTION * UMOL_PER_J_PAR * GLASS_TRANSMISSION * 86400) / 1e6;

/** Growing facts from general horticultural references; correct them for your varieties. */
export type CropFacts = {
  /** Soil nutrient role: heavy/medium/light feeders use up nitrogen, fixers add it. */
  feeder: "heavy" | "medium" | "light" | "fixer";
  soil: string;
  companions: string[];
  avoid: string[];
  attracts: string[];
  pests: string[];
  sun: string;
  water: string;
  spacingCm: string;
  ph: string;
};

export type Crop = {
  name: string;
  baseC: number; // growth stops below this temperature
  gddToHarvest: number; // °C·d from planting to harvest
  dliMin: number; // mol/m²/day below which growth is poor
  dliIdeal: number; // mol/m²/day for good growth
  facts: CropFacts;
};

export const CROPS: Crop[] = [
  {
    name: "Lettuce", baseC: 4, gddToHarvest: 650, dliMin: 10, dliIdeal: 14,
    facts: {
      feeder: "medium", soil: "Moderate nitrogen; shallow roots, so keep the top layer fertile and moist.",
      companions: ["Carrot", "Radish", "Strawberry", "Cucumber", "Onion"], avoid: ["Celery", "Parsley"],
      attracts: ["Hoverflies (if flowering herbs nearby)"], pests: ["Aphids", "Slugs", "Thrips"],
      sun: "Part to full sun; bolts in heat", water: "Even, frequent", spacingCm: "20–30", ph: "6.0–7.0",
    },
  },
  {
    name: "Spinach", baseC: 2, gddToHarvest: 500, dliMin: 8, dliIdeal: 12,
    facts: {
      feeder: "heavy", soil: "Heavy nitrogen user: leafy growth needs a nitrogen-rich bed.",
      companions: ["Strawberry", "Pea", "Bean", "Radish"], avoid: ["Fennel"],
      attracts: ["Leaf miner flies"], pests: ["Leaf miners", "Aphids", "Downy mildew"],
      sun: "Part to full sun; cool season", water: "Even", spacingCm: "10–15", ph: "6.5–7.5",
    },
  },
  {
    name: "Radish", baseC: 5, gddToHarvest: 450, dliMin: 10, dliIdeal: 14,
    facts: {
      feeder: "light", soil: "Light feeder; excess nitrogen gives leaves instead of roots.",
      companions: ["Lettuce", "Carrot", "Cucumber", "Pea", "Spinach"], avoid: ["Hyssop"],
      attracts: ["Flea beetles (works as a trap crop)"], pests: ["Flea beetles", "Root maggot"],
      sun: "Full to part sun", water: "Even; dry spells make roots woody", spacingCm: "3–5", ph: "6.0–7.0",
    },
  },
  {
    name: "Basil", baseC: 10, gddToHarvest: 700, dliMin: 12, dliIdeal: 17,
    facts: {
      feeder: "light", soil: "Light feeder; rich in oils when not over-fertilised.",
      companions: ["Tomato", "Pepper"], avoid: ["Sage", "Rue"],
      attracts: ["Bees", "Hoverflies"], pests: ["Aphids", "Slugs", "Downy mildew"],
      sun: "Full sun, warm", water: "Moderate; keep leaves dry", spacingCm: "20–30", ph: "6.0–7.0",
    },
  },
  {
    name: "Tomato", baseC: 10, gddToHarvest: 1300, dliMin: 15, dliIdeal: 25,
    facts: {
      feeder: "heavy", soil: "Heavy feeder: nitrogen early, then potassium and phosphorus once fruiting.",
      companions: ["Basil", "Carrot", "Onion", "Parsley", "Marigold"], avoid: ["Fennel", "Brassicas", "Potato", "Corn"],
      attracts: ["Bees (buzz pollination)", "Hornworms"], pests: ["Whitefly", "Aphids", "Blight", "Botrytis"],
      sun: "Full sun", water: "Deep and regular; uneven watering splits fruit", spacingCm: "45–60", ph: "6.0–6.8",
    },
  },
  {
    name: "Pepper", baseC: 10, gddToHarvest: 1500, dliMin: 15, dliIdeal: 22,
    facts: {
      feeder: "medium", soil: "Medium-heavy feeder; too much nitrogen means leaves over fruit.",
      companions: ["Basil", "Onion", "Carrot"], avoid: ["Fennel", "Brassicas"],
      attracts: ["Bees", "Aphids"], pests: ["Aphids", "Thrips", "Whitefly"],
      sun: "Full sun, warm", water: "Regular", spacingCm: "40–50", ph: "6.0–6.8",
    },
  },
  {
    name: "Cucumber", baseC: 12, gddToHarvest: 900, dliMin: 15, dliIdeal: 22,
    facts: {
      feeder: "heavy", soil: "Heavy feeder; wants rich, well-composted soil.",
      companions: ["Bean", "Pea", "Radish", "Lettuce", "Dill"], avoid: ["Sage", "Potato"],
      attracts: ["Bees", "Spider mites", "Whitefly"], pests: ["Spider mites", "Aphids", "Powdery mildew"],
      sun: "Full sun, warm", water: "Heavy and regular", spacingCm: "30–45 (trellised)", ph: "6.0–7.0",
    },
  },
  {
    name: "Bean", baseC: 10, gddToHarvest: 800, dliMin: 12, dliIdeal: 18,
    facts: {
      feeder: "fixer", soil: "Nitrogen fixer: root bacteria add nitrogen to the soil. Don't over-feed nitrogen; it helps the next crop.",
      companions: ["Carrot", "Cucumber", "Lettuce", "Radish"], avoid: ["Onion", "Garlic", "Fennel"],
      attracts: ["Bees"], pests: ["Aphids", "Spider mites"],
      sun: "Full sun", water: "Moderate", spacingCm: "8–10 (bush)", ph: "6.0–7.0",
    },
  },
  {
    name: "Pea", baseC: 4, gddToHarvest: 800, dliMin: 10, dliIdeal: 15,
    facts: {
      feeder: "fixer", soil: "Nitrogen fixer: leaves nitrogen behind for the next crop.",
      companions: ["Carrot", "Radish", "Lettuce", "Cucumber"], avoid: ["Onion", "Garlic"],
      attracts: ["Bees"], pests: ["Aphids", "Powdery mildew"],
      sun: "Full to part sun; cool season", water: "Even", spacingCm: "5", ph: "6.0–7.5",
    },
  },
];

export const cropByName = (name: string): Crop | undefined =>
  CROPS.find((c) => c.name.toLowerCase() === name.toLowerCase());

export type DailyPoint = { t: number; temperature_c: number | null; total_radiation_w_m2: number | null };

/** Degree-days above `baseC` from `sinceT` (unix s) up to now, over daily points. */
export function gddSince(points: DailyPoint[], baseC: number, sinceT: number): number {
  const now = Date.now() / 1000;
  return points.reduce(
    (sum, p) =>
      p.t >= sinceT && p.t <= now && p.temperature_c != null ? sum + Math.max(0, p.temperature_c - baseC) : sum,
    0,
  );
}

/** Mean daily light integral over the last `days` observed daily points. */
export function recentDli(points: DailyPoint[], days = 7): number | null {
  const now = Date.now() / 1000;
  const recent = points
    .filter((p) => p.t <= now && p.total_radiation_w_m2 != null)
    .slice(-days)
    .map((p) => (p.total_radiation_w_m2 as number) * DLI_PER_WM2_DAY);
  return recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : null;
}
