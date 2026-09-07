# Engineering Blueprint: Long-Horizon Autonomy for TWFarmBot

A single **See → Reason → Act → Reason** cycle solves the *instantaneous* loop. But crops grow over 60–120 days. To make this system work reliably over weeks and months without human babysitting, it must solve the 5 classic challenges of long-horizon autonomous robotics:

```
                  ┌────────────────────────────────────────┐
                  │          Long-Horizon Engine           │
                  └───────────────────┬────────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│   Temporal   │              │   Delayed    │              │  Cumulative  │
│    Memory    │              │ Verification │              │    Safety    │
│  (No amnesia)│              │ (24h-48h loop│              │  (Daily caps)│
└──────────────┘              └──────────────┘              └──────────────┘
       │                              │                              │
       └──────────────────────────────┼──────────────────────────────┘
                                      │
       ┌──────────────────────────────┴──────────────────────────────┐
       ▼                                                             ▼
┌──────────────┐                                             ┌──────────────┐
│   Adaptive   │                                             │ Self-Healing │
│   Cadence    │                                             │  & Baselines │
│  (Sun/Stress)│                                             │ (Homing/Warp)│
└──────────────┘                                             └──────────────┘
```

---

## 1. Temporal Memory: Ending Agent Amnesia

### The Problem
Currently, every autonomous cycle starts from scratch. If the agent waters at $(x=200, y=500)$ at 10:00, the 14:00 cycle has zero knowledge of it. It doesn't know if the reading improved, degraded, or stayed flat.

### The Solution: Coordinate-Keyed Ledger (SQLite via stdlib)
Use Python's built-in `sqlite3` (zero new dependencies) to record time-series observations indexed by spatial coordinate:

```sql
CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    ndre_mean REAL,
    stress_fraction REAL,
    veg_fraction REAL,
    soil_moisture REAL,
    action_taken TEXT,       -- "water_30s", "none", etc.
    action_seconds REAL,
    summary TEXT
);

CREATE INDEX idx_obs_pos_time ON observations (x, y, timestamp);
```

### How the Agent Uses It
Before reasoning, the agent automatically queries the nearest historical observations within $\pm 100\text{ mm}$:

> *"At $(200, 500)$, 24 hours ago: NDRE was $0.08$ (stressed), watered $25\text{s}$. Current NDRE is $0.15$ ($\mathbf{+87\%}$ vigor recovery). Soil moisture rose from $22\%$ to $48\%$."*

The LLM switches from reactive snapshot thinking to **rate-of-change ($\Delta\text{NDRE}/\Delta t$) agronomic reasoning**.

---

## 2. Delayed-Feedback Verification (Closing the Temporal Loop)

### The Problem
In robotics, motor moves are instant. But in biology, **feedback is delayed**:
* Plants take **12–48 hours** to show spectral recovery after irrigation.
* An immediate post-water photo shows wet soil, not revitalized chlorophyll.

### The Solution: Hypothesis-Verification Registry
When the agent takes an action, it logs an **Agronomic Hypothesis**:

```json
{
  "intervention_id": "act_8f29",
  "timestamp": "2026-09-06T08:00:00Z",
  "target": {"x": 200, "y": 500, "z": 0},
  "action": {"kind": "water", "seconds": 30},
  "hypothesis": "Canopy is water-stressed; expect stress_fraction < 0.2 within 24h",
  "verify_after": "2026-09-07T08:00:00Z",
  "verified": false
}
```

On subsequent cycles, the worker checks `verify_after <= now()`:
1. Returns to $(200, 500)$ to observe the outcome.
2. Compares the actual result against the hypothesis.
3. If successful: records the win in memory.
4. If failed: escalates (e.g. flags potential drip nozzle blockage or chronic root issue).

---

## 3. Cumulative Safety: 24-Hour Resource Budgets

### The Problem
`safety_service` currently enforces **instantaneous** safety:
`water(seconds <= 300)` is safe *once*. But an agent calling 300s every 20 minutes will flood the lab and kill the plants.

### The Solution: Sliding-Window Budget Gate
Enhance `safety_service` with temporal rate limiting:

| Safety Rule | Current (Instant) | Long-Horizon (Cumulative) |
|---|---|---|
| **Max water per action** | $\le 300\text{ s}$ | $\le 300\text{ s}$ |
| **Max water per 24h window** | None | $\le 120\text{ s}$ per $\text{m}^2$ |
| **Minimum soak interval** | None | $\ge 4\text{ hours}$ between watering at same $(x,y)$ |
| **Max total gantry travel / day** | None | $\le 200\text{ m}$ (prevents infinite jog loops) |

If an action breaches the 24h sliding window, `safety_service.validate()` raises `UnsafeActionError("24h watering budget exceeded (already watered 110s in last 24h)")`, which feeds back into the LLM prompt.

---

## 4. Adaptive Cadence: Sun-Aware & State-Driven Timing

### The Problem
A rigid `time.sleep(21600)` (every 6h) means:
* Observations at 12:00 PM suffer harsh sunlight and harsh shadows, corrupting NDRE calculation.
* A dying plant has to wait 6 hours for help.

### The Solution:
1. **Lighting-Aware Windows**:
   Restrict multi-spectral capture to diffuse lighting periods (morning/dusk or shaded enclosure), avoiding specular reflection on leaf wax.
2. **State-Driven Recheck Intervals**:
   Let the agent suggest its next wake-up interval:
   * Healthy canopy $\rightarrow$ wake up in $12\text{–}24\text{ hours}$.
   * Stressed / just watered $\rightarrow$ wake up in $3\text{ hours}$ for verification.

---

## 5. Self-Healing & Drift Compensation

Over weeks of autonomous operation, physical systems degrade:
1. **Stepper Motor Missed Steps**:
   Accumulated micro-slips cause the gantry to lose absolute zero.
   * *Rule*: Auto-execute `find_home("all")` every 24 hours or before any critical multi-spectral scan.
2. **Spectral Alignment Drift**:
   Vibrations can shift dual-camera calibration.
   * *Rule*: Run edge correlation between NIR and RedEdge during `analyze_spectral_pair`; if registration error exceeds $5\text{ px}$, trigger a notification.
3. **Camera Failure Closed**:
   If udev symlink `/dev/camera-nir` disconnects, the cycle cleanly skips with an alert rather than crashing.

---

## Recommended Execution Plan

To build this cleanly without bloat, we can implement it in 3 focused iterations:

1. **Iteration 1: Observation & History Ledger (SQLite)**
   Add a tiny `history_store.py` in `spatial_service` or `planning_service`. Auto-log observations on every cycle and inject recent coordinate history into the agent's prompt.
2. **Iteration 2: 24h Cumulative Safety Budgets**
   Add sliding-window rate limiting in `safety_service` so water and travel have daily caps.
3. **Iteration 3: Hypothesis Verification**
   Add the delayed-feedback check so the agent tracks whether past watering actually fixed the stress.
