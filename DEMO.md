# Demo Rehearsal Runbook (task 8.3)

How to set up, run and rehearse the live dashboard demo. Everything is timed and checked
on this project's laptop (24 Nov 2026). Every number in the talking points is read from
the dashboard itself, so you can point at it while you say it.

---

## 1. One-time setup (the day before)

> ⚠️ **`python` on this laptop is Python 3.11, which doesn't have the project's packages.**
> Running `python src/...` fails with `No module named 'geopandas'`. Always use
> **`py -3.13`**. Run every command from the repo root (`D:\spatiotemporal_graph`).

1. **Pull `main`.** Make sure the dashboard changes are merged.
2. **Check the generated data exists.** `data/processed/` is gitignored, so it only exists on
   the laptop that ran the pipeline. Present from **this laptop**. If anything is missing,
   regenerate it in this order:
   ```powershell
   py -3.13 src/dashboard/prepare_dashboard_data.py          # per-phase map layers
   py -3.13 src/optimizer/build_vulnerability_index.py       # task 7.1
   py -3.13 src/optimizer/equity_optimizer.py                # task 7.2
   py -3.13 src/optimizer/build_resource_plan.py             # task 7.3
   ```
3. **Run the tests:** `py -3.13 -m pytest -q`. Expect **378 passed**.
4. **Internet:** the map backgrounds (Esri tiles) and the font (Google Fonts) load from the
   internet. Without a connection the roads and markers still draw, but on a blank
   background. Test on the venue's Wi-Fi if you can; otherwise use a phone hotspot.
5. **Backup:** copy `docs/figures/*.png` to the desktop. They're static versions of the
   key charts (see §6).

## 2. Thirty minutes before you present

| Step | Action | Why |
|---|---|---|
| 1 | `py -3.13 -m streamlit run src/dashboard/app.py` | Starts the app. Leave this window open and don't close it. |
| 2 | Open `http://localhost:8501` in Chrome | The first page takes about **10 s**. |
| 3 | Click **every tab once**. On **Flood Prediction**, select Ground Truth, then Baseline, then GNN | Each map view builds on first use (**~20 s** cold). After this, everything is instant. |
| 4 | Press **▶ Play** once on the flood map, then **Pause** | Confirms the animation works. |
| 5 | On **Fair Relief**, check the sliders read **ρ = 0.60** and **25%** | They're the recommended setting. Refresh the page if you moved them. |
| 6 | Press **F11** (full screen) and set browser zoom to **110–125%** | Makes the text readable from the back of the room. |
| 7 | Go back to **🧭 The Story** | That's where the demo starts. |

Turn off notifications (Windows Focus Assist) and close Slack/WhatsApp so nothing pops up
on screen.

## 3. The demo script (~12 minutes)

Suggested split for 4 presenters, one per track (swap freely):

| Part | Tab | Who | Time |
|---|---|---|---|
| A | 🧭 The Story | P2 | 2 min |
| B | 🌊 Flood Prediction + 📈 Model Verdict | P1 + P3 | 4 min |
| C | 📡 Distress Signals | P4 | 2.5 min |
| D | ⚖️ Fair Relief | P4 | 2.5 min |
| E | 🔍 Limits & Honesty + close | P2 | 1 min |

### A · The Story (2 min)

- **Open on:** the orange "The problem" banner.
- **Say:** "In December 2015, Chennai had its heaviest single-day rainfall since 1901.
  Three things go wrong in a flood like this: flood maps react too late, help lines only
  hear from people who can reach them, and relief follows whoever is loudest. We built one
  system that tackles all three."
- **Point at the three stages, Predict → Listen → Decide:** "Each part feeds the next: the
  flood state and the distress signals both flow into the relief decision."
- **Land the three findings:**
  1. **0.00**: "A rule-based model never sees the flood coming."
  2. **97%**: "Free text becomes a correct map location."
  3. **3 of 4**: "The most vulnerable wards were silent."

### B · Flood Prediction → Model Verdict (4 min)

- **On Flood Prediction:** point at the four phase cards. "The event runs 8 Nov to 14 Dec;
  the peak is 1 and 2 December."
- **Ground Truth → ▶ Play.** Wait for the Peak. "This is the observed flood from satellite
  radar, NRSC data and news: 87% of roads under water at the peak. Low ground fills first."
- **Switch to Baseline Prediction → ▶ Play.** ⭐ **This is the moment to slow down.** "The
  hand-written rule stays blue through the peak. It only turns red when the water is
  already receding."
- **Scroll to "Share of roads flooded, phase by phase":** point at the two annotations.
  "Baseline misses the peak entirely… and reacts only afterwards. The graph model's average
  flood probability climbs from 0.27 to 0.58 as the flood arrives."
- **Model Verdict tab:**
  - **Left chart:** "At flood onset the rule scores zero, on every group of wards."
  - **Right chart:** "AUC tests whether the model can *rank* which roads flood. On unseen
    validation wards it's 0.59 to 0.60, above a coin flip. Modest, but real."
- **Say it before they ask:** "The test wards come out near 0.5. That's a ground-truth
  limitation we diagnosed, not a bug. It's on the last tab."

### C · Distress Signals (2.5 min)

- **Walk the four steps:** raw text → is it distress? → where? → on the map.
- **Click one pulsing marker** to show the real 2015 passage behind it.
- ⭐ **Live box:** clear it and type something new, for example:
  `Boats needed in Saidapet and Kotturpuram, water is rising fast`. The green chips and map
  pins appear instantly.
- **Then type** `please help us` → "No known place names found". "It refuses to guess
  rather than inventing a location."
- **Be clear about what runs live:** the geoparser runs live. The 237M-parameter classifier
  doesn't; the markers use its saved predictions.

### D · Fair Relief (2.5 min)

- **Point at the four tiles.** "628,140 people affected. 46% of them live in wards that
  posted no distress at all. Allocating by reported distress leaves **12 of 16 wards**
  without relief; our plan leaves **0**."
- ⭐ **Slider live:**
  - Drag **ρ to 0**. The silent wards' bars disappear and "Starved wards" jumps to **12**.
  - Drag it back to **0.6**: back to **0**. "That costs only **7.2%** of the reported-distress
    coverage."
  - Then try **ρ = 1**. "Even stronger protection: the most vulnerable wards reach 19%."
- **Scroll to the plan table:** "Every ward gets a dispatch order, water, food rations and
  emergency toilets, sized by the Sphere humanitarian minimums. The Firmness column says
  which numbers to trust."
- **Say the limitation yourself:** the "What the floor does not fix" card. "Above the
  floor, money still follows the loudest wards: 70% to three wards. That's a modelling
  choice we've left open."
- **Before moving on:** put the sliders back to **0.6 / 25%**.

### E · Limits & Honesty + close (1 min)

- **Scroll through the cards.** "Every headline number has a limitation that shapes it, and
  each one was found by checking the data."
- **Closing line:** "Predict, listen, decide: one connected system, built and tested on real
  data from a real flood."

## 4. Rehearsal plan

| Run | Goal | How |
|---|---|---|
| **1 · Solo walk-through** | Learn the clicks | Each presenter runs their part alone with this script open, and times it. |
| **2 · Full run, timed** | Hit ~12 min | Everyone presents in order, with one person keeping time. Cut words, not clicks, if you run over. |
| **3 · Mock Q&A** | Not get caught out | One teammate fires the §5 questions at random. Every answer under 30 s. |
| **4 · Disaster drill** | Stay calm | Mid-run, close the browser tab and recover using §6. |
| **5 · Dress rehearsal** | Venue conditions | Projector, venue Wi-Fi, full screen, zoom set, notifications off. |

**After each run:** note any stumble, fix the wording here, and re-run that part only.

## 5. Likely questions, with honest answers

| Question | Answer (30 s) |
|---|---|
| "The GNN F1 is 0.98. Isn't that suspicious?" | Yes, and we say so: the thresholded model flags every road. The honest measure is AUC on unseen wards: 0.59–0.60 on validation. See the Limits tab. |
| "Why is test AUC near 0.5?" | Only 1 of 16 wards has a real mix of flooded and dry roads, and it fell in validation, not test. The test wards are over 96% flooded, so no model has anything to rank. That's a ground-truth limit, not a bug. |
| "Why not real tweets?" | Historical X/Twitter search is Enterprise-only (~$42k/month). We used real published 2015 reporting instead: 19 passages, 17 labelled. |
| "Census 2011 is 15 years old." | It's the latest ward-level census. We also had to rebuild the post-2011 wards from the old 155-ward map, and say so. |
| "Why ρ = 0.6?" | It's the cheapest setting where no ward is starved: it costs 7.2% of reported-distress coverage. The slider lets the decision-maker choose differently. |
| "Why observed flooding, not the GNN, for the relief plan?" | Relief is sent during and after the peak, when observed extent exists. We ran the GNN forecast as a sensitivity check, and the conclusion holds. |
| "Would this work live?" | The pipeline is in place; live use needs a streaming text source and a fresh satellite pass. It's a proven architecture, not a deployed system. |
| "What would you do next?" | Re-split wards by label variance (fixes the test-AUC issue), a per-ward cap in the optimizer, and a larger real social-media corpus. |

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| Page blank or shows an error | In the terminal, press `Ctrl+C`, run `py -3.13 -m streamlit run src/dashboard/app.py` again, then refresh. Ready in about 10 s. |
| Map shows roads but no background | Internet is down. Keep going: the roads and markers still tell the story. |
| Play button does nothing | Refresh the page and click the Flood Prediction tab again. |
| A slider was left in a weird state | Refresh the page: the sliders reset to ρ = 0.6 / 25%. |
| Laptop dies | Present the static figures from `docs/figures/`: `phase5_f1_comparison.png`, `phase7_vulnerability_index.png`, `phase7_allocation_comparison.png`, `phase7_resource_plan.png`. |
