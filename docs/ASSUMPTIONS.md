# Assumptions

These are the numbers we chose when we built the simulator and the model config, and why we chose them. All of them live in `configs/plant_line_a.yaml` or `configs/model.yaml`; none of them come from a real plant.

## 1. 24 stations

Nine body construction (BC-01 through BC-09), six paint (PT-01 through PT-06), nine final assembly (FA-01 through FA-09). This matches the hackathon spec's station count directly. We split it three ways by area because that's how a real body-in-white to paint to final-assembly line is organized, and because the defect taxonomy below (weld, paint, assembly) needed a physical area to anchor to. There's nothing special about 24 beyond that; it's big enough that a soft-sensor donor station is rarely adjacent to its blind station, which is the harder, more honest case.

## 2. 60 second takt

Every station has the same 60 second base cycle time. A mixed-model line in reality often has station-to-station takt variation, but we wanted buffer utilization and constraint identification to mean something without a second axis of complexity on top of it, so takt is uniform and the SUV variant's 1.05x cycle time multiplier is the only thing that stretches it. If we revisit this, station-specific takt is the first thing we'd add, not the last.

## 3. The 14 rich / 5 partial / 5 manual instrumentation mix

This is the hackathon spec's number, not one we picked freely. What we did choose is which specific stations land in which tier. Manual stations are the ones where a real line is genuinely hardest to instrument cheaply: torque and fastening checks (FA-05, FA-06, FA-08), and visual paint inspection (PT-05), where a sensor retrofit is either expensive or the defect is inherently a human judgment call. Partial stations sit next to a rich station covering a similar process (BC-05 door hang next to BC-06 body-side weld, PT-03 basecoat next to PT-04 clearcoat) specifically so the soft-sensor archetypes have a donor to borrow from. An instrumentation mix where the blind stations were scattered randomly would make soft sensing much easier than it should be, because you'd only ever need to borrow from a station a few steps away.

## 4. The 4.38% defect rate

84 of 1920 units in the seed-42 run have a ground-truth defect (`defects.csv`, any row, detected or not). We didn't target this number directly; it falls out of the fault sources in `configs/model.yaml`: a small background rate (0.02% per visit) plus tool wear ramps at BC-06 and FA-05, two supplier batch events, operator variation on the FA and PT-05 stations, and an ambient-humidity effect on paint. We tuned the fault source magnitudes so the combined rate would land in the low single digits, because that's roughly where "defects are rare enough that a naive high-recall alarm strategy blows the alarm budget instantly" starts being true, which is the operating condition this whole project is meant to be tested under. Of those 84, only 76 are actually caught by a gate (3.96% detected rate), because gate detection isn't perfect either. See the next item.

## 5. Detection probability at each gate

The paint gate (PT-06) catches 93% of paint defects that pass through it. The final gate (FA-09) catches 90% of weld and assembly defects. Neither is 100%, on purpose: a perfect inspection gate would make the root-cause and containment machinery pointless, because nothing would ever slip through uninspected. We set final assembly slightly lower than paint because a functional test and a dimensional check are, in our judgment, less likely to catch every defect type than a visual and gloss inspection is for paint, but we won't pretend this reflects a real audit of either process. It's a plausible number, not a measured one.

## 6. Rework cost of 250 currency units

This feeds the backtest's net-benefit calculation directly: a true positive is worth 250, a false alarm costs 40 in investigation time (`trust.investigation_cost_currency`). We didn't source either number from a real cost model. What we did do is pick a ratio, about 6 to 1, that makes the alarm budget's cost tradeoff actually bite: cheap enough investigation that a handful of false alarms doesn't sink you, expensive enough rework that missing a real defect isn't free. If you want the business case on the Leadership page to mean something for an actual plant, this is the first number to replace with a real one, because the whole net-benefit column scales off it.

## 7. No real company data anywhere

Every reading, every defect, every unit ID in this repo comes from `sim/`, seeded with 42. Sensor nominal values (weld current at 850A, film thickness at 25 microns, and so on) are plausible figures for the named process, not values pulled from any supplier datasheet or OEM spec. No proprietary process parameters, no real plant layout, no real defect rates went into this. That's a deliberate boundary, not an oversight: this project was built to be checked against a known, controllable ground truth, and mixing in real numbers from somewhere would have made it impossible to tell whether a result reflected the model or the data source.

## 8. Zero arrival jitter

Unit start times in the simulator are exactly one takt apart. No jitter, no stochastic gaps between units entering the line. That makes throughput deterministic: 60 units an hour, every hour, by construction, not something we measured. A real line has arrival variability, minor stoppages, and small timing drift between units. We left it out because it would not have changed any of the conclusions this project checks (defect detection, root-cause tracing, containment) and would have invalidated every run we had already validated against a known ground truth. If we added it later, we would add it after the fact and re-run the backtest, not fold it in quietly.
