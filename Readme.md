# Fly Brain Fuckery

Driving a virtual creature with a real fly brain. This project runs the
[FlyWire](https://codex.flywire.ai) *Drosophila melanogaster* connectome as an
actual spiking neural simulation, then uses its real outputs to control a
simple virtual bee — eventually aimed at a Minecraft mod.

Built on top of [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
(Shiu et al. 2023's whole-brain leaky integrate-and-fire model), with the
subsetting/discovery workflow inspired by
[Eon Systems' embodied fly-brain writeup](https://eon.systems/updates/embodied-brain-emulation).

## Demo

Scripted approach-and-feed sequence — bee moves toward the flower, real MN9
response crosses threshold, bee feeds:

![Visual sandbox demo](Drosophila_brain_model/renders/visual_sandbox_preview.gif)

Interactive version — click to place flower/water/bad food, bee reacts live,
including bad-food avoidance:

![Interactive sandbox demo](Drosophila_brain_model/renders/interactive_sandbox_preview.gif)

## What's real here, and what isn't

- **The brain is real.** Every neuron, every synapse weight, every
  excitatory/inhibitory sign comes directly from the FlyWire connectome. We
  don't train anything — we stimulate real sensory neurons and read real
  motor neuron output, the same way the original paper validates the model
  against actual optogenetic experiments.
- **The behavior decoding is not.** Turning "MN9 fired N times" into "the bee
  feeds" is a threshold we chose, not something derived from the connectome.
  Same for the bad-food repulsion logic in the sandbox — real bitter-neuron
  stimulation strength drives it, but the decision to turn that into movement
  is ordinary game logic, not a decoded brain output.
- **Locomotion (walking/flying) is not modeled.** FlyWire only covers the
  *brain*. Actual leg/wing motor control lives in the ventral nerve cord
  (VNC), a separate connectome (MANC) we don't use here. Descending neurons
  give us command-like signals; a real body simulator (e.g. NeuroMechFly)
  would be needed for anything below that.

## Pipeline

```
sandbox.py                    -- run real brain stimulation scenarios (sugar,
                                  bitter, water, combinations) against the
                                  FULL connectome, log which neurons actually
                                  fire into neuron_registry.json
        |
        v
build_subset.py                -- collapse the full 127k-neuron connectome
                                  down to just the neurons that appeared in
                                  the registry, producing
                                  completeness_subset.csv +
                                  connectivity_subset.parquet
        |
        v
precompute_response_curve.py   -- run the (much faster) subset network at
                                  several stimulation frequencies per sense,
                                  save the results as mn9_response_curves.json
        |
        v
visual_sandbox.py               -- scripted demo: bee approaches a flower,
interactive_sandbox.py             feeds when real MN9 response crosses
                                  threshold. Interactive version adds
                                  click-to-place flower/water/bad food, live
                                  aversion behavior, and an on-demand real
                                  (non-curve) brain verification check.
visualizer.py                   -- Dash web app: pick any of the 100+ real
                                  neuron populations we've catalogued,
                                  stimulate two at once, see an interactive
                                  raster + firing-rate chart.
```

Each stage's output feeds the next. If you change what the bee can sense
(a new sensory population), start back at `sandbox.py` so the subset network
and curves stay valid.

## Setup

```bash
conda create -n brain python=3.12 -y
conda activate brain
pip install brian2 numpy pandas pyarrow matplotlib pillow networkx dash plotly pygame joblib
```

Python 3.12 specifically — both Brian2's C-code generation and pygame's
prebuilt wheels lag behind on brand-new Python versions, which will otherwise
cost you a working afternoon on compiler errors that have nothing to do with
this project.

Run everything from this directory (`Drosophila_brain_model/`), not from
inside `scripts/` — the pipeline scripts open data files by bare relative
path, so the working directory has to stay at the repo root:

```bash
python scripts/sandbox.py
python scripts/build_subset.py
python scripts/precompute_response_curve.py
python scripts/interactive_sandbox.py
```

## Current neuron map

| sense/action | status | real cell type(s) |
|---|---|---|
| sweet taste | validated | sugar-sensing GRNs |
| bitter taste | validated | bitter-sensing GRNs |
| water/hygrosensation | validated | water GRNs |
| feeding (motor output) | validated | MN9 (proboscis motor neuron) |
| vision, olfaction | not yet looked up | in-scope for FlyWire, needs Codex lookup |
| walking, flight | named, no FlyWire IDs yet | oDN1, DNa01/DNa02 (command signal only — real motor pattern needs VNC/MANC, out of scope here) |
| grooming | named, no FlyWire IDs yet | aDN1, via Johnston's Organ mechanosensory input |

## Known limitations / next steps

- Response curves are single-trial (stochastic Poisson input, not averaged)
  — treat values as approximate, not precise measurements.
- The subset network is only valid for the stimulus conditions it was built
  from (`neuron_registry.json`). New senses need a fresh discovery pass.
- No live brain-driven locomotion yet — movement is greedy nearest-target
  game logic; only the feed/drink/avoid decision is brain-driven.
- Minecraft/Fabric port not started. Planned architecture: a fast locomotion
  loop (always real-time) fed by a slow brain-decision loop (realistically
  a few Hz at full connectome scale, faster on the subset network), bridged
  over a local socket to a persistent Python process holding the brain.

## Credits

- Connectome data and LIF model: Shiu et al. 2023,
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- Connectome: [FlyWire](https://flywire.ai)
- Architecture inspiration: [Eon Systems](https://eon.systems/updates/embodied-brain-emulation)
