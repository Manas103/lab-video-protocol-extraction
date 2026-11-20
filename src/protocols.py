"""Four canonical wet-lab protocol scripts, authored for this repository
(not transcribed from a real lab's SOP). Each step carries the sentence an
operator would actually narrate, a telemetry signature (which simulated
sensor channel should be active and at roughly what level while the step
runs), and a nominal duration used only to size the synthetic video; the
real, measured duration every session actually uses is the length of that
step's own synthesized narration clip (see session_builder.py), so video,
narration and telemetry are time-aligned to a real measurement, not to a
hand-picked number.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepSpec:
    name: str
    narration: str
    telemetry: str  # one of: "idle", "heat", "spin", "weigh", "timer"
    telemetry_level: float  # target magnitude for that channel


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    steps: tuple[StepSpec, ...]


DNA_EXTRACTION = ProtocolSpec(
    "dna_extraction",
    (
        StepSpec("lyse_sample", "Add lysis buffer to the sample tube and mix by inversion.", "idle", 0.0),
        StepSpec("add_proteinase", "Add proteinase K and vortex briefly.", "idle", 0.0),
        StepSpec("incubate", "Incubate the tube at fifty six degrees for ten minutes.", "heat", 56.0),
        StepSpec("add_ethanol", "Add cold ethanol and mix thoroughly.", "idle", 0.0),
        StepSpec("load_column", "Transfer the lysate onto the spin column.", "idle", 0.0),
        StepSpec("spin_wash", "Centrifuge the column at ten thousand RPM for one minute.", "spin", 10000.0),
        StepSpec("wash_buffer", "Wash the column with wash buffer and centrifuge again.", "spin", 10000.0),
        StepSpec("elute", "Elute the purified DNA into a clean tube.", "idle", 0.0),
    ),
)

GEL_ELECTROPHORESIS = ProtocolSpec(
    "gel_electrophoresis",
    (
        StepSpec("prepare_gel", "Pour the agarose gel into the casting tray.", "idle", 0.0),
        StepSpec("set_gel", "Let the gel set for twenty minutes at room temperature.", "idle", 0.0),
        StepSpec("load_buffer", "Place the gel into the tank and add running buffer.", "idle", 0.0),
        StepSpec("load_samples", "Load each sample into its own well with a pipette.", "idle", 0.0),
        StepSpec("run_voltage", "Run the gel at one hundred twenty volts for thirty minutes.", "spin", 120.0),
        StepSpec("stain_gel", "Stain the gel and image it under UV light.", "idle", 0.0),
    ),
)

CELL_PASSAGING = ProtocolSpec(
    "cell_passaging",
    (
        StepSpec("aspirate_media", "Aspirate the old media from the culture flask.", "idle", 0.0),
        StepSpec("rinse_pbs", "Rinse the cell layer with PBS.", "idle", 0.0),
        StepSpec("add_trypsin", "Add trypsin and incubate at thirty seven degrees for five minutes.", "heat", 37.0),
        StepSpec("neutralize", "Neutralize the trypsin with fresh growth media.", "idle", 0.0),
        StepSpec("count_cells", "Pipette a sample onto the hemocytometer and count the cells.", "weigh", 1.0),
        StepSpec("spin_cells", "Centrifuge the cell suspension at two hundred RPM for three minutes.", "spin", 200.0),
        StepSpec("replate", "Resuspend the pellet and replate into a new flask.", "idle", 0.0),
    ),
)

TITRATION = ProtocolSpec(
    "acid_base_titration",
    (
        StepSpec("rinse_burette", "Rinse the burette with distilled water.", "idle", 0.0),
        StepSpec("fill_burette", "Fill the burette with the titrant to the zero mark.", "idle", 0.0),
        StepSpec("weigh_sample", "Weigh out the analyte sample on the balance.", "weigh", 2.5),
        StepSpec("add_indicator", "Add a few drops of indicator to the flask.", "idle", 0.0),
        StepSpec("titrate", "Titrate dropwise while swirling until the color changes.", "timer", 1.0),
        StepSpec("record_volume", "Record the final burette reading.", "idle", 0.0),
    ),
)

ALL_PROTOCOLS: tuple[ProtocolSpec, ...] = (DNA_EXTRACTION, GEL_ELECTROPHORESIS, CELL_PASSAGING, TITRATION)

# 23 sessions total: 6 + 6 + 6 + 5 repetitions, the same uneven-but-balanced
# shape the real BioVL-QR dataset uses (six repetitions per experiment);
# authored here, not drawn from that or any other real dataset (see README
# "Honest framing").
SESSION_PLAN: tuple[tuple[str, int], ...] = (
    (DNA_EXTRACTION.protocol_id, 6),
    (GEL_ELECTROPHORESIS.protocol_id, 6),
    (CELL_PASSAGING.protocol_id, 6),
    (TITRATION.protocol_id, 5),
)

PROTOCOLS_BY_ID = {p.protocol_id: p for p in ALL_PROTOCOLS}

assert sum(n for _, n in SESSION_PLAN) == 23
