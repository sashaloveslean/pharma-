#!/usr/bin/env python3
"""Extract structured HPLC (chromatography) conditions from dissolution chunks.

The pharma НД documents describe each analytical method with a fairly stable
"Хроматографические условия" block: column, temperature, detector wavelength,
mobile phase (+ pH), flow rate, injection volume, run time.

This module turns that free (OCR) text into structured records so the data can
be used as labels for a predictive model. The extraction is deterministic
(regex/rules) — no API key required.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


# --- stationary phase / column ------------------------------------------------

PHASE_RE = re.compile(
    r"\b(C\s?30|C\s?18|C\s?8|C\s?4|C\s?1\b|фенил|phenyl|HILIC|амино|amino|"
    r"циано|cyano|\bCN\b|NH2)\b",
    re.IGNORECASE,
)

# 150 × 4,6 мм  /  250x4.6 mm  /  100 х 3,0 мм
DIMENSIONS_RE = re.compile(
    r"(\d{2,3})\s*[×xхX*]\s*(\d{1,2}[.,]?\d?)\s*мм",
)

PARTICLE_RE = re.compile(r"(\d[.,]?\d?)\s*мкм")
PORE_RE = re.compile(r"(\d{2,3})\s*(?:A|Å|À|А)\b")

# well-known column brand families, used to normalise the raw string
COLUMN_BRANDS = [
    "YMC", "Nucleosil", "Hypersil", "Zorbax", "Symmetry", "XBridge", "XTerra",
    "Luna", "Kromasil", "Inertsil", "Chromolith", "Acquity", "Ascentis",
    "Poroshell", "SunFire", "Discovery", "Диасфер", "Диасорб", "Kinetex",
    "Zic-HILIC", "Gemini", "Atlantis", "Purospher", "LiChrospher",
]


# --- detector -----------------------------------------------------------------

WAVELENGTH_RE = re.compile(r"(\d{3})\s*нм")
DETECTOR_DAD_RE = re.compile(r"диодн|DAD|ДМД", re.IGNORECASE)
DETECTOR_UV_RE = re.compile(r"\bУФ\b|спектрофотометрическ|UV", re.IGNORECASE)
DETECTOR_FLUOR_RE = re.compile(r"флуориметр|флюориметр|fluoresc", re.IGNORECASE)
DETECTOR_RI_RE = re.compile(r"рефрактометр|refractive", re.IGNORECASE)


# --- mobile phase -------------------------------------------------------------

PH_RE = re.compile(r"p\s?[HН]\s*[:=]?\s*(\d{1,2}[.,]\d|\d{1,2})")
FLOW_RE = re.compile(r"(\d{1,2}[.,]\d{1,2})\s*мл\s*/?\s*мин")
INJECTION_RE = re.compile(r"(\d{1,3})\s*мкл")
TEMPERATURE_RE = re.compile(r"(\d{2})\s*°?\s*[CС]\b")
RUNTIME_RE = re.compile(r"(\d{1,3})\s*мин")

SOLVENTS = {
    "acetonitrile": re.compile(r"ацетонитрил|acetonitrile", re.IGNORECASE),
    "methanol": re.compile(r"метанол|methanol", re.IGNORECASE),
    "water": re.compile(r"\bвод[аеы]\b|water", re.IGNORECASE),
    "thf": re.compile(r"тетрагидрофуран|\bТГФ\b|THF", re.IGNORECASE),
    "phosphate_buffer": re.compile(r"фосфат", re.IGNORECASE),
    "acetate_buffer": re.compile(r"ацетат", re.IGNORECASE),
    "tfa": re.compile(r"трифторуксусн|\bТФУ\b|TFA", re.IGNORECASE),
    "triethylamine": re.compile(r"триэтиламин|triethylamine", re.IGNORECASE),
}

# Reagent catalog: canonical Russian name -> (category, regex of RU/EN synonyms).
# These are the chemicals a chemist must have on hand to run the method — the most
# actionable prediction. Ordered most-specific first so ion-pair salts and buffer
# salts are matched before the generic "фосфат"/"ацетат" family words.
REAGENTS: list[tuple[str, str, "re.Pattern[str]"]] = [
    # ion-pairing agents
    ("Натрия пентансульфонат", "ion_pairing", re.compile(r"пентансульфонат|pentanesulfonate", re.I)),
    ("Натрия гексансульфонат", "ion_pairing", re.compile(r"гексансульфонат|hexanesulfonate", re.I)),
    ("Натрия гептансульфонат", "ion_pairing", re.compile(r"гептансульфонат|heptanesulfonate", re.I)),
    ("Натрия октансульфонат", "ion_pairing", re.compile(r"октансульфонат|octanesulfonate", re.I)),
    ("Натрия додецилсульфат", "ion_pairing", re.compile(r"додецилсульфат|лаурилсульфат|dodecyl sulfate|SDS", re.I)),
    # buffer / salts
    ("Калия дигидрофосфат", "buffer_salt", re.compile(r"кали[йя]\s+дигидро(?:орто)?фосфат|дигидрофосфат\s+кали|однозамещ\w*\s+фосфорнокисл\w*\s+кали|KH2PO4", re.I)),
    ("Натрия дигидрофосфат", "buffer_salt", re.compile(r"натри[йя]\s+дигидро(?:орто)?фосфат|дигидрофосфат\s+натри|NaH2PO4", re.I)),
    ("Натрия гидрофосфат", "buffer_salt", re.compile(r"натри[йя]\s+гидро(?:орто)?фосфат|гидрофосфат\s+натри|Na2HPO4", re.I)),
    ("Калия гидрофосфат", "buffer_salt", re.compile(r"кали[йя]\s+гидро(?:орто)?фосфат|K2HPO4", re.I)),
    ("Аммония ацетат", "buffer_salt", re.compile(r"аммони[йя]\s+ацетат|ацетат\s+аммони|ammonium acetate", re.I)),
    ("Аммония формиат", "buffer_salt", re.compile(r"аммони[йя]\s+формиат|формиат\s+аммони|ammonium formate", re.I)),
    ("Аммония дигидрофосфат", "buffer_salt", re.compile(r"аммони[йя]\s+дигидрофосфат|ammonium.*phosphate", re.I)),
    ("Натрия перхлорат", "buffer_salt", re.compile(r"перхлорат\s+натри|натри[йя]\s+перхлорат|perchlorate", re.I)),
    # acids
    ("Ортофосфорная кислота", "acid", re.compile(r"(?:орто)?фосфорн\w*\s+кислот|phosphoric acid", re.I)),
    ("Трифторуксусная кислота", "acid", re.compile(r"трифторуксусн\w*\s+кислот|\bТФУ\b|trifluoroacetic|TFA", re.I)),
    ("Хлористоводородная кислота", "acid", re.compile(r"хлористоводородн\w*\s+кислот|солян\w*\s+кислот|hydrochloric", re.I)),
    ("Уксусная кислота", "acid", re.compile(r"уксусн\w*\s+кислот|acetic acid", re.I)),
    ("Муравьиная кислота", "acid", re.compile(r"муравьин\w*\s+кислот|formic acid", re.I)),
    ("Серная кислота", "acid", re.compile(r"серн\w*\s+кислот|sulfuric acid", re.I)),
    # bases / modifiers
    ("Триэтиламин", "modifier", re.compile(r"триэтиламин|triethylamine|\bТЭА\b", re.I)),
    ("Диэтиламин", "modifier", re.compile(r"диэтиламин|diethylamine", re.I)),
    ("Натрия гидроксид", "base", re.compile(r"натри[йя]\s+гидроксид|гидроксид\s+натри|едк\w*\s+натр|NaOH", re.I)),
    ("Калия гидроксид", "base", re.compile(r"кали[йя]\s+гидроксид|гидроксид\s+кали|KOH", re.I)),
    ("Аммиак", "base", re.compile(r"аммиак\w*|ammonia|аммони[йя]\s+гидроксид", re.I)),
    # organic solvents
    ("Ацетонитрил", "organic_solvent", re.compile(r"ацетонитрил|acetonitrile", re.I)),
    ("Метанол", "organic_solvent", re.compile(r"метанол|methanol", re.I)),
    ("Тетрагидрофуран", "organic_solvent", re.compile(r"тетрагидрофуран|\bТГФ\b|tetrahydrofuran|THF", re.I)),
    ("Изопропанол", "organic_solvent", re.compile(r"изопропанол|пропанол-2|2-пропанол|изопропилов\w*\s+спирт|isopropanol", re.I)),
    ("Бутанол", "organic_solvent", re.compile(r"бутанол|butanol", re.I)),
    ("Этанол", "organic_solvent", re.compile(r"этанол|ethanol", re.I)),
]

# a chunk is treated as a chromatography-conditions block if it looks like one
BLOCK_SIGNAL_RE = re.compile(
    r"хроматографическ\w*\s+услови|подвижная\s+фаза|скорость\s+потока",
    re.IGNORECASE,
)


@dataclass
class Conditions:
    source: str
    start_page: int | None
    section: str | None

    column_raw: str | None = None
    column_phase: str | None = None
    column_length_mm: float | None = None
    column_id_mm: float | None = None
    particle_um: float | None = None
    pore_a: int | None = None

    column_temp_c: int | None = None

    detector: str | None = None
    wavelengths_nm: list[int] = field(default_factory=list)

    mobile_phase_raw: str | None = None
    mobile_phase_ph: float | None = None
    solvents: list[str] = field(default_factory=list)

    flow_ml_min: float | None = None
    injection_ul: int | None = None
    runtime_min: int | None = None

    reagents: list[dict] = field(default_factory=list)
    mobile_phase_ratio: str | None = None
    mobile_phase_components: list[dict] = field(default_factory=list)

    def completeness(self) -> int:
        core = [
            self.column_phase, self.column_length_mm, self.wavelengths_nm,
            self.flow_ml_min, self.mobile_phase_raw,
        ]
        return sum(1 for value in core if value)


def _num(value: str) -> float:
    return float(value.replace(",", "."))


def parse_column(text: str) -> dict:
    result: dict = {}
    phase = PHASE_RE.search(text)
    if phase:
        result["column_phase"] = re.sub(r"\s+", "", phase.group(1)).upper()
    dims = DIMENSIONS_RE.search(text)
    if dims:
        result["column_length_mm"] = _num(dims.group(1))
        result["column_id_mm"] = _num(dims.group(2))
    particle = PARTICLE_RE.search(text)
    if particle:
        value = _num(particle.group(1))
        if 1.0 <= value <= 20.0:  # HPLC particle sizes; larger = pore/filter noise
            result["particle_um"] = value
    pore = PORE_RE.search(text)
    if pore:
        result["pore_a"] = int(pore.group(1))

    for brand in COLUMN_BRANDS:
        match = re.search(re.escape(brand), text, re.IGNORECASE)
        if match:
            window = text[match.start(): match.start() + 60]
            result["column_raw"] = re.sub(r"\s+", " ", window).strip()
            break
    return result


def parse_detector(text: str) -> dict:
    result: dict = {}
    if DETECTOR_DAD_RE.search(text):
        result["detector"] = "DAD"
    elif DETECTOR_FLUOR_RE.search(text):
        result["detector"] = "fluorescence"
    elif DETECTOR_RI_RE.search(text):
        result["detector"] = "RI"
    elif DETECTOR_UV_RE.search(text):
        result["detector"] = "UV"

    waves = sorted({int(w) for w in WAVELENGTH_RE.findall(text) if 190 <= int(w) <= 400})
    if waves:
        result["wavelengths_nm"] = waves
    return result


def parse_mobile_phase(text: str) -> dict:
    result: dict = {}
    ph = PH_RE.search(text)
    if ph:
        value = _num(ph.group(1))
        if 1.0 <= value <= 12.0:
            result["mobile_phase_ph"] = value

    solvents = [name for name, pattern in SOLVENTS.items() if pattern.search(text)]
    if solvents:
        result["solvents"] = solvents

    # capture the phrase after "подвижная фаза" up to the next field label
    marker = re.search(r"подвижная\s+фаза", text, re.IGNORECASE)
    if marker:
        window = text[marker.end(): marker.end() + 200]
        # stop at the next condition-block field so the recipe stays clean
        cut = re.search(
            r"(скорость\s+потока|температур|вводимый\s+объ|время\s+хроматограф|"
            r"промывочн|раствор\s+для\s+промывки|детектор|\bНД\b\s*[СC]\.)",
            window,
            re.IGNORECASE,
        )
        if cut:
            window = window[: cut.start()]
        window = re.sub(r"\s+", " ", window).strip(" :\n-")
        if len(window) >= 5:
            result["mobile_phase_raw"] = window
    return result


def parse_operating(text: str) -> dict:
    result: dict = {}

    flow = FLOW_RE.search(text)
    if flow:
        value = _num(flow.group(1))
        if 0.1 <= value <= 5.0:
            result["flow_ml_min"] = value

    injection = INJECTION_RE.search(text)
    if injection:
        value = int(injection.group(1))
        if 1 <= value <= 200:
            result["injection_ul"] = value

    # temperature: look near a temperature/column cue
    temp_zone = text
    cue = re.search(r"температур\w*\s+колонк\w*", text, re.IGNORECASE)
    if cue:
        temp_zone = text[cue.end(): cue.end() + 40]
    temp = TEMPERATURE_RE.search(temp_zone)
    if temp:
        value = int(temp.group(1))
        if 15 <= value <= 80:
            result["column_temp_c"] = value

    runtime_cue = re.search(r"время\s+хроматографир\w*", text, re.IGNORECASE)
    if runtime_cue:
        window = text[runtime_cue.end(): runtime_cue.end() + 40]
        runtime = RUNTIME_RE.search(window)
        if runtime:
            result["runtime_min"] = int(runtime.group(1))
    return result


# "(85:15)"  "(69: 18:8: 5)"  "(50 : 45 : 5)"
RATIO_RE = re.compile(r"\(\s*(\d{1,3})\s*(?::\s*\d{1,3}\s*){1,4}\)")
COMPONENT_SPLIT_RE = re.compile(r"\s*[:/]\s*|\s+[-–—]\s+|(?<=[а-яa-z])-(?=[а-яa-z])")


# tokens that are labels/boilerplate, not real mobile-phase solutions
COMPONENT_STOPWORDS = re.compile(
    r"подвижн|аналогичн|\bили\b|состав|раствор\s+для|промывк|уравновеш|"
    r"\bпф\b|\bсорбент|колонк|скорост|детектор|приготовл",
    re.IGNORECASE,
)


def _canonical_component(raw: str) -> str | None:
    token = re.sub(r"\s+", " ", raw).strip(" ,;.:-–—()")
    if not token or len(token) < 3:
        return None
    if COMPONENT_STOPWORDS.search(token) or len(token.split()) > 4:
        return None
    lower = token.lower()
    named = {
        "ацетонитрил": r"ацетонитрил",
        "метанол": r"метанол",
        "вода": r"\bвод[аеоы]",
        "фосфатный буфер": r"фосфат",
        "ацетатный буфер": r"ацетат",
        "буфер": r"буфер",
        "тетрагидрофуран": r"тетрагидрофуран|тгф",
        "бутанол": r"бутанол",
        "изопропанол": r"изопропанол|пропанол",
        "этилацетат": r"этилацетат|этил\W*ацетат",
        "гептан": r"гептан",
        "гексан": r"гексан",
    }
    for canonical, pattern in named.items():
        if re.search(pattern, lower):
            return canonical
    # keep an unrecognised but plausible solvent word as-is (short, single phrase)
    if len(token) <= 40 and re.search(r"[а-яёa-z]", lower):
        return token
    return None


def parse_mobile_phase_composition(text: str) -> dict | None:
    """Extract the mobile-phase solutions and their ratio, e.g.
    'вода - ацетонитрил (85:15)' -> ratio '85:15', components вода/ацетонитрил."""
    match = RATIO_RE.search(text)
    if not match:
        return None
    ratio_numbers = [int(n) for n in re.findall(r"\d{1,3}", match.group(0))]

    prefix = text[: match.start()].rstrip(" :")
    prefix = prefix[-80:]  # component list sits right before the ratio
    parts = [p for p in COMPONENT_SPLIT_RE.split(prefix) if p.strip()]
    components: list[str] = []
    for part in reversed(parts):  # walk back from the ratio
        canonical = _canonical_component(part)
        if canonical:
            components.insert(0, canonical)
        if len(components) >= len(ratio_numbers):
            break

    if len(components) != len(ratio_numbers):
        # ratio found but components unclear — still return the ratio
        return {"ratio": ":".join(str(n) for n in ratio_numbers), "components": []}

    return {
        "ratio": ":".join(str(n) for n in ratio_numbers),
        "components": [
            {"solution": name, "part": part}
            for name, part in zip(components, ratio_numbers)
        ],
    }


def parse_reagents(text: str) -> list[dict]:
    """Detect named reagents (solvents, buffer salts, acids, ion-pair agents)."""
    found: list[dict] = []
    seen: set[str] = set()
    for name, category, pattern in REAGENTS:
        if pattern.search(text) and name not in seen:
            seen.add(name)
            found.append({"name": name, "category": category})
    return found


def extract_from_chunk(chunk: dict) -> Conditions | None:
    text = chunk["text"]
    if not BLOCK_SIGNAL_RE.search(text):
        return None

    metadata = chunk["metadata"]
    conditions = Conditions(
        source=Path(metadata["source"]).name,
        start_page=metadata.get("start_page"),
        section=metadata.get("section"),
    )
    for parsed in (
        parse_column(text),
        parse_detector(text),
        parse_mobile_phase(text),
        parse_operating(text),
    ):
        for key, value in parsed.items():
            setattr(conditions, key, value)

    conditions.reagents = parse_reagents(text)
    composition = parse_mobile_phase_composition(text)
    if composition:
        conditions.mobile_phase_ratio = composition["ratio"]
        conditions.mobile_phase_components = composition["components"]

    # require at least a real column or a mobile phase to keep the record
    if conditions.completeness() < 2:
        return None
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/dissolution_chunks.jsonl")
    parser.add_argument("--output", default="data/processed/chromatography_conditions.jsonl")
    args = parser.parse_args()

    records: list[dict] = []
    with open(args.input, encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            conditions = extract_from_chunk(chunk)
            if conditions:
                records.append(asdict(conditions))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Extracted {len(records)} condition blocks to {args.output}")


if __name__ == "__main__":
    main()
