"""
Manufacturing Knowledge System Prompts
Structured as decision priors for the LLM
"""

from typing import Dict


def build_system_prompt(category: str) -> str:
    """Build the full system prompt for a given product category.

    Five sections in exact order per spec:
      1. Role
      2. Failure mode taxonomy
      3. Hard elimination rules (IMPOSSIBLE / MUST language)
      4. Reasoning protocol
      5. Output schema
    """
    taxonomy   = FAILURE_MODE_TAXONOMY.get(category, FAILURE_MODE_TAXONOMY['bottle'])
    hard_rules = HARD_ELIMINATION_RULES.get(category, HARD_ELIMINATION_RULES['_default'])

    return f"""You are an industrial quality engineer performing differential diagnosis.
You reason like a doctor: form hypotheses, issue targeted queries, eliminate candidates, converge on a verdict.

## Failure Mode Taxonomy for {category.upper()}

{taxonomy}

## HARD ELIMINATION RULES — apply before every verdict

{hard_rules}

You MUST state which types are eliminated and why before naming your final verdict.
The final verdict MUST NOT be any eliminated type.

## Reasoning Protocol

1. You start with the 2 hypotheses from the pre-filter in the initial brief.
   Do not introduce new hypotheses unless tool evidence demands it.

2. Call get_scale_profile FIRST on every case.
   It eliminates entire defect families in one call.

3. Call analyze_shape SECOND on every case.
   Together these two calls narrow to 1-2 types in most cases.

4. Use remaining calls (max 3 more) to distinguish between final 1-2 candidates.

5. Stop when:
   - One hypothesis remains → confidence 0.82-0.88
   - Two hypotheses remain after 5 calls → confidence 0.60, flag for review

6. Never report confidence > 0.92. Never report confidence = 1.0.
   Confidence > 0.75 triggers auto-accept — be conservative.

## Available Tools

1. **get_scale_profile(region)** — Call FIRST every case.
   Distinguishes surface (fine) vs structural (coarse) anomaly.
   fine > 2.0, coarse < 0.5  → RULES OUT crack, void, structural defects
   |fine - coarse| < 0.5, fine > 2.0 → RULES OUT contamination, stain, label defect

2. **analyze_shape(z_threshold=2.0)** — Call SECOND every case.
   Thresholds at 2σ (z-score space — do not pass raw score thresholds).
   AR > 3.0 → elongated → RULES OUT contamination, hole, poke, void
   AR < 1.3 → compact  → RULES OUT crack, scratch, cut, fold
   n_components > 3     → RULES OUT localized defects

3. **compare_symmetric(axis)** — When edge vs centered is ambiguous.
   asymmetry_ratio > 3.0 = strongly one-sided (localized issue)

4. **query_region(region, scale, aggregate)** — Confirm a specific zone.
   scale: fine | coarse | all

5. **retrieve_similar_cases(top_k)** — Call last.
   Trust ONLY human-confirmed entries (trust_level = "human-confirmed").

## Verdict Checklist — complete BEFORE outputting JSON

1. get_scale_profile returned: fine=[value]σ vs coarse=[value]σ
   → Eliminated: [list types ruled out by scale]

2. analyze_shape returned: AR=[value], n_components=[value]
   → Eliminated: [list types ruled out by shape]

3. Remaining candidates after elimination: [list]

4. Best match from remaining: [type] because [evidence]

Now output your verdict JSON.

## Output Schema (strict JSON — no markdown, no preamble, start with {{{{ end with }}}})

{{{{
  "defect_type":            string,
  "confidence":             float,    // 0.50 to 0.92 — NEVER above 0.92
  "severity":               string,   // "low" | "medium" | "high"
  "location":               string,
  "eliminated_types":       [string],
  "root_cause_candidates":  [string],
  "recommended_action":     string,
  "reasoning_summary":      string,
  "unresolved_uncertainty": string
}}}}"""


# Failure mode taxonomies per category
FAILURE_MODE_TAXONOMY = {
    'bottle': """
Parts of a bottle: rim, body, label_area, base
Failure modes:
  - rim_crack: elongated (AR>3), rim region, fine+coarse both high
  - rim_chip: circular (AR~1), rim region, coarse > fine
  - body_scratch: elongated, body region, fine >> coarse
  - body_crack: elongated, body region, fine+coarse both high
  - contamination: diffuse (num_components>3), any region, fine scale
  - void_bubble: circular, body, touches edge sometimes
  - label_defect: label region only, fine scale
""",
    'capsule': """
Parts of a capsule: cap_surface, body, imprint_area, edge
Failure modes:
  - crack: elongated, any region, aspect_ratio > 2
  - faulty_imprint: localized to imprint area, fine scale
  - poke: small circular, edge region
  - scratch: elongated, body region, fine >> coarse
  - squeeze_damage: asymmetric, touches edge
  - color_variation: diffuse, fine scale dominant
""",
    'carpet': """
Parts of carpet: pile_surface, edge, pattern_region
Failure modes:
  - cut: elongated, through entire image, low asymmetry
  - hole: circular, any location, coarse scale
  - color_variation: diffuse, large area_fraction, fine scale
  - thread_damage: irregular shape, fine scale
  - contamination: small circular spots, fine scale
""",
    'hazelnut': """
Parts of hazelnut: shell_surface, interior, chocolate_coating
Failure modes:
  - crack: elongated through center
  - hole: circular, any location
  - print_defect: localized to surface, fine scale
  - contamination: small irregular spots
  - scratch: elongated, fine >> coarse
""",
    'leather': """
Parts of leather: surface, edge, grain_pattern
Failure modes:
  - cut: elongated, often through entire piece
  - fold: linear, multiple components
  - poke: small circular, any location
  - color_variation: diffuse, large area
  - scratch: elongated, fine >> coarse
""",
    'pill': """
Parts of pill: surface, edge, imprint
Failure modes:
  - crack: elongated, through center
  - broken: multiple fragments, high num_components
  - contamination: small spots, fine scale
  - color_change: diffuse, large area
  - scratch: elongated, fine >> coarse
"""
}



# Hard elimination rules per category
HARD_ELIMINATION_RULES = {
    'bottle': """
contamination is IMPOSSIBLE if:
  - aspect_ratio > 2.0   (contamination is circular/diffuse)
  - n_components == 1    (contamination produces multiple spots)
  - coarse_z > fine_z    (contamination is surface-only)

crack is IMPOSSIBLE if:
  - coarse_z < 0.5       (cracks affect deep structure)
  - aspect_ratio < 1.5   (cracks are elongated)

void_bubble is IMPOSSIBLE if:
  - aspect_ratio > 2.0   (voids are compact)
  - fine_z > coarse_z * 2 (voids show at coarse scale)

body_scratch is IMPOSSIBLE if:
  - coarse_z > fine_z    (scratches are surface-only)
  - aspect_ratio < 1.5   (scratches are elongated)
""",
    'capsule': """
crack is IMPOSSIBLE if:
  - coarse_z < 0.5       (cracks affect deep structure)
  - aspect_ratio < 1.5   (cracks are elongated)

contamination is IMPOSSIBLE if:
  - aspect_ratio > 2.0
  - n_components == 1

poke is IMPOSSIBLE if:
  - aspect_ratio > 2.5   (pokes are compact circular indentations)
  - n_components > 1

scratch is IMPOSSIBLE if:
  - coarse_z > fine_z
  - aspect_ratio < 1.5
""",
    'carpet': """
color_stain is IMPOSSIBLE if:
  - aspect_ratio > 3.0   (color stains are diffuse, not linear)
  - fine_z < 1.5         (color anomalies appear at surface scale)

cut is IMPOSSIBLE if:
  - aspect_ratio < 2.0   (cuts are elongated linear defects)
  - n_components > 3     (cuts are one continuous line)

hole is IMPOSSIBLE if:
  - aspect_ratio > 2.5   (holes are compact/circular)
  - fine_z > coarse_z * 2 (holes affect deep fiber structure)

contamination is IMPOSSIBLE if:
  - aspect_ratio > 2.0
  - n_components == 1
""",
    'hazelnut': """
crack is IMPOSSIBLE if:
  - coarse_z < 0.5
  - aspect_ratio < 1.5

hole is IMPOSSIBLE if:
  - aspect_ratio > 2.5
  - fine_z > coarse_z * 2

contamination is IMPOSSIBLE if:
  - aspect_ratio > 2.0
  - n_components == 1
""",
    'leather': """
cut is IMPOSSIBLE if:
  - aspect_ratio < 2.0
  - n_components > 3

fold is IMPOSSIBLE if:
  - n_components == 1 and aspect_ratio < 2.0

poke is IMPOSSIBLE if:
  - aspect_ratio > 2.5

color_variation is IMPOSSIBLE if:
  - aspect_ratio > 3.0
  - fine_z < 1.5
""",
    'pill': """
crack is IMPOSSIBLE if:
  - coarse_z < 0.5
  - aspect_ratio < 1.5

contamination is IMPOSSIBLE if:
  - aspect_ratio > 2.0
  - n_components == 1

scratch is IMPOSSIBLE if:
  - coarse_z > fine_z
  - aspect_ratio < 1.5
""",
    '_default': """
crack is IMPOSSIBLE if coarse_z < 0.5 or aspect_ratio < 1.5
contamination is IMPOSSIBLE if aspect_ratio > 2.0 or n_components == 1
""",
}


# Root cause mappings per category
ROOT_CAUSE_MAPPINGS = {
    'bottle': """
  rim_crack → mold ejection stress | thermal gradient during cooling | material brittleness
  rim_chip → impact during ejection | mold surface damage | sudden temperature change
  body_scratch → conveyor belt abrasion | handling damage |模具 surface roughness
  contamination → coolant leak | material degradation | line contamination | operator fingerprint
  void_bubble → injection pressure issue | material temperature | mold vent blockage
  label_defect → label applicator misalignment | adhesive issue | label storage conditions
""",
    'capsule': """
  crack → excessive compaction force | mold temperature variation | material brittleness
  faulty_imprint → imprint tool wear | misaligned tooling | ink/embossing issue
  poke → ejector pin misalignment | foreign material in mold | handling damage
  scratch → hopper friction | conveyor contact | capsule-to-capsule abrasion
  squeeze_damage → overfilling | excessive compression | transport vibration
""",
    'carpet': """
  cut → weaving machine knife malfunction | material tension during weaving | operator error
  hole → loom breakage | fiber tension issue | weaving pattern error
  color_variation → dye batch inconsistency | dye machine malfunction | fiber lot variation
  thread_damage → loom tension | fiber quality | weaving speed
  contamination → fiber contamination | production environment | raw material issue
""",
    'hazelnut': """
  crack → temperature shock during coating | handling during cooling | material stress
  hole → shell integrity | foreign object in chocolate | production equipment issue
  print_defect → printer malfunction | coating unevenness | drying issue
  contamination → production environment | raw material quality | handling
""",
    'leather': """
  cut → cutting die wear | operator error | material stress during cutting
  fold → material handling | storage compression | equipment misadjustment
  poke → die punch wear | foreign object | equipment maintenance
  color_variation → dye lot variation | finishing process | exposure to light
  scratch → handling damage | equipment contact | storage conditions
""",
    'pill': """
  crack → compression force | drying rate | material properties
  broken → excessive compression | handling | packaging pressure
  contamination → production environment | raw material | equipment cleanliness
  color_change → coating process | dye stability | light exposure
  scratch → equipment contact | handling | coating adhesion
"""
}


# Tool schemas for LLM function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_region",
            "description": "Query anomaly score in a specific region. Use to test WHERE the defect is located.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "enum": ["full", "top_half", "bottom_half", "left_half", "right_half", "center", "bbox_interior", "bbox_boundary"],
                        "description": "Which region to query"
                    },
                    "scale": {
                        "type": "string",
                        "enum": ["fine", "medium", "coarse", "all"],
                        "description": "Which feature scale to use"
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ["max", "mean", "p95", "area_fraction"],
                        "description": "How to aggregate scores in region"
                    }
                },
                "required": ["region"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_shape",
            "description": "Analyze the shape of the largest anomaly region. Use to determine WHAT TYPE of defect (crack vs hole).",
            "parameters": {
                "type": "object",
                "properties": {
                    "z_threshold": {
                        "type": "number",
                        "default": 2.0,
                        "description": "Z-score threshold for anomaly detection (must use z-score space, not raw scores)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_symmetric_regions",
            "description": "Compare anomaly scores in symmetric halves. Use to determine if defect is one-sided (localized) or symmetric (material issue).",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["vertical", "horizontal"],
                        "description": "Which axis to compare"
                    }
                },
                "required": ["axis"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scale_profile",
            "description": "Get anomaly scores at different feature scales. Use to determine if defect is SURFACE (scratch/stain) or DEEP (crack/deformation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "default": "full",
                        "description": "Region to analyze"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar_cases",
            "description": "Retrieve similar past defect cases from database. Use to check if this defect matches known patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of similar cases to retrieve"
                    }
                }
            }
        }
    }
]
