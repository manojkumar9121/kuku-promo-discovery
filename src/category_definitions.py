#!/usr/bin/env python3
"""Semantic category definitions for Kuku promo discovery.

Each category carries a natural-language description written the way the
category would be *explained to a reader* — not a keyword list. These
descriptions are embedded with EmbeddingGemma 2 and compared against candidate
content by cosine similarity.

Prompt-strategy mapping (EmbeddingGemma 2 model card, "Task Instruction
Prefixes"):
* ASYMMETRIC_SEARCH  — the drama/broad categories act as retrieval queries and
  get ``prompt_name="SearchQuery"`` (``task: search result | query: {text}``);
  candidate content is the corpus side and is embedded as a titled document
  (``title: {title} | text: {content}``).
* SYMMETRIC_SIMILARITY — both sides share ``prompt_name="SentenceSimilarity"``.

Scores are cosine similarities in [-1, 1]. They are NOT probabilities and are
never treated as likelihoods by this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PROMPT_ASYMMETRIC_SEARCH = "ASYMMETRIC_SEARCH"
PROMPT_SYMMETRIC_SIMILARITY = "SYMMETRIC_SIMILARITY"
VALID_PROMPT_STRATEGIES = (PROMPT_ASYMMETRIC_SEARCH, PROMPT_SYMMETRIC_SIMILARITY)


@dataclass(frozen=True)
class CategoryDefinition:
    """One semantic promo category.

    `description` must be rich natural language: it is embedded verbatim as a
    semantic query (or similarity reference, depending on the strategy).
    """

    category_id: str
    name: str
    description: str
    group: str  # "drama" | "structural"
    symmetric: bool = False  # True -> compare symmetric-similarity strategy too


DRAMA_CATEGORIES: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        category_id="betrayal",
        name="Betrayal",
        group="drama",
        description=(
            "A trusted person breaks their promise or loyalty. Someone who loved, "
            "supported or protected the speaker turns out to be working against "
            "them, cheats on them, lies to their face, chooses someone else, or "
            "switches sides at the worst possible moment. The moment exposes the "
            "violation of trust, often with a victim realising the truth while a "
            "confident villain acts without shame."
        ),
    ),
    CategoryDefinition(
        category_id="shocking_reveal",
        name="Shocking reveal",
        group="drama",
        description=(
            "A hidden truth explodes into the open. A secret identity, bloodline, "
            "illicit affair, forged document, crime or conspiracy is uncovered and "
            "the room reacts: gasps, frozen faces, dropped objects, someone "
            "stumbling to leave. What changes know that what seemed solid was a "
            "lie all along. The important thing is the stunned emotional impact "
            "of previously unknown information becoming real within the scene."
        ),
    ),
    CategoryDefinition(
        category_id="revenge",
        name="Revenge",
        group="drama",
        description=(
            "Payback in motion. A character deliberately returns harm or "
            "humiliation inflicted on them: plotting, cold smiles while an enemy "
            "collapses, gloating declarations of debts to be paid, orchestrated "
            "downfalls, contracts or sabotage executed to right a past wrong. The "
            "moment should convey purposeful retribution from someone wronged, "
            "not random cruelty."
        ),
    ),
    CategoryDefinition(
        category_id="humiliation",
        name="Humiliation",
        group="drama",
        description=(
            "A character is stripped of dignity before others. Mocked, insulted, "
            "verbally reduced, laughed at, exposed by an unflattering truth, "
            "thrown out, slapped or defeated in full view of the very people "
            "whose opinion hurts most. The audience should feel the sting: shame, "
            "tears, a public display of power over somebody vulnerable."
        ),
    ),
    CategoryDefinition(
        category_id="threat",
        name="Threat",
        group="drama",
        description=(
            "Overt menace. A character promises harm to another through speech or "
            "action — raised voices, pointed fingers, veiled or direct warnings, "
            "forced removals, violent imagery, conditions that must be obeyed or "
            "consequences will follow. The situation carries a real possibility "
            "of physical, social or emotional destruction aimed at the victim."
        ),
    ),
    CategoryDefinition(
        category_id="confession",
        name="Confession",
        group="drama",
        description=(
            "A raw admission creates a turning point. A character voluntarily "
            "reveals their own fault, feeling or secret: broken confessions of "
            "love spilling out at the wrong time, guilt admitted through tears, "
            "apologies that acknowledge specific harm, whistleblowing against "
            "family or allies. Spontaneity and emotional exposure distinguish "
            "confession from cool strategic revelation."
        ),
    ),
    CategoryDefinition(
        category_id="romantic_tension",
        name="Romantic tension",
        group="drama",
        description=(
            "Romantic charge with something unresolved: a forbidden glance held a "
            "beat too long, an accidental touch, a kiss that lands mid-conflict, "
            "or a love confession almost interrupted. Characters are drawn "
            "together while duty, rivalry, marriage politics or pride pull them "
            "apart. The moment should feel like two people fighting attraction "
            "rather than a settled happy romance."
        ),
    ),
    CategoryDefinition(
        category_id="suspense",
        name="Suspense",
        group="drama",
        description=(
            "Dread of what happens next. Slow footsteps behind a door, prying "
            "eyes and eavesdropping almost exposed, a plan about to be caught, "
            "dark corridors, night vigils, someone waiting for a blow that has "
            "not yet landed. Dialogue and staging keep outcomes uncertain and "
            "danger close; the candidate should leave the viewer bracing rather "
            "than devastated."
        ),
    ),
    CategoryDefinition(
        category_id="unexpected_entrance",
        name="Unexpected entrance",
        group="drama",
        description=(
            "Someone or something arrives who nobody expected: doors slamming "
            "open at a decisive moment, the lost heir surfacing, enemies walking "
            "into a private meeting, a rescuer charging in mid-danger, a voice "
            "interrupted by a figure appearing behind the speaker. The entrance "
            "inverts the power or emotional balance precisely because nobody saw "
            "it coming."
        ),
    ),
    CategoryDefinition(
        category_id="emotional_confrontation",
        name="Emotional confrontation",
        group="drama",
        description=(
            "A raw, personal head-to-head between people bound by family, love "
            "or history. Raised emotive dialogue, tears and fury at full "
            "intensity, accusations flying, old wounds reopened, decisions "
            "forced to a breaking point. Not a physical fight — the collision is "
            "verbal and emotional, and the relationship between the parties is "
            "the battlefield."
        ),
    ),
)

STRUCTURAL_CATEGORIES: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        category_id="strong_opening_hook",
        name="Strong opening hook",
        group="structural",
        description=(
            "A self-contained opening moment that grabs a stranger within "
            "seconds: an arresting visual, a shocking first line, an image of "
            "someone in trouble or in splendour, an immediate question that "
            "demands an answer. A viewer who knows nothing about the story "
            "cannot stop watching once this begins. Works without prior "
            "knowledge of the series or characters."
        ),
    ),
    CategoryDefinition(
        category_id="unresolved_cliffhanger",
        name="Unresolved cliffhanger",
        group="structural",
        description=(
            "A moment that ends mid-tension, dangling an unanswered question: a "
            "proposal or demand with no reply yet, a hand about to strike, a "
            "door opening on something unseen, a vow with no consequence shown "
            "yet. Ideal for cutting away and leaving a viewer desperate to "
            "continue. The resolution must NOT be contained inside the clip."
        ),
    ),
    CategoryDefinition(
        category_id="standalone_dramatic_moment",
        name="Standalone dramatic moment",
        group="structural",
        description=(
            "A complete emotional arc compressed into one window: setup, peak "
            "and payoff inside the clip itself, understandable without context "
            "from other scenes. An insult lands, betrayal revealed and "
            "vowed-avenged, love declared and rejected — all present. Best used "
            "where the promo must feel whole rather than teasing."
        ),
    ),
)

CATEGORIES: tuple[CategoryDefinition, ...] = DRAMA_CATEGORIES + STRUCTURAL_CATEGORIES

CATEGORIES_BY_ID: dict[str, CategoryDefinition] = {category.category_id: category for category in CATEGORIES}


def ordered_descriptions(category_ids: Optional[list[str]] = None) -> list[CategoryDefinition]:
    """Category list honouring an explicit, deduplicated order."""
    if category_ids is None:
        return list(CATEGORIES)
    unknown = [category_id for category_id in category_ids if category_id not in CATEGORIES_BY_ID]
    if unknown:
        raise ValueError(f"Unknown category ids: {unknown}")
    seen: set[str] = set()
    ordered: list[CategoryDefinition] = []
    for category_id in category_ids:
        if category_id in seen:
            continue
        seen.add(category_id)
        ordered.append(CATEGORIES_BY_ID[category_id])
    return ordered
