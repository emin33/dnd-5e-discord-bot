"""Read models for the canonical sourcebook layer.

``sourcebook.py`` is the authoring contract — what a book IS. These are what
queries against the imported book RETURN: a claim with its per-campaign
overlay already resolved, a faction roster, an authored tie in its full
24-kind vocabulary rather than the graph's collapsed nine.

They are deliberately separate from the authoring models. A ``KnowledgeClaim``
is immutable canon; a :class:`CampaignClaim` is that claim *as this party has
experienced it so far*, which is a different thing and must not be mistaken
for the book.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .sourcebook import CanonStatus, KnowledgeClaim, RelationshipKind, Visibility


class SourcebookHeader(BaseModel):
    """An imported book version, without its contents."""

    sourcebook_key: str
    sourcebook_id: str
    schema_version: str = "1.0"
    title: str = ""
    pitch: str = ""
    ruleset: str = "dnd5e"
    imported_at: str = ""


class BindReceipt(BaseModel):
    """What binding a campaign to a book version actually moved.

    ``superseded_keys`` is non-empty when the campaign was already playing a
    different version of the SAME book — advancing to a new version is not a
    silent operation, because it retires the old binding and carries the
    party's overlay across.
    """

    campaign_id: str
    sourcebook_key: str
    superseded_keys: list[str] = Field(default_factory=list)
    claims_carried: int = 0
    visits_carried: int = 0


class CampaignClaim(BaseModel):
    """Authored canon plus this campaign's overlay on it.

    ``claim`` is the book, untouched. Everything beside it is play.
    """

    sourcebook_key: str
    claim: KnowledgeClaim
    discovered: bool = False
    discovered_at_turn: int | None = None
    discovered_via: str = ""
    # Set by play when a later claim overturns this one. Resolution order is
    # campaign overlay OVER book, so a campaign can supersede authored canon
    # without the immutable book changing.
    superseded_by_claim_id: str | None = None
    # The book's status unless the campaign overrode it.
    effective_canon_status: CanonStatus = CanonStatus.CANON
    note: str = ""

    @property
    def claim_id(self) -> str:
        return str(self.claim.id)

    @property
    def text(self) -> str:
        return self.claim.text

    @property
    def visibility(self) -> Visibility:
        return self.claim.visibility

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by_claim_id is not None


class FactionMember(BaseModel):
    """One row of a faction roster."""

    faction_id: str
    npc_id: str
    npc_name: str
    membership_role: str = "member"
    status: str = "alive"


class AuthoredTie(BaseModel):
    """A relationship as the BOOK wrote it, not as the graph indexed it.

    The graph maps ``rival_of``, ``fears`` and ``hostile_to`` all onto one
    ``hostile_to`` edge. Answering "who is hostile to X" from here keeps the
    distinction the author made.
    """

    relationship_id: str
    source_id: str
    source_name: str = ""
    target_id: str
    target_name: str = ""
    kind: RelationshipKind
    custom_kind: str | None = None
    directed: bool = True
    valence: int | None = None
    public_description: str = ""
    # DM-side. Present because this layer is the system of record; the
    # compiler is what refuses to project a tie that has only this.
    private_description: str = ""


class RegionContents(BaseModel):
    """Everything authored inside a location subtree.

    ``location_ids`` includes the region itself and every descendant, so the
    other lists are "authored anywhere in here", not "authored at this exact
    node".

    ``unvisited_location_ids`` is only populated when the query was given a
    ``campaign_id`` — with no campaign there is no visit state, and the field
    stays empty rather than pretending nothing has been visited.
    """

    region_id: str
    location_ids: list[str] = Field(default_factory=list)
    npc_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    quest_ids: list[str] = Field(default_factory=list)
    faction_ids: list[str] = Field(default_factory=list)
    unvisited_location_ids: list[str] = Field(default_factory=list)

    @property
    def is_untouched(self) -> bool:
        """True when the party has visited nothing in this subtree.

        False for an empty region: "nothing authored here" is not the same
        answer as "authored and unvisited", and conflating them would make
        a typo'd region id read as a whole untouched world.
        """
        return bool(self.location_ids) and (
            len(self.unvisited_location_ids) == len(self.location_ids)
        )


class ImportReceipt(BaseModel):
    """What an import actually wrote — the design doc's projection receipt."""

    sourcebook_key: str
    sourcebook_id: str
    already_imported: bool = False
    row_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts.values())


class RebuildReceipt(BaseModel):
    """What a rebuild regenerated, what it preserved, and what it refused to.

    ``projected_*`` is what canon asked for; ``*_added`` is what the graph
    actually took, MEASURED before and after. The two differ whenever the
    graph merges or abstains on a proper-name collision — which it does
    silently, returning no rejection — so reporting only the projection's own
    counts would overstate the rebuild.
    """

    sourcebook_key: str
    campaign_id: str
    projected_nodes: int = 0
    projected_edges: int = 0
    nodes_added: int = 0
    edges_added: int = 0
    # Node ids left untouched because the graph already had them. These carry
    # play's version of an entity (a death, a renamed NPC, a description
    # written during a scene), which canon must not revert.
    preserved_nodes: list[str] = Field(default_factory=list)
    graph_rejections: list[str] = Field(default_factory=list)
    embedded: int = 0
    embed_failures: int = 0
    vector_skipped: bool = True
    warnings: list[str] = Field(default_factory=list)

    @property
    def vector_complete(self) -> bool:
        """False when the index was attempted and did not fully land."""
        return self.vector_skipped or self.embed_failures == 0
