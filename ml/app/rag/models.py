from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

_SEVERITIES = {"high", "moderate", "low"}


class Evidence(BaseModel):
    citation: str
    url: str


class Monograph(BaseModel):
    id: str
    drug_a: str
    drug_b: str
    drug_a_aliases: list[str] = Field(default_factory=list)
    drug_b_aliases: list[str] = Field(default_factory=list)
    drug_class_a: str
    drug_class_b: str
    severity: str
    sections: dict[str, str]
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("severity")
    @classmethod
    def _severity_ok(cls, v: str) -> str:
        if v not in _SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_SEVERITIES)}")
        return v

    @field_validator("sections")
    @classmethod
    def _sections_nonempty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("sections must not be empty")
        return v

    def all_drug_names(self) -> list[str]:
        names = [self.drug_a, self.drug_b, *self.drug_a_aliases, *self.drug_b_aliases]
        seen: list[str] = []
        for n in names:
            low = n.lower().strip()
            if low and low not in seen:
                seen.append(low)
        return seen


class Chunk(BaseModel):
    text: str
    source_doc_id: str
    section: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)
