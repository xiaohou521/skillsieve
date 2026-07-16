import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

ATTACK_TYPES = [
    "prompt_injection", "credential_theft", "remote_exec",
    "data_exfil", "typosquatting", "obfuscation", "social_engineering",
]


@dataclass
class Annotation:
    skill_name: str
    label: str  # "benign" or "malicious"
    attack_types: list[str] = field(default_factory=list)
    stealth_rating: int = 1  # 1-5
    annotator: str = ""
    notes: str = ""
    timestamp: str = ""


class AnnotationTool:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.annotations: list[Annotation] = []
        if self.output_path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.output_path.read_text())
        self.annotations = [Annotation(**a) for a in data]

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps([asdict(a) for a in self.annotations], indent=2)
        )

    def add_annotation(self, skill_name: str, label: str, attack_types: list[str] = None,
                       stealth_rating: int = 1, annotator: str = "", notes: str = "") -> Annotation:
        if label not in ("benign", "malicious"):
            raise ValueError(f"label must be 'benign' or 'malicious', got '{label}'")
        if attack_types:
            for t in attack_types:
                if t not in ATTACK_TYPES:
                    raise ValueError(f"Unknown attack type: {t}")
        if not 1 <= stealth_rating <= 5:
            raise ValueError(f"stealth_rating must be 1-5, got {stealth_rating}")

        ann = Annotation(
            skill_name=skill_name, label=label,
            attack_types=attack_types or [], stealth_rating=stealth_rating,
            annotator=annotator, notes=notes,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.annotations.append(ann)
        return ann

    def get_stats(self) -> dict:
        total = len(self.annotations)
        benign = sum(1 for a in self.annotations if a.label == "benign")
        malicious = total - benign
        return {"total": total, "benign": benign, "malicious": malicious}

    def get_by_label(self, label: str) -> list[Annotation]:
        return [a for a in self.annotations if a.label == label]
