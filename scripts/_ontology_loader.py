from __future__ import annotations

from pathlib import Path


def read_ontology(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        pass

    ontology = {"entity_types": {}, "allowed_relations": {}}
    section = ""
    current_entity = ""
    current_relation = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            section = line.rstrip(":")
            current_entity = ""
            current_relation = ""
            continue
        if section == "entity_types":
            if line.startswith("  ") and not line.startswith("    "):
                current_entity = line.strip().rstrip(":")
                ontology["entity_types"][current_entity] = {}
            elif current_entity and line.strip().startswith("layer:"):
                ontology["entity_types"][current_entity]["layer"] = (
                    line.split(":", 1)[1].strip()
                )
        elif section == "allowed_relations":
            if line.startswith("  ") and not line.startswith("    "):
                current_relation = line.strip().rstrip(":")
                ontology["allowed_relations"][current_relation] = []
            elif current_relation and line.strip().startswith("- ["):
                pair = line.strip().removeprefix("- [").removesuffix("]")
                ontology["allowed_relations"][current_relation].append(
                    [item.strip() for item in pair.split(",")]
                )
    return ontology
