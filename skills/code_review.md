# Skill: code_review

## Purpose
Pre-commit checklist for every module before merging to dev.

## Checklist
- [ ] Handles missing or malformed input without crashing
- [ ] LLM output validated with Pydantic before use
- [ ] Exclusion criteria treated as hard eliminators (no LLM override)
- [ ] Every NEI verdict has a generated clinical question
- [ ] Output is deterministic and reproducible (seed / temperature=0)
- [ ] No LLM call for age, gender, numeric, or date operations
- [ ] CLAUDE.md updated if any new convention discovered
- [ ] Unit test or validation script exists in /tests
- [ ] No broken code on main branch
