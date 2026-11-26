# skills

Agent skills from Renaissance Labs. Each folder is one self-contained skill: a `SKILL.md`
with instructions for the agent, plus any scripts it needs.

| Skill | What it does |
|---|---|
| [hf-dataset-search](hf-dataset-search/) | Search metadata for every public dataset on the Hugging Face Hub from a local index: by meaning, by keyword and by structured filters, without downloading any dataset. |

## Install

For Claude Code, link or copy a skill folder into `~/.claude/skills/`:

```bash
git clone https://github.com/RenaissanceLabs-EU/skills.git
ln -s "$PWD/skills/hf-dataset-search" ~/.claude/skills/hf-dataset-search
```

Other agents can use a skill by reading its `SKILL.md`.
