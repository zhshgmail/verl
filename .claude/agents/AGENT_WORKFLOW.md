# Agent Workflow Protocol

**Purpose**: Define the multi-agent research and verification workflow for complex technical investigations.
**Status**: PERMANENT - Do NOT remove during markdown cleanup

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PLANNER (Main Claude)                        │
│  - Receives user request                                        │
│  - Decomposes into research tasks                               │
│  - Coordinates expert agents                                    │
│  - Synthesizes findings                                         │
│  - Proposes conclusions/solutions                               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Expert Agent 1│ │ Expert Agent 2│ │ Expert Agent 3│ │ Expert Agent N│
│ (Explore)     │ │ (Explore)     │ │ (general)     │ │ (...)         │
│               │ │               │ │               │ │               │
│ Specific      │ │ Specific      │ │ Specific      │ │ Specific      │
│ research      │ │ research      │ │ research      │ │ research      │
│ domain        │ │ domain        │ │ domain        │ │ domain        │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │               │               │               │
        └───────────────┼───────────────┴───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PLANNER (Main Claude)                        │
│  - Collects all expert findings                                 │
│  - Creates synthesis report                                     │
│  - Draws conclusions                                            │
│  - Proposes solutions/plans                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QA AGENT                                   │
│  - Validates findings                                           │
│  - Checks for inconsistencies                                   │
│  - Identifies low-confidence claims                             │
│  - Verifies evidence supports conclusions                       │
│  - Reports gaps and issues                                      │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                                 │
│  - QA-verified findings                                         │
│  - Confidence levels noted                                      │
│  - Action items/solutions                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Types and Roles

### 1. Planner (Main Claude Session)
**Role**: Orchestrator and synthesizer
**Responsibilities**:
- Decompose complex requests into research tasks
- Assign tasks to appropriate expert agents
- Run agents in parallel when tasks are independent
- Collect and synthesize findings
- Draw conclusions and propose solutions
- Submit output to QA agent for verification

### 2. Expert Agents (Subagents)
**Role**: Specialized research
**Types**:
- `Explore` - Fast codebase exploration (quick/medium/very thorough)
- `general-purpose` - Complex multi-step research tasks
- `Plan` - Architecture and implementation planning

**Responsibilities**:
- Deep dive into assigned domain
- Search code, read files, analyze patterns
- Report findings with file paths and line numbers
- Note uncertainty or gaps in research

### 3. QA Agent (Verification)
**Role**: Quality assurance and validation
**Responsibilities**:
- Review planner's synthesis for accuracy
- Verify claims have supporting evidence
- Identify inconsistencies between expert findings
- Flag low-confidence conclusions
- Check for missing verification steps

---

## QA Agent Rules and Restrictions

### MANDATORY Checks

The QA agent MUST verify:

1. **Evidence Support**
   - Every claim must cite specific file paths and line numbers
   - "Works" claims must have actual test evidence or code proof
   - "Broken" claims must show specific failure scenarios

2. **Consistency Check**
   - No contradictions between different expert findings
   - Conclusions must follow logically from evidence
   - If experts disagree, flag for resolution

3. **Confidence Assessment**
   - Mark claims as: VERIFIED / LIKELY / UNCERTAIN / UNVERIFIED
   - Explain basis for confidence level
   - Low confidence claims must be highlighted

4. **Completeness Check**
   - All original questions answered
   - No critical aspects overlooked
   - Edge cases considered

### QA Report Format

```markdown
## QA Verification Report

### Overall Assessment: [PASS/PARTIAL/FAIL]

### Verified Claims
| Claim | Evidence | Confidence |
|-------|----------|------------|
| ...   | file:line| VERIFIED   |

### Unverified Claims
| Claim | Issue | Risk Level |
|-------|-------|------------|
| ...   | ...   | HIGH/MED/LOW |

### Inconsistencies Found
- [List any contradictions]

### Missing Verifications
- [List claims needing actual testing]

### Recommendations
- [Next steps to increase confidence]
```

### QA Agent Restrictions

**MUST NOT**:
- Blindly accept planner conclusions
- Skip checking cited evidence
- Ignore contradictions
- Mark unverified claims as verified
- Assume code works without proof

**MUST**:
- Read cited files to verify claims
- Cross-check between expert reports
- Question assumptions
- Report uncertainty honestly
- Suggest verification steps for unproven claims

---

## When to Use This Workflow

### Use Multi-Agent + QA For:
- Complex compatibility investigations (multiple codebases)
- Architecture analysis spanning 3+ systems
- Migration/porting feasibility studies
- Root cause analysis of cross-system issues
- Any research where wrong conclusions are costly

### Skip QA For:
- Simple code searches
- Single-file investigations
- Straightforward bug fixes
- Well-documented features

---

## Example Workflow

### User Request:
> "Check if VERL's vLLM eager mode dependencies work with vllm-ascend"

### Planner Actions:
1. Launch Expert Agent 1: "Study VERL vLLM eager mode dependencies"
2. Launch Expert Agent 2: "Study vllm-ascend override patterns"
3. Collect findings from both agents
4. Synthesize into compatibility assessment
5. Launch QA Agent to verify findings

### QA Agent Verification:
```markdown
## QA Verification Report

### Overall Assessment: PARTIAL

### Verified Claims
| Claim | Evidence | Confidence |
|-------|----------|------------|
| enforce_eager defaults to True | rollout.py:143 | VERIFIED |
| vllm-ascend patches weight_loader | vllm_ascend/patch/worker/ | VERIFIED |

### Unverified Claims
| Claim | Issue | Risk Level |
|-------|-------|------------|
| "static_forward_context unavailable in graph mode" | No actual test | HIGH |
| "KV cache init conflicts" | Theoretical, not tested | HIGH |

### Recommendations
1. Run actual test on Ascend hardware to verify KV cache behavior
2. Check vllm-ascend graph mode documentation
```

---

## File Naming Conventions

### Agent Input Files (NEVER delete)
```
.claude/agents/           # Agent definitions
.claude/commands/         # Slash commands
_AGENT_*.md              # Prefixed with _AGENT_
*.AGENT_INPUT.md         # Suffixed with .AGENT_INPUT
```

### Temporary Files (Delete after task)
```
TEMP_DO_NOT_COMMIT_*     # In .gitignore
```

### Output Files (Keep for reference)
```
docs/*_Baseline.md       # Investigation baselines
docs/*_Analysis.md       # Analysis reports
```

---

## Integration with CLAUDE.md

Add to project CLAUDE.md when using this workflow:

```markdown
## Agent Workflow

For complex investigations, use the multi-agent workflow:
1. See `.claude/agents/AGENT_WORKFLOW.md` for protocol
2. Always run QA agent for high-stakes conclusions
3. Mark confidence levels on all findings
```
