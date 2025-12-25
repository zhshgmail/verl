search internet or local disk as you need. response as accuracy as possible. don't provide me low confident information, or you need to highlight which part of your response is in low condident.

## Agent Workflow

For complex multi-codebase investigations, use the multi-agent workflow:
1. See `.claude/agents/AGENT_WORKFLOW.md` for protocol
2. Pattern: Planner → Expert Agents (parallel) → Synthesis → QA Agent → Final Output
3. Always run QA agent for high-stakes conclusions
4. Mark confidence levels: VERIFIED / LIKELY / UNCERTAIN / UNVERIFIED

## Remote Host Best Practices

### Use tmux for Long-Running Commands

**IMPORTANT**: Any command expected to run longer than 1 minute on remote hosts (A100, H100, etc.) MUST be run inside a tmux session.

**Why**: Company network connections to NPU/GPU hosts and even Claude Code backend can be interrupted at any time. tmux ensures commands continue running even if the SSH session is disconnected.

**Pattern**:
```bash
# Create a named tmux session for the task
ssh user@host "tmux new-session -d -s task_name 'command_here 2>&1 | tee /path/to/logfile.log'"

# Check progress later
ssh user@host "tmux capture-pane -t task_name -p 2>/dev/null | tail -10"

# Or attach to the session
ssh user@host -t "tmux attach -t task_name"
```

**Examples of commands that MUST use tmux**:
- Model quantization (hours)
- Training runs (hours to days)
- Large file downloads/uploads
- Model conversion scripts
- Any batch processing

**DO NOT** run long commands directly like:
```bash
# BAD - will die if SSH disconnects
ssh user@host "python quantize_model.py"

# GOOD - survives disconnection
ssh user@host "tmux new-session -d -s quant 'python quantize_model.py 2>&1 | tee quant.log'"
```

### Temporary Files Naming Convention

For temporary tracking files that should NOT be committed:
- Use prefix: `TEMP_DO_NOT_COMMIT_`
- These are already in `.gitignore`
- Delete them when the tracked tasks complete

