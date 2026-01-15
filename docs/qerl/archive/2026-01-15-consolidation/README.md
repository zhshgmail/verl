# Documentation Consolidation Archive - 2026-01-15

## Purpose

This archive contains documentation files that were moved during the docs/qerl consolidation on 2026-01-15. These files contained outdated plans, interim status, session notes, or information that has been superseded by consolidated documents.

## Archived Files (27 files)

### Completed Planning Documents
- `2EPOCH_EXPERIMENT_PLAN.md` - Planning for 2-epoch experiments (completed)
- `AQN_VALIDATION_EXPERIMENT_PLAN.md` - AQN validation planning (completed)
- `E12_AGGRESSIVE_EXPERIMENT_PLAN.md` - E12 experiment planning (completed)
- `W4A4_MXFP4_LORA_AQN_EXPERIMENT_PLAN.md` - E13 W4A4 planning (completed, E13g/h done)
- `QERL_STYLE_EXPERIMENTS.md` - QeRL reproduction planning (completed)

### Session Notes & Handoffs
- `SESSION_LOWER_AQN_SRDD_GUIDED.md` - Session notes for lower sigma + RIN experiments
- `SRDD_COLIBRARY_SESSION_HANDOFF.md` - Session handoff notes
- `SRDD_COLIBRARY_VALIDATION.md` - Validation session notes

### Interim Analysis & Diagnostics
- `MXFP4_AQN_NEXT_STEPS.md` - Interim next steps (now in consolidated docs)
- `NOISE_INJECTION_DIAGNOSTIC.md` - Diagnostic notes (info in HW_ERROR_INJECTION_EXPERIMENTS.md)

### Redundant/Superseded Experiment Logs
- `LORA_EXPERIMENT_RESULTS_20260111.md` - LoRA experiments (consolidated into ALL_EXPERIMENTS_SUMMARY.md)
- `SRDD_MXFP4_QUANT_EXPERIMENT.md` - Specific SRDD experiment (info in MXFP4_NVFP4_EXPERIMENT_REGISTRY.md)

### Chinese Documentation
- `AQN_EXPERIMENT_SUMMARY_CN.md` - Chinese AQN summary (English version kept)
- `SRDD_GUIDED_AQN_PROPOSAL_CN.md` - Chinese RIN proposal
- `SRDD_TECH_REPORT_CN.md` - Chinese SRDD technical report
- `SRDD_USE_CASE_CN.md` - Chinese SRDD use cases
- `npu_training_guide_zh.md` - Chinese NPU training guide

### SRDD Implementation Details
- `SRDD_GUIDED_AQN_EXPERIMENT_DESIGN.md` - RIN implementation design (now completed, RIN terminology updated in kept docs)

### NPU-Specific Documentation
- `Ascend_910C_Baseline.md` - Ascend NPU baseline experiments
- `NPU_910C_Setup_TestPlan.md` - NPU setup and test planning
- `NPU_ARCHITECTURE_CORRECTION.md` - NPU architecture notes
- `MEGATRON_NPU_SETUP.md` - Megatron-LM NPU setup

### Analysis & Research Documents
- `QA_vLLM_Eager_Mode_Ascend_Compatibility.md` - vLLM Ascend compatibility analysis
- `QA_verl_npu_Branch_Analysis.md` - verl NPU branch analysis
- `QERL_REPRODUCTION_ANALYSIS.md` - QeRL paper reproduction analysis
- `QeRL_Quantization_Research.md` - Quantization research notes
- `R3_VLLM_NPU_COMPATIBILITY_ANALYSIS.md` - R3 vLLM NPU compatibility

## Active Documentation (Kept in docs/qerl/)

The following 8 files remain as the core documentation:

### Experiments & Results (3 files)
1. **ALL_EXPERIMENTS_SUMMARY.md** - Master summary of all experiments (E1-E13+)
2. **HW_ERROR_INJECTION_EXPERIMENTS.md** - HW error injection experiments (E5, E9 series with RIN)
3. **MXFP4_NVFP4_EXPERIMENT_REGISTRY.md** - Quantization experiments (E3, E4, E6, E8 series)

### Infrastructure & Workflow (2 files)
4. **A100_CONTAINER_AND_DEV_WORKFLOW.md** - A100 access, git workflow, dev environment
5. **WANDB_UPLOAD_GUIDE.md** - Training progress monitoring with WandB

### Technical Documentation (3 files)
6. **AQN_CONSOLIDATED_FINDINGS.md** - Consolidated AQN/RIN findings with expert review
7. **E13_W4A4_EXPERIMENT_LOG.md** - W4A4 experiments (E13 series) with STE fix discovery
8. **NPU_LESSONS_LEARNED.md** - NPU/Ascend hardware lessons learned

## How to Access Archived Information

If you need information from archived files:

1. **Experiment Results**: Check `ALL_EXPERIMENTS_SUMMARY.md` first - it contains consolidated results
2. **Planning/Status**: Archived planning docs show historical context but may be outdated
3. **Chinese Documentation**: English versions kept in active docs
4. **SRDD/RIN Implementation**: Check `AQN_CONSOLIDATED_FINDINGS.md` for current methodology

## Restoration

If you need to restore any archived file:
```bash
cd /home/zheng/workspace/verl/docs/qerl
git mv archive/2026-01-15-consolidation/<filename> .
```
