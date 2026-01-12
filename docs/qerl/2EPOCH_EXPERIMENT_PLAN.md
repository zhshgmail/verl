# 2-Epoch Experiment Plan

**Date**: 2026-01-12
**Purpose**: Re-run key experiments with 2 epochs to study longer training effects
**Server**: A100 (90.90.102.18)

---

## Experiments to Run

| Priority | Exp ID | Original Name | Original Score | New Name | Config |
|----------|--------|---------------|----------------|----------|--------|
| 1 | E6b-2ep | LoRA_MXFP4_DAPO_1ep_AQN_67.48 | 67.48% | LoRA_MXFP4_DAPO_2ep_AQN | MXFP4 + LoRA + AQN |
| 2 | E6a-2ep | LoRA_MXFP4_DAPO_1ep_65.88 | 65.88% | LoRA_MXFP4_DAPO_2ep | MXFP4 + LoRA |
| 3 | E7a-2ep | LoRA_BF16_DAPO_1ep_71.27 | 71.27% | LoRA_BF16_DAPO_2ep | BF16 + LoRA |
| 4 | E3a-2ep | Q_MXFP4_DAPO_fullFT_73.77 | 73.77% | Q_MXFP4_DAPO_fullFT_2ep | MXFP4 + Full FT |
| 5 | E3b-2ep | Q_MXFP4_DAPO_fullFT_AQN_74.37 | 74.37% | Q_MXFP4_DAPO_fullFT_AQN_2ep | MXFP4 + Full FT + AQN |
| 6 | E8a-2ep | Q_BF16_DAPO_fullFT_1ep_74.75 | 74.75% | Q_BF16_DAPO_fullFT_2ep | BF16 + Full FT |
| 7 | E12-2ep | LoRA_MXFP4_DAPO_1ep_AQN-high_72.48 | 72.48% | LoRA_MXFP4_DAPO_2ep_AQN-high | MXFP4 + LoRA + AQN-high |

---

## Key Configuration Change

```yaml
# Change from 1 epoch to 2 epochs
trainer.total_epochs: 2  # was 1
trainer.test_freq: 10    # was 20 - more frequent evaluation

# Expected step counts:
# - LoRA experiments: ~58 steps (was ~29)
# - Full FT experiments: ~58 steps (was ~29)
# - Evaluations: ~6 per experiment (steps 0, 10, 20, 30, 40, 50, 58)
```

---

## Scripts to Create/Modify

| Script | Based On | Changes |
|--------|----------|---------|
| `test_mxfp4_v6.0_dapo_lora_2ep.sh` | `test_mxfp4_v6.0_dapo_lora.sh` | total_epochs=2 |
| `test_mxfp4_v6.1_dapo_lora_aqn_2ep.sh` | `test_mxfp4_v6.1_dapo_lora_aqn.sh` | total_epochs=2 |
| `test_bf16_v7.0_dapo_lora_2ep.sh` | `test_bf16_v7.0_dapo_lora.sh` | total_epochs=2 |
| `test_mxfp4_v3.0_dapo_2ep.sh` | `test_mxfp4_v3.0_dapo.sh` | total_epochs=2 |
| `test_mxfp4_v3.1_dapo_aqn_2ep.sh` | `test_mxfp4_v3.1_dapo_aqn.sh` | total_epochs=2 |
| `test_bf16_v8.0_dapo_fullft_2ep.sh` | `test_bf16_v8.0_dapo_fullft.sh` | total_epochs=2 |
| `test_mxfp4_v6.2_dapo_lora_aqn_high_sigma_2ep.sh` | `test_mxfp4_v6.2_dapo_lora_aqn_high_sigma.sh` | total_epochs=2 |

---

## Execution Order

Run sequentially on A100 server (each ~2-3 hours):

```bash
# 1. LoRA experiments first (faster, smaller memory)
bash scripts/test_mxfp4_v6.1_dapo_lora_aqn_2ep.sh 8   # E6b-2ep
bash scripts/test_mxfp4_v6.0_dapo_lora_2ep.sh 8       # E6a-2ep
bash scripts/test_bf16_v7.0_dapo_lora_2ep.sh 8        # E7a-2ep

# 2. Full FT experiments (larger memory)
bash scripts/test_mxfp4_v3.0_dapo_2ep.sh 8            # E3a-2ep
bash scripts/test_mxfp4_v3.1_dapo_aqn_2ep.sh 8        # E3b-2ep
bash scripts/test_bf16_v8.0_dapo_fullft_2ep.sh 8      # E8a-2ep
```

---

## Expected Outcomes

### Questions to Answer

1. **Does 2 epochs improve accuracy?**
   - 1ep results may be undertrained
   - 2ep may show continued improvement or reward hacking

2. **Does DAPO prevent reward hacking at 2 epochs?**
   - Previous PPO experiments showed length explosion at epoch 2
   - DAPO's overlong penalty should prevent this

3. **AQN benefit at 2 epochs?**
   - Does AQN provide more benefit with longer training?
   - Does AQN prevent overfitting?

### Metrics to Track

- `val-core/openai/gsm8k/acc/mean` - Primary accuracy metric
- `response_length/mean` - Watch for length explosion
- `actor/entropy` - Watch for entropy collapse
- `critic/reward/mean` - Training reward progression

---

## Status Tracking

| Exp ID | Script | Status | Start Time | End Time | Result |
|--------|--------|--------|------------|----------|--------|
| E6b-2ep | `test_mxfp4_v6.1_dapo_lora_aqn_2ep.sh` | ✅ Complete | Jan 11 11:32 | Jan 11 14:01 | **73.24%** (+5.76%) |
| E6a-2ep | `test_mxfp4_v6.0_dapo_lora_2ep.sh` | ✅ Complete | Jan 11 14:01 | Jan 11 16:30 | **72.93%** (+7.05%) |
| E7a-2ep | `test_bf16_v7.0_dapo_lora_2ep.sh` | ⚠️ Ended Early | Jan 11 16:30 | Jan 11 18:01 | 73.84% @step40 (69%) |
| E3a-2ep | `test_mxfp4_v3.0_dapo_2ep.sh` | ✅ Complete | Jan 11 18:01 | Jan 11 19:09 | **72.78%** (-0.99%) |
| E3b-2ep | `test_mxfp4_v3.1_dapo_aqn_2ep.sh` | ✅ Complete | Jan 11 19:09 | Jan 11 20:15 | **70.05%** (-4.32%, dropped) |
| E8a-2ep | `test_bf16_v8.0_dapo_fullft_2ep.sh` | ⚠️ Ended Early | Jan 11 20:15 | Jan 11 20:54 | **75.97%** @step40 (+1.22%) |
| E12-2ep | `test_mxfp4_v6.2_dapo_lora_aqn_high_sigma_2ep.sh` | ⏳ Pending | - | - | Run in v3 batch |

**v2 batch**: `run_all_2ep_experiments_v2.sh` - COMPLETED
**Master log**: `/tmp/2ep_experiments_master/master.log`
**Next**: Run v3 script for E12-2ep and reruns (E7a-2ep, E8a-2ep)

---

## Notes

- All experiments use DAPO algorithm (prevents reward hacking)
- AQN uses standard sigma (0.01→0.0001) unless noted
- LoRA: rank=32, alpha=16
- Full FT: All parameters trainable
- Server: A100 8x80GB (90.90.102.18, container: verl-r3-test)
