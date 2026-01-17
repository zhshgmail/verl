# verl val_only Mode Bug - Handover Document

## Problem Statement
E13j checkpoint (trained with MXFP4 W4A4) needs proper evaluation WITH MXFP4 fake quantization. Previous evaluation scripts did NOT apply MXFP4, giving inflated BF16 scores (73.31%). We need true W4A4 accuracy.

## Bug Discovery
When using `trainer.val_only=True` to evaluate checkpoints:
- Expected accuracy: ~70%+ (trained model)
- Actual accuracy: 8.49% (same as untrained base model ~6%)
- Both BF16 and MXFP4 modes give identical 8.49% → proves weights not loading

## Root Cause (CONFIRMED)
**File:** `/home/zheng/workspace/verl/recipe/dapo/dapo_ray_trainer.py` lines 98-109

**Execution order bug:**
1. `init_workers()` → vLLM initialized with BASE MODEL weights
2. `_load_checkpoint()` → Weights loaded into FSDP only (NOT synced to vLLM)
3. `_validate()` → vLLM still has BASE MODEL weights!

The checkpoint weights load into FSDP but are NEVER synced to vLLM before validation.

## Proposed Fix
Add explicit weight sync after checkpoint loading in `dapo_ray_trainer.py`:

```python
def fit(self):
    self._load_checkpoint()

    # FIX: Sync checkpoint weights to vLLM before validation
    if self.config.trainer.get("val_only", False):
        loop = get_event_loop()
        loop.run_until_complete(self.actor_rollout_wg.rollout_mode())
        loop.run_until_complete(self.actor_rollout_wg.trainer_mode())

    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        # ...
```

## Key Files
- `/home/zheng/workspace/verl/recipe/dapo/dapo_ray_trainer.py` - FIX GOES HERE (lines 98-109)
- `/home/zheng/workspace/verl/verl/workers/fsdp_workers.py` - rollout_mode() syncs weights (line 681-763)
- `/home/zheng/workspace/verl/verl/trainer/ppo/ray_trainer.py` - _validate() and _load_checkpoint()
- `/home/zheng/workspace/verl/scripts/eval_checkpoint_verl.sh` - Evaluation script that triggers bug

## Test Commands
```bash
# On A100 server (ssh root@90.90.102.18, docker exec verl-r3-test bash)
cd /home/z00637938/workspace/verl

# Test val_only mode after fix
bash scripts/eval_checkpoint_verl.sh /tmp/mxfp4_w4a4_e13j_global_aqn/checkpoints/global_step_29 e13j_test 8
```

## Expected Results After Fix
- val_only with MXFP4: Should be ~50-70% (true W4A4 accuracy)
- val_only with BF16: Should be ~70%+ (matches previous inflated score)

## Evidence Collected
| Test | Accuracy | Meaning |
|------|----------|---------|
| Base model (no LoRA) | 6.0% | Baseline |
| verl val_only (broken) | 8.49% | ≈ base model, bug confirmed |
| Previous BF16 eval | 73.31% | Inflated, no MXFP4 |

## Concerns
1. Fix might have side effects on normal training flow - test training after fix
2. Async rollout mode may have different code path - check if fix covers both sync/async
3. The `rollout_mode()` → `trainer_mode()` cycle might be expensive - consider caching

## Lessons Learned
1. Always test base model as baseline first
2. Identical results across different configs (BF16 vs MXFP4) = weights not loading
3. Don't trust evaluation results without verification
4. verl's val_only mode was not well-tested for checkpoint evaluation

## Checkpoint Paths
- E13j: `/tmp/mxfp4_w4a4_e13j_global_aqn/checkpoints/global_step_29/actor/lora_adapter`
- Base model: `/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Test data: `/data/z00637938/gsm8k/test.parquet`

## Next Steps
1. Apply the fix to `dapo_ray_trainer.py`
2. Commit and push to GitHub
3. Pull on A100 server container
4. Test val_only mode with E13j checkpoint
5. Verify accuracy is now ~50-70% (not 8.49%)
6. If working, evaluate all E13 checkpoints (E13j, E13k, E13l, E13m, E13n) with MXFP4

## Git Workflow
```bash
# Local: make changes, commit, push
git add recipe/dapo/dapo_ray_trainer.py
git commit -m "fix: Sync checkpoint weights to vLLM in val_only mode"
git push personal feature/npu-aqn-test

# Server: pull and test
ssh root@90.90.102.18
docker exec verl-r3-test bash -c 'source /home/z00637938/setup_proxy.sh && cd /home/z00637938/workspace/verl && git fetch personal && git reset --hard personal/feature/npu-aqn-test'
```
