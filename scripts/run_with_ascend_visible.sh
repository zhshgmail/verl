#!/usr/bin/env bash
set -euo pipefail

# Ray 有时不会给 driver/worker 设置 CUDA_VISIBLE_DEVICES（尤其是 job driver 进程）
# Ascend/torch_npu 关键是 ASCEND_RT_VISIBLE_DEVICES 不能是空字符串/未设置

# 1) 如果 Ray 给了 CUDA_VISIBLE_DEVICES，就映射到 Ascend
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  export ASCEND_RT_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  export NPU_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

# 2) 如果 Ascend 可见变量仍为空（None 或 ""），就兜底让它看到全部卡（先跑起来优先）
#    你现在机器是 16 卡；后面稳定后再改回更严格的隔离策略
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  export NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
fi

# 打到 stderr，方便 ray job logs 里确认
echo "[ascend-visible] CVD=${CUDA_VISIBLE_DEVICES:-<none>} ARVD=${ASCEND_RT_VISIBLE_DEVICES:-<none>} AVD=${ASCEND_VISIBLE_DEVICES:-<none>}" >&2

exec "$@"
