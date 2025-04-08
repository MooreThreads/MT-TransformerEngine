#!/bin/bash

TP_SIZE=8

WARMUP_ITERS=10
TIMING_ITERS=20

# default
#BATCH_SIZE=2
#SEQ_LENGTH=512
#NUM_HEADS=12
#HEAD_SIZE=64

# set engine of musaMemcpyAsync, 1 for DMA, 2 for TDM, 3 for CE
# export MUSA_MEMCPY_PATH=2

# llama3 70B
BATCH_SIZE=1
SEQ_LENGTH=8192
NUM_HEADS=64
HEAD_SIZE=128

# llama3 405B
# BATCH_SIZE=1
# SEQ_LENGTH=8192
# NUM_HEADS=128
# HEAD_SIZE=128

# bulk overlap rs
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --bulk-overlap --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --check-numerics"

# bulk overlap rs over ce
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --bulk-overlap --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --use-ce --check-numerics"

# bulk overlap ag
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type ag --bulk-overlap --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --check-numerics"

# bulk overlap ag over ce
# export MUSA_MEMCPY_PATH=3  # set engine of musaMemcpyAsync is CE for bulk ag
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type ag --bulk-overlap --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --use-ce --check-numerics"

# pipline overlap rs
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS  --check-numerics"

# pipline overlap rs over ce
cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --use-ce --check-numerics"

# ring_exchange overlap rs
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --p2p --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --check-numerics"

# ring_exchange overlap rs over ce
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --p2p --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --use-ce --check-numerics"

# ring_exchange overlap ag
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type ag --p2p --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --check-numerics"

# ring_exchange overlap ag over ce
# cmd="torchrun --nproc-per-node=$TP_SIZE tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type ag --p2p --verbose --dtype bf16 --batch-size $BATCH_SIZE --seq-length $SEQ_LENGTH --num-heads $NUM_HEADS --head-dim $HEAD_SIZE --warmup-iters $WARMUP_ITERS --timing-iters $TIMING_ITERS --use-ce --check-numerics"


echo $cmd
eval $cmd
