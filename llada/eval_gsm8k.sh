# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# # baseline
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True 


# # prefix cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True 


# # parallel
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True


# # prefix cache+parallel
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True

# Benchmark optimal block_length and steps (dual cache+parallel)
task=gsm8k
length=256
num_fewshot=5

logs_dir="eval_${task}_${num_fewshot}shot_length${length}_dualcache_parallel_logs"

rm -rf ${logs_dir}
mkdir -p ${logs_dir}

echo "Benchmarking settings:"
echo "- Task: ${task}"
echo "- Few-shot: ${num_fewshot}"
echo "- Generation length: ${length}"
echo "Logs saved to: ${logs_dir}"
echo "================================"
echo "Benchmarking variable block_length with confidence-parallel enabled (each block set steps=1)"
block_length=4
while [ $block_length -le $length ]; do
    steps=$((length / block_length))
    echo "Evaluating block_length=${block_length}, steps=${steps}"
    logfile="${logs_dir}/steps${steps}_block${block_length}_th0.9.log"
    
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
        2>&1 | tee ${logfile}
    
    echo "Results saved to: ${logfile}"
    echo "--------------------------------"
    block_length=$((block_length * 2))
done

echo "================================"
echo "Benchmarking variable threshold with fixed block_length=32"
block_length=32
steps=$((length / block_length))
thresholds=("0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
for threshold in "${thresholds[@]}"
do
    echo "Evaluating block_length=${block_length}, threshold=${threshold}"
    logfile="${logs_dir}/steps${steps}_block${block_length}_th${threshold}.log"
    
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=${threshold},show_speed=True \
        2>&1 | tee ${logfile}
    
    echo "Results saved to: ${logfile}"
    echo "--------------------------------"
done