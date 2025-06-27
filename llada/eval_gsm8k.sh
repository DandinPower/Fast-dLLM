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

# Grid search for optimal block_length and steps (dual cache+parallel)
task=gsm8k
length=256
num_fewshot=5

logs_dir="eval_${task}_${num_fewshot}shot_length${length}_dualcache_parallel_logs"

# Create logs directory
rm -rf ${logs_dir}
mkdir -p ${logs_dir}

echo "Starting grid search for block_length and steps parameters..."
echo "Base length: ${length}"
echo "================================"

# Generate block_length values: 4, 8, 16, 32, 64, 128, ... (stop when >= length)
block_length=4
while [ $block_length -le $length ]; do
    # Calculate base steps (length / block_length)
    base_steps=$((length / block_length))
    
    # Test steps from base_steps to length, doubling each time
    steps=$base_steps
    while [ $steps -le $length ]; do
        echo "Testing: block_length=${block_length}, steps=${steps}"
        
        logfile="${logs_dir}/steps${steps}_block${block_length}_th0.9.log"
        
        accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
        2>&1 | tee ${logfile}
        
        echo "Results saved to: ${logfile}"
        echo "--------------------------------"
        
        # Double the steps for next iteration
        steps=$((steps * 2))
    done
    
    # Double the block_length for next iteration
    block_length=$((block_length * 2))
done

echo "Grid search completed!"
echo "All results saved in ${logs_dir}/ directory"