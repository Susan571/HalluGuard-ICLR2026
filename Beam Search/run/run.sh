export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SKIP_MEMORY_PROFILING=True

CUDA_VISIBLE_DEVICES=0 python -m run.run_ttc --dataset math --model io --reward none --postfix 0
CUDA_VISIBLE_DEVICES=1 python -m run.run_ttc --dataset math --model beam_search --reward rm --config beam=2 budget=8 --postfix beam2_budget8
CUDA_VISIBLE_DEVICES=0 python -m run.run_ttc --dataset math --model beam_search --reward perplexity --config beam=2 budget=32 --num-workers 50 --postfix perplexity_beam2_budget32 
CUDA_VISIBLE_DEVICES=1 python -m run.run_ttc --dataset math --model beam_search --reward perplexity --config beam=4 budget=16 --num-workers 100 --postfix perplexity_beam4_budget16 
CUDA_VISIBLE_DEVICES=2 python -m run.run_ttc --dataset math --model beam_search --reward perplexity --config beam=2 budget=8 --num-workers 1 --postfix perplexity_beam2_budget8 
CUDA_VISIBLE_DEVICES=3 python -m run.run_ttc --dataset math --model beam_search --reward perplexity --config beam=4 budget=32 --num-workers 50 --postfix perplexity_beam4_budget32

# ENV_FILE=.env.llama python -m run.run_inf --dataset math --model vanilla --postfix llama-0 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset math --model vanilla --postfix llama-1 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset math --model vanilla --postfix llama-2 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset math --model vanilla --postfix llama-3 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset math --model vanilla --postfix llama-4 &

# ENV_FILE=.env.llama python -m run.run_inf --dataset instruction --model vanilla --postfix llama-0 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset instruction --model vanilla --postfix llama-1 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset instruction --model vanilla --postfix llama-2 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset instruction --model vanilla --postfix llama-3 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset instruction --model vanilla --postfix llama-4 &

# ENV_FILE=.env.llama python -m run.run_inf --dataset travelplanner --model vanilla --postfix llama-0 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset travelplanner --model vanilla --postfix llama-1 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset travelplanner --model vanilla --postfix llama-2 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset travelplanner --model vanilla --postfix llama-3 &
# ENV_FILE=.env.llama python -m run.run_inf --dataset travelplanner --model vanilla --postfix llama-4 &

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model vanilla --postfix o3mini-0
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model global_budget --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_global --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_uniform --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay constant --postfix o3mini-constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay linear --postfix o3mini-linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay polynomial --postfix o3mini-polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay exponential --postfix o3mini-exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay cosine --postfix o3mini-cosine-0 &

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model vanilla --postfix o4mini-0
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model global_budget --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_global --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_uniform --postfix o4mini-0 
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay constant --postfix o4mini-constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay linear --postfix o4mini-linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay polynomial --postfix o4mini-polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay exponential --postfix o4mini-exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay cosine --postfix o4mini-cosine-0

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model vanilla --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model global_budget --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_global --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_uniform --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay constant --postfix o4mini-constant-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay linear --postfix o4mini-linear-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay polynomial --postfix o4mini-polynomial-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay exponential --postfix o4mini-exponential-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay cosine --postfix o4mini-cosine-1 &

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=128 --dataset math --model vanilla --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=128 --dataset math --model planned --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=128 --dataset math --model global_budget --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=128 --dataset math --model planned_global --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=128 --dataset math --model planned_local_uniform --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=128 --dataset math --model planned_local_weighted --decay constant --postfix o4mini-constant-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay linear --postfix o4mini-linear-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay polynomial --postfix o4mini-polynomial-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay exponential --postfix o4mini-exponential-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset math --model planned_local_weighted --decay cosine --postfix o4mini-cosine-2 &


##################################### Math Dataset #####################################
# python -m run.run_inf --dataset math --model vanilla --postfix -0 &
# python -m run.run_inf --dataset math --model planned --postfix -0 &
ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset math --model global_budget --config token_bias=100 token_slope=75 --postfix _b100_s75-0 &
# python -m run.run_inf --dataset math --model planned_global --postfix -0 &
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -0 
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-0

# python -m run.run_inf --dataset math --model vanilla --postfix -1 &
# python -m run.run_inf --dataset math --model planned --postfix -1 &
# python -m run.run_inf --dataset math --model global_budget --postfix -1 &
# python -m run.run_inf --dataset math --model planned_global --postfix -1 &
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -1 
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-1 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-1 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-1 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-1 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-1

# python -m run.run_inf --dataset math --model vanilla --postfix -2 &
# python -m run.run_inf --dataset math --model planned --postfix -2 &
# python -m run.run_inf --dataset math --model global_budget --postfix -2 &
# python -m run.run_inf --dataset math --model planned_global --postfix -2 &
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -2 
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-2 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-2 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-2 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-2 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-2

# python -m run.run_inf --dataset math --model vanilla --postfix -3 &
# python -m run.run_inf --dataset math --model planned --postfix -3 &
# python -m run.run_inf --dataset math --model global_budget --postfix -3 &
# python -m run.run_inf --dataset math --model planned_global --postfix -3 &
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -3 
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-3 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-3 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-3 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-3 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-3

# python -m run.run_inf --dataset math --model vanilla --postfix -4 &
# python -m run.run_inf --dataset math --model planned --postfix -4 &
# python -m run.run_inf --dataset math --model global_budget --postfix -4 &
# python -m run.run_inf --dataset math --model planned_global --postfix -4 &
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -4 
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-4 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-4 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-4 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-4 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-4





##################################### Instruction Dataset #####################################
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model vanilla --postfix o3mini-0
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model global_budget --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_global --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_uniform --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay constant --postfix o3mini-constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay linear --postfix o3mini-linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay polynomial --postfix o3mini-polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay exponential --postfix o3mini-exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay cosine --postfix o3mini-cosine-0 &

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model vanilla --postfix o4mini-0
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model global_budget --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_global --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_uniform --postfix o4mini-0 
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay constant --postfix o4mini-constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay linear --postfix o4mini-linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay polynomial --postfix o4mini-polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay exponential --postfix o4mini-exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay cosine --postfix o4mini-cosine-0

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model vanilla --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model global_budget --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_global --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_uniform --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay constant --postfix o4mini-constant-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay linear --postfix o4mini-linear-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay polynomial --postfix o4mini-polynomial-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay exponential --postfix o4mini-exponential-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay cosine --postfix o4mini-cosine-1 &

# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model vanilla --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model global_budget --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_global --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_uniform --postfix o4mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay constant --postfix o4mini-constant-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay linear --postfix o4mini-linear-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay polynomial --postfix o4mini-polynomial-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay exponential --postfix o4mini-exponential-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --num-workers=500 --dataset instruction --model planned_local_weighted --decay cosine --postfix o4mini-cosine-2 &


# python -m run.run_inf --dataset instruction --model vanilla --postfix -0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model vanilla --postfix o3mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model vanilla --postfix o3mini-2 &
# python -m run.run_inf --dataset instruction --model vanilla --postfix -3 &
# python -m run.run_inf --dataset instruction --model vanilla --postfix -4

# python -m run.run_inf --dataset instruction --model planned --postfix -0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned --postfix o3mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned --postfix o3mini-2 &
# python -m run.run_inf --dataset instruction --model planned --postfix -3 &
# python -m run.run_inf --dataset instruction --model planned --postfix -4

# python -m run.run_inf --dataset instruction --model global_budget --postfix -0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model global_budget --postfix o3mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model global_budget --postfix o3mini-2 &
# python -m run.run_inf --dataset instruction --model global_budget --postfix -3 &
# python -m run.run_inf --dataset instruction --model global_budget --postfix -4

# python -m run.run_inf --dataset instruction --model planned_global --postfix -0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_global --postfix o3mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_global --postfix o3mini-2 &
# python -m run.run_inf --dataset instruction --model planned_global --postfix -3 &
# python -m run.run_inf --dataset instruction --model planned_global --postfix -4

# python -m run.run_inf --dataset instruction --model planned_local_uniform --postfix -0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_uniform --postfix o3mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_uniform --postfix o3mini-2 &
# python -m run.run_inf --dataset instruction --model planned_local_uniform --postfix -3 &
# python -m run.run_inf --dataset instruction --model planned_local_uniform --postfix -4

# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay constant --postfix constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay constant --postfix o3mini_constant-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay constant --postfix o3mini_constant-2 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay constant --postfix constant-3 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay constant --postfix constant-4

# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay linear --postfix linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay linear --postfix o3mini_linear-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay linear --postfix o3mini_linear-2 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay linear --postfix linear-3 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay linear --postfix linear-4

# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay polynomial --postfix polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay polynomial --postfix o3mini_polynomial-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay polynomial --postfix o3mini_polynomial-2 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay polynomial --postfix polynomial-3 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay polynomial --postfix polynomial-4

# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay exponential --postfix exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay exponential --postfix o3mini_exponential-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay exponential --postfix o3mini_exponential-2 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay exponential --postfix exponential-3 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay exponential --postfix exponential-4

# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay cosine --postfix cosine-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay cosine --postfix o3mini_cosine-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset instruction --model planned_local_weighted --decay cosine --postfix o3mini_cosine-2 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay cosine --postfix cosine-3 &
# python -m run.run_inf --dataset instruction --model planned_local_weighted --decay cosine --postfix cosine-4


# Travel Planner Dataset
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix o3mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix o3mini_constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix o3mini_linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix o3mini_polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix o3mini_exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix o3mini_cosine-0 &

# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix o4mini-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix o4mini_constant-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix o4mini_linear-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix o4mini_polynomial-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix o4mini_exponential-0 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix o4mini_cosine-0 &

# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix o4mini-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix o4mini_constant-1 &
# ENV_FILE=.env.gpt python -m run.run_inf  --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix o4mini_linear-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix o4mini_polynomial-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix o4mini_exponential-1 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix o4mini_cosine-1 &

# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix o3mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned --postfix o3mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix o3mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix o3mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix o3mini-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix o3mini_constant-2 &
# ENV_FILE=.env.gpt python -m run.run_inf  --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix o3mini_linear-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix o3mini_polynomial-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix o3mini_exponential-2 &
# ENV_FILE=.env.gpt python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix o3mini_cosine-2 &


# python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -0 &
# python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -0 
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-0 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-0

# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-2 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-2 &

# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-4 &
# ENV_FILE=.env.ds_qwen python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-4 &

ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-4 &
ENV_FILE=.env.qwq python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-4 &

ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-4 &
ENV_FILE=.env.ds_llama python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-4 &

# python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -2 &
# python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -2
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-2 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-2

# python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -3 &
# python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -3
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-3 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-3

# python -m run.run_inf --dataset travelplanner --keep --model vanilla --postfix -4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned --postfix -4 &
# python -m run.run_inf --dataset travelplanner --keep --model global_budget --postfix -4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_global --postfix -4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_uniform --postfix -4
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay constant --postfix constant-4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay linear --postfix linear-4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay polynomial --postfix polynomial-4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay exponential --postfix exponential-4 &
# python -m run.run_inf --dataset travelplanner --keep --model planned_local_weighted --decay cosine --postfix cosine-4

# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix -0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix -0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix -0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix -0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix -0 
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix constant-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix linear-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix polynomial-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix exponential-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix cosine-0

# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix -1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix -1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix -1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix -1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix -1 
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix constant-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix linear-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix polynomial-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix exponential-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix cosine-1 &

# Evaluate travel planner exp #2
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix -2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix -2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix -2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix -2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix -2 
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix constant-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix linear-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix polynomial-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix exponential-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix cosine-2 &

# Evaluate travel planner exp #3
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix -3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix -3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix -3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix -3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix -3 
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix constant-3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix linear-3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix polynomial-3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix exponential-3 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix cosine-3 &

# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix o3mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix o3mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix o3mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix o3mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix o3mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_constant-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_linear-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_polynomial-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_exponential-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_cosine-0 &

# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix o3mini-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix o3mini-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix o3mini-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix o3mini-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix o3mini-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_constant-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_linear-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_polynomial-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_exponential-2 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o3mini_cosine-2 &

# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix o4mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix o4mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix o4mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix o4mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix o4mini-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_constant-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_linear-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_polynomial-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_exponential-0 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_cosine-0 &

# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model vanilla --postfix o4mini-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned --postfix o4mini-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model global_budget --postfix o4mini-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_global --postfix o4mini-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_uniform --postfix o4mini-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_constant-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_linear-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_polynomial-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_exponential-1 &
# ENV_FILE=.env.llama python -m run.run_eval --dataset travelplanner --model planned_local_weighted --postfix o4mini_cosine-1 &


# python -m run.run_inf --dataset travelplanner --model vanilla --keep --postfix -0 &
# python -m run.run_inf --dataset travelplanner --model vanilla --keep --postfix -1 &
# python -m run.run_inf --dataset travelplanner --model vanilla --keep --postfix -2 &
# python -m run.run_inf --dataset travelplanner --model vanilla --keep --postfix -3 &
# python -m run.run_inf --dataset travelplanner --model vanilla --keep --postfix -4

# python -m run.run_inf --dataset travelplanner --model planned --keep --postfix -0 &
# python -m run.run_inf --dataset travelplanner --model planned --keep --postfix -1 &
# python -m run.run_inf --dataset travelplanner --model planned --keep --postfix -2 &
# python -m run.run_inf --dataset travelplanner --model planned --keep --postfix -3 &
# python -m run.run_inf --dataset travelplanner --model planned --keep --postfix -4

# python -m run.run_inf --dataset travelplanner --model global_budget --keep --postfix -0 &
# python -m run.run_inf --dataset travelplanner --model global_budget --keep --postfix -1 &
# python -m run.run_inf --dataset travelplanner --model global_budget --keep --postfix -2 &
# python -m run.run_inf --dataset travelplanner --model global_budget --keep --postfix -3 &
# python -m run.run_inf --dataset travelplanner --model global_budget --keep --postfix -4

# python -m run.run_inf --dataset travelplanner --model planned_global --keep --postfix -0
# python -m run.run_inf --dataset travelplanner --model planned_global --keep --postfix -1 &
# python -m run.run_inf --dataset travelplanner --model planned_global --keep --postfix -2 &
# python -m run.run_inf --dataset travelplanner --model planned_global --keep --postfix -3 &
# python -m run.run_inf --dataset travelplanner --model planned_global --keep --postfix -4

# python -m run.run_inf --dataset travelplanner --model planned_local_uniform --keep --postfix -0 &
# python -m run.run_inf --dataset travelplanner --model planned_local_uniform --keep --postfix -1 &
# python -m run.run_inf --dataset travelplanner --model planned_local_uniform --keep --postfix -2 &
# python -m run.run_inf --dataset travelplanner --model planned_local_uniform --keep --postfix -3 &
# python -m run.run_inf --dataset travelplanner --model planned_local_uniform --keep --postfix -4

# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay constant --postfix constant-0 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay constant --postfix constant-1 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay constant --postfix constant-2 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay constant --postfix constant-3 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay constant --postfix constant-4

# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay linear --postfix linear-0 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay linear --postfix linear-1 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay linear --postfix linear-2 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay linear --postfix linear-3 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay linear --postfix linear-4

# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay polynomial --postfix polynomial-0 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay polynomial --postfix polynomial-1 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay polynomial --postfix polynomial-2 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay polynomial --postfix polynomial-3 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay polynomial --postfix polynomial-4

# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay exponential --postfix exponential-0 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay exponential --postfix exponential-1 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay exponential --postfix exponential-2 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay exponential --postfix exponential-3 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay exponential --postfix exponential-4

# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay cosine --postfix cosine-0 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay cosine --postfix cosine-1 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay cosine --postfix cosine-2 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay cosine --postfix cosine-3 &
# python -m run.run_inf --dataset travelplanner --model planned_local_weighted --keep --decay cosine --postfix cosine-4

# Math Dataset
# python -m run.run_inf --dataset math --model vanilla --postfix -0 &
# python -m run.run_inf --dataset math --model planned --postfix -0 &
# python -m run.run_inf --dataset math --model global_budget --postfix -0 &
# python -m run.run_inf --dataset math --model planned_global --postfix -0 &
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -0
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-0 &
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-0




# ENV_FILE=.env.gpt python -m run.run_inf --dataset math --model io --postfix o1mini-0 --num-workers=250

# python -m run.run_inf --dataset math --model vanilla --postfix -0
# python -m run.run_inf --dataset math --model planned --postfix -0
# python -m run.run_inf --dataset math --model global_budget --postfix -0
# python -m run.run_inf --dataset math --model planned_global --postfix -0
# python -m run.run_inf --dataset math --model planned_local_uniform --postfix -0
# python -m run.run_inf --dataset math --model planned_local_weighted --decay constant --postfix constant-0
# python -m run.run_inf --dataset math --model planned_local_weighted --decay linear --postfix linear-0
# python -m run.run_inf --dataset math --model planned_local_weighted --decay polynomial --postfix polynomial-0
# python -m run.run_inf --dataset math --model planned_local_weighted --decay exponential --postfix exponential-0
# python -m run.run_inf --dataset math --model planned_local_weighted --decay cosine --postfix cosine-0

# python -m run.run_inf --dataset instruction --model vanilla --postfix -0
# python -m run.run_inf --dataset instruction --model planned --postfix -0
# python -m run.run_inf --dataset instruction --model global_budget --postfix -0
# python -m run.run_inf --dataset instruction --model planned_global --postfix -0
# python -m run.run_inf --dataset instruction --model weighted --decay constant --postfix constant-0
# python -m run.run_inf --dataset instruction --model weighted --decay linear --postfix linear-0
# python -m run.run_inf --dataset instruction --model weighted --decay polynomial --postfix polynomial-0
# python -m run.run_inf --dataset instruction --model weighted --decay exponential --postfix exponential-0
# python -m run.run_inf --dataset instruction --model weighted --decay cosine --postfix cosine-0