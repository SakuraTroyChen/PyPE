model=llava-1.5-7b-665k
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=checkpoints/${model},attn_implementation="sdpa" \
    --tasks mmbench,mmt \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_mmb \
    --output_path ./logs/${model}

model=llava-1.5-7b-665k
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=checkpoints/${model},attn_implementation="sdpa" \
    --tasks ai2d,mmvet,mmmu,mmmu_pro,ocrbench,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_mmb \
    --output_path ./logs/${model}
