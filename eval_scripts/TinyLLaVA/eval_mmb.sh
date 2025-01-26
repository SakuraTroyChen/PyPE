model=tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained=checkpoints/${model} \
    --tasks mmbench,mmt \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_mmb \
    --output_path ./logs/${model}

model=tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained=checkpoints/${model} \
    --tasks ai2d,mmvet,mmmu,mmmu_pro,ocrbench,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_mmb \
    --output_path ./logs/${model}
