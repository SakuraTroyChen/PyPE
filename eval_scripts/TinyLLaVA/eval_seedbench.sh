model=tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained=checkpoints/${model} \
    --tasks seedbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_seedbench \
    --output_path ./logs/${model}
