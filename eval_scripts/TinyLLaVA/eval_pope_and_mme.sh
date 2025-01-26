model=tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained=checkpoints/${model} \
    --tasks pope_full,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_pope_full_and_mme \
    --output_path ./logs/${model}
