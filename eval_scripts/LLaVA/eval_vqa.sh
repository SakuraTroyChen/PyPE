model=llava-1.5-7b-665k
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=checkpoints/${model},attn_implementation="sdpa" \
    --tasks ok_vqa,gqa,vizwiz_vqa,textvqa,scienceqa_img,realworldqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_six_vqa \
    --output_path ./logs/${model}
