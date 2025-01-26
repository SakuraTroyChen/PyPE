model=tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained=checkpoints/${model} \
    --tasks ok_vqa,gqa,vizwiz_vqa,textvqa,scienceqa_img,realworldqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_six_vqa \
    --output_path ./logs/${model}
