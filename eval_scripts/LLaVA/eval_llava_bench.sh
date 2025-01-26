model=llava-1.5-7b-665k
save_attention_path=attention_visualization/llava-bench-in-the-wild/${model}
python -m llava.eval.model_vqa \
    --model-path checkpoints/${model} \
    --question-file playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file playground/data/eval/llava-bench-in-the-wild/answers/${model}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --save-attention-path ${save_attention_path} \
    --visualize \
    --layer-wise-attention \
