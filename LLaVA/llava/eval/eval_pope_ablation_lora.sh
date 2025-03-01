root=/home/xingyun
llava_root=${root}/xingy/cca-llava
huggingface_root=${root}/huggingface
pretrain_root=${llava_root}/ablation_pretrain_checkpoints
model_root=${llava_root}/ablation_checkpoints
# ---- test scenario
scenario=("aokvqa")  # "coco", "gqa", "aokvqa"
# ---- pope subset
pope_subset=("adversarial")  # "random" "popular" "adversarial"
# ---- image root
img_roots=${root}/data/coco/val2014  #  "coco/val2014" "gqa/images"
# ---- output root
output_root='ablation_outputs'
# ---- modeling llama
modeling_llama_root=${llava_root}/src/transformers-4.37.2/src/transformers/models/llama
method="rope_2d_d"  # "rope_2d_d" "rope_identical" "official" "rope_2d_d_offset"
# ---- model
base=hf-llava-1.5-7b-s1-558k-8x4x8-${method}
lora=llava-1.5-7b-s3-lora-5k-4x4x8-${method}

echo "------------- Running for model: $method -------------"    
echo "------------- Running for model: $model -------------"    
echo "------------- Running for scenario: $scenario -------------"
echo "------------- Running for subset: $pope_subset -------------"

rm ${modeling_llama_root}/modeling_llama.py
ln -s ${modeling_llama_root}/modeling_llama_${method}.py ${modeling_llama_root}/modeling_llama.py

question_file=${llava_root}/pope/${scenario}/${scenario}_pope_${pope_subset}.json
answer_file=${llava_root}/${output_root}/${scenario}/${model}_${scenario}_pope_${pope_subset}_ans.json
            
if test -e ${answer_file}; then
    python eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file}
else
    python model_vqa.py \
        --model-path ${model_root}/${lora} \
        --model-base ${pretrain_root}/${base} \
        --question-file ${question_file} \
        --image-folder ${img_roots} \
        --answers-file ${answer_file}
    python eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file}
fi
