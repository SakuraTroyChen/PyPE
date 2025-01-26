DATA_PATH=../LLaVA/playground/data/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=../LLaVA/playground/data/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=../LLaVA/playground/data/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=../LLaVA/playground/data #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"


# bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" > logs/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain-inv_pyramid3x_correct.out 2>&1
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" > logs/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune.out 2>&1
