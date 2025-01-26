DATA_PATH=../LLaVA/playground/data/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=../LLaVA/playground/data/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=../LLaVA/playground/data/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=../LLaVA/playground/data #finetune image dir

LLM_VERSION=Qwen/Qwen2.5-3B # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2_5-3b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

bash scripts/train/qwen2/pretrain_qwen2.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" > logs/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain-inv_pyramid1x.out 2>&1
# bash scripts/train/qwen2/finetune_qwen2.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" > logs/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune-inv_pyramid1x.out 2>&1"$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
