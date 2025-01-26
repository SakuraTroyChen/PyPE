import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

from PIL import Image

### CCA ###
import transformers
from llava.cca_utils.cca import llamaforcausallm_forward, navigator_forward

transformers.models.llama.LlamaModel.forward = navigator_forward
transformers.models.llama.LlamaForCausalLM.forward = llamaforcausallm_forward


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    with open(os.path.expanduser(args.question_file), "r") as f:
        data = json.load(f)

    img_list = []
    for datum in data:
        img_path = "VG_100K" + datum["image"]["url"].split("VG_100K")[-1]
        img_id = datum["image_id"]
        img_list.append((img_id, img_path))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for img in tqdm(img_list):
        img_id = img[0]
        image_file = img[1]
        qs = "Describe this image in detail."

        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                temperature=1,
                num_beams=5,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)[
            0
        ].strip()

        ans_file.write(
            json.dumps(
                {
                    img_id: outputs,
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument(
        "--question-file", type=str, default="llava/eval/val_images_final.json"
    )
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    args = parser.parse_args()

    eval_model(args)
