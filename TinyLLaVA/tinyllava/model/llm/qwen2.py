import transformers
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    logging,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
import ipdb

from . import register_llm
from .gilbert import gilbert2d

QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "Qwen2Config"

# SigLIP
IMG_SIZE = 384
PATCH_SIZE = 14
H = W = int(IMG_SIZE // PATCH_SIZE)
IMG_TOKEN_LEN = int(H * W)

ONE_FLAG, CCA_FLAG = False, False
PRESET_IMG_TOKEN_LEN = IMG_TOKEN_LEN
JUMP_NUM = 1

one_pos = torch.full((H, W), 0, dtype=torch.int64)
concentric_pos = torch.zeros(
    H,
    W,
    dtype=torch.int64,
)

# GILBERT CURVE
gilbert_pos = torch.zeros(H, W, dtype=torch.int64)


@register_llm("qwen2")
def return_qwen2class_for_train(tiny_config):
    global IMG_SIZE, PATCH_SIZE, H, W, IMG_TOKEN_LEN, ONE_FLAG, CCA_FLAG, PRESET_IMG_TOKEN_LEN, JUMP_NUM, one_pos, concentric_pos, gilbert_pos

    Qwen2ForCausalLM.forward = qwen2forcausallm_forward
    Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation

    pe_model_name_or_path = "inv_pyramid1x"
    # pe_model_name_or_path = "gilbert"
    # pe_model_name_or_path = "cca"
    # pe_model_name_or_path = "one"
    # pe_model_name_or_path = "vanilla"

    IMG_SIZE = tiny_config.vision_config.image_size
    PATCH_SIZE = tiny_config.vision_config.patch_size
    H = W = int(IMG_SIZE // PATCH_SIZE)
    IMG_TOKEN_LEN = int(H * W)

    if "inv_pyramid" in pe_model_name_or_path:
        transformers.models.qwen2.Qwen2Model.forward = inv_pyramid_forward
        model_type = pe_model_name_or_path.split("pyramid")[-1]
        JUMP_NUM = int(model_type.split("x")[0])
        print(
            "################# Using INV_PYRAMID with jump_num: {} #################".format(
                JUMP_NUM
            )
        )

    elif "gilbert" in pe_model_name_or_path:
        transformers.models.qwen2.Qwen2Model.forward = gilbert_forward
        # GILBERT CURVE
        dist_index = 0
        for x, y in gilbert2d(H, W):
            gilbert_pos[x, y] = dist_index
            dist_index += 1
        print("################# Using GILBERT #################")

    elif "cca" in pe_model_name_or_path:
        transformers.models.qwen2.Qwen2Model.forward = cca_forward
        CCA_FLAG = True
        PRESET_IMG_TOKEN_LEN = H // 2
        for pos in range(1, H // 2):
            concentric_pos[pos : H - pos, pos : W - pos] = pos
        print("################# Using CCA #################")

    elif "one" in pe_model_name_or_path:
        transformers.models.qwen2.Qwen2Model.forward = cca_forward
        ONE_FLAG = True
        PRESET_IMG_TOKEN_LEN = 1
        print("################# Using ONE #################")

    else:
        transformers.models.qwen2.Qwen2Model.forward = qwen2model_forward
        print("################# Using VANILLA #################")

    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
        return tokenizer

    return (Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load))


# @register_llm("qwen2")
# def return_qwen2class(tiny_config):
#     global IMG_SIZE, PATCH_SIZE, H, W, IMG_TOKEN_LEN, ONE_FLAG, CCA_FLAG, PRESET_IMG_TOKEN_LEN, JUMP_NUM, one_pos, concentric_pos, gilbert_pos

#     Qwen2ForCausalLM.forward = qwen2forcausallm_forward
#     Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation

#     pe_model_name_or_path = tiny_config._name_or_path

#     IMG_SIZE = tiny_config.vision_config.image_size
#     PATCH_SIZE = tiny_config.vision_config.patch_size
#     H = W = int(IMG_SIZE // PATCH_SIZE)
#     IMG_TOKEN_LEN = int(H * W)

#     if "inv_pyramid" in pe_model_name_or_path:
#         transformers.models.qwen2.Qwen2Model.forward = inv_pyramid_forward
#         model_type = pe_model_name_or_path.split("pyramid")[-1]
#         JUMP_NUM = int(model_type.split("x")[0])
#         print(
#             "################# Using INV_PYRAMID with jump_num: {} #################".format(
#                 JUMP_NUM
#             )
#         )

#     elif "gilbert" in pe_model_name_or_path:
#         transformers.models.qwen2.Qwen2Model.forward = gilbert_forward
#         # GILBERT CURVE
#         dist_index = 0
#         for x, y in gilbert2d(H, W):
#             gilbert_pos[x, y] = dist_index
#             dist_index += 1
#         print("################# Using GILBERT #################")

#     elif "cca" in pe_model_name_or_path:
#         transformers.models.qwen2.Qwen2Model.forward = cca_forward
#         CCA_FLAG = True
#         PRESET_IMG_TOKEN_LEN = H // 2
#         for pos in range(1, H // 2):
#             concentric_pos[pos : H - pos, pos : W - pos] = pos
#         print("################# Using CCA #################")

#     elif "one" in pe_model_name_or_path:
#         transformers.models.qwen2.Qwen2Model.forward = cca_forward
#         ONE_FLAG = True
#         PRESET_IMG_TOKEN_LEN = 1
#         print("################# Using ONE #################")

#     else:
#         transformers.models.qwen2.Qwen2Model.forward = qwen2model_forward
#         print("################# Using VANILLA #################")

#     def tokenizer_and_post_load(tokenizer):
#         tokenizer.unk_token = tokenizer.pad_token
#         return tokenizer

#     return (Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load))


def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
):
    def vanilla_prepare_inputs_for_generation(
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    images = kwargs.pop("images", None)
    image_sizes = kwargs.pop("image_sizes", None)
    batch_img_token_pos = kwargs.pop(
        "batch_img_token_pos", None
    )  # ! img_start position

    inputs = vanilla_prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    if images is not None:
        inputs["images"] = images
    if image_sizes is not None:
        inputs["image_sizes"] = image_sizes
    # ! img_start position
    if batch_img_token_pos is not None:
        inputs["batch_img_token_pos"] = batch_img_token_pos

    return inputs


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def qwen2model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    batch_img_token_pos: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if (
        attention_mask is not None
        and self._attn_implementation == "flash_attention_2"
        and use_cache
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def inv_pyramid_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    batch_img_token_pos: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    past_key_values_length = 0

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if (
        attention_mask is not None
        and self._attn_implementation == "flash_attention_2"
        and use_cache
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    # meta info for one batch.
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    ### Pyramid ###
    preset_pyramid_img_token_len = 0
    layer_index = 0
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        ### pyramid{jump_num}x ###
        if layer_index % JUMP_NUM == 0 and preset_pyramid_img_token_len < H // 2:
            preset_pyramid_img_token_len += 1

        layer_index += 1

        pyramid_pos = torch.zeros(
            H,
            W,
            dtype=torch.int64,
        )

        for pos in range(1, H // 2 + 1 - preset_pyramid_img_token_len):
            pyramid_pos[pos : H - pos, pos : W - pos] = pos

        if batch_img_token_pos[0] == -1:
            batch_pyramid_position_ids = position_ids
        else:
            batch_pyramid_position_ids = []

            # image input.
            for b_idx, img_token_pos in enumerate(batch_img_token_pos):
                # position_ids可能是[1,seq_len],也可能是[batch_size,seq_len]
                if b_idx > position_ids.shape[0] - 1:
                    break

                if seq_len == 1:
                    pos = past_key_values[0][0].shape[-2]
                    pyramid_position_ids = torch.ones(1).to(torch.long).to(
                        self.device
                    ) * (pos - IMG_TOKEN_LEN + preset_pyramid_img_token_len)
                else:
                    # flatten pyramid_pos, concat with text positions.
                    pyramid_position_ids = (
                        torch.cat(
                            [
                                torch.arange(0, img_token_pos),
                                pyramid_pos.flatten()
                                + img_token_pos.to(pyramid_pos.device),
                                torch.arange(
                                    img_token_pos + preset_pyramid_img_token_len,
                                    seq_len
                                    - IMG_TOKEN_LEN
                                    + preset_pyramid_img_token_len,
                                ),
                            ]
                        )
                        .to(torch.long)
                        .to(self.device)
                    )

                pyramid_position_ids = pyramid_position_ids.unsqueeze(0)
                batch_pyramid_position_ids.append(pyramid_position_ids)

            batch_pyramid_position_ids = torch.cat(batch_pyramid_position_ids)

        # pyramid causal masking.
        if seq_len == 1:
            pyramid_attention_mask = attention_mask
        else:
            if attention_mask is not None:
                pyramid_attention_mask = attention_mask.clone()
            else:
                # 上三角全为-inf，下三角全为0，相当于直接造一个causal mask
                pyramid_attention_mask = torch.triu(
                    float("-inf")
                    * torch.ones(batch_size, 1, seq_len, seq_len, device=self.device),
                    diagonal=1,
                ).to(self.dtype)

            if batch_img_token_pos[0] == -1:
                pass
            else:
                # set causal mask according to pyramid positions.
                if batch_pyramid_position_ids.shape[0] < batch_size:
                    # 将batch_pyramid_position_ids扩展到batch_size
                    repeat_batch_pyramid_position_ids = (
                        batch_pyramid_position_ids.repeat(
                            batch_size // batch_pyramid_position_ids.shape[0], 1
                        )
                    )
                else:
                    repeat_batch_pyramid_position_ids = batch_pyramid_position_ids

                for (b_idx, img_token_pos), pyramid_position_ids in zip(
                    enumerate(batch_img_token_pos), repeat_batch_pyramid_position_ids
                ):
                    pyramid_attention_mask[
                        b_idx,
                        :,
                        img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                        img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                    ] = float("-inf") * torch.ones(
                        (IMG_TOKEN_LEN, IMG_TOKEN_LEN), device=self.device
                    )
                    # 将causal mask的image对应小于等于当前pos的（需要激活的）mask设置为0.0，因为后面attention_mask是加在attention_weight上的，则-inf的部分的weight会被直接遮掉，而0.0的部份就是激活的
                    for pos in torch.arange(
                        img_token_pos, img_token_pos + preset_pyramid_img_token_len
                    ):
                        k_pos = torch.nonzero(
                            (pyramid_position_ids <= pos)
                            * (pyramid_position_ids >= img_token_pos)
                        )[:, 0]
                        q_pos = torch.nonzero(
                            (pyramid_position_ids == pos)
                            * (pyramid_position_ids >= img_token_pos)
                        )[:, 0]
                        m_pos = torch.cartesian_prod(q_pos, k_pos)
                        pyramid_attention_mask[b_idx, 0, m_pos[:, 0], m_pos[:, 1]] = 0.0

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                pyramid_attention_mask,
                batch_pyramid_position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=pyramid_attention_mask,
                position_ids=batch_pyramid_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def gilbert_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    batch_img_token_pos: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    inputs_embeds = self.embed_dropout(inputs_embeds)

    if (
        attention_mask is not None
        and self._attn_implementation == "flash_attention_2"
        and use_cache
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    # meta info for one batch.
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    if batch_img_token_pos[0] == -1:
        batch_position_ids = position_ids
    else:
        batch_position_ids = []

        # image input.
        for b_idx, img_token_pos in enumerate(batch_img_token_pos):
            # position_ids可能是[1,seq_len],也可能是[batch_size,seq_len]
            if b_idx > position_ids.shape[0] - 1:
                break

            if seq_len == 1:
                pos = past_key_values[0][0].shape[-2]
                gilbert_position_ids = torch.ones(1).to(torch.long).to(self.device) * (
                    pos
                )
            else:
                gilbert_position_ids = (
                    torch.cat(
                        [
                            torch.arange(0, img_token_pos),
                            gilbert_pos.flatten()
                            + img_token_pos.to(gilbert_pos.device),
                            torch.arange(img_token_pos + IMG_TOKEN_LEN, seq_len),
                        ]
                    )
                    .to(torch.long)
                    .to(self.device)
                )

            gilbert_position_ids = gilbert_position_ids.unsqueeze(0)
            batch_position_ids.append(gilbert_position_ids)

        batch_position_ids = torch.cat(batch_position_ids)

    # one causal masking.
    if seq_len == 1:
        gilbert_attention_mask = attention_mask
    else:
        if attention_mask is not None:
            gilbert_attention_mask = attention_mask.clone()
        else:
            # 上三角全为-inf，下三角全为0，相当于直接造一个causal mask
            gilbert_attention_mask = torch.triu(
                float("-inf")
                * torch.ones(batch_size, 1, seq_len, seq_len, device=self.device),
                diagonal=1,
            ).to(self.dtype)

        if batch_img_token_pos[0] == -1:
            pass
        else:
            # set causal mask according to one positions.
            if batch_position_ids.shape[0] < batch_size:
                # 将batch_position_ids扩展到batch_size
                repeat_batch_position_ids = batch_position_ids.repeat(
                    batch_size // batch_position_ids.shape[0], 1
                )
            else:
                repeat_batch_position_ids = batch_position_ids

            for (b_idx, img_token_pos), gilbert_position_ids in zip(
                enumerate(batch_img_token_pos), repeat_batch_position_ids
            ):
                gilbert_attention_mask[
                    b_idx,
                    :,
                    img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                    img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                ] = float("-inf") * torch.ones(
                    (IMG_TOKEN_LEN, IMG_TOKEN_LEN), device=self.device
                )
                # 将causal mask的image对应小于等于当前pos的（需要激活的）mask设置为0.0，因为后面attention_mask是加在attention_weight上的，则-inf的部分的weight会被直接遮掉，而0.0的部份就是激活的
                for pos in torch.arange(img_token_pos, img_token_pos + IMG_TOKEN_LEN):
                    k_pos = torch.nonzero(
                        (gilbert_position_ids <= pos)
                        * (gilbert_position_ids >= img_token_pos)
                    )[:, 0]
                    q_pos = torch.nonzero(
                        (gilbert_position_ids == pos)
                        * (gilbert_position_ids >= img_token_pos)
                    )[:, 0]
                    m_pos = torch.cartesian_prod(q_pos, k_pos)
                    gilbert_attention_mask[b_idx, 0, m_pos[:, 0], m_pos[:, 1]] = 0.0

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                gilbert_attention_mask,
                batch_position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=gilbert_attention_mask,
                position_ids=batch_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def cca_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    batch_img_token_pos: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    inputs_embeds = self.embed_dropout(inputs_embeds)

    if (
        attention_mask is not None
        and self._attn_implementation == "flash_attention_2"
        and use_cache
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    ### CCA ###
    # meta info for one batch.
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    # v1.5每次只从一个模态中sample,所以每次的batch要么都是图片要么都是文字
    # text only.
    if batch_img_token_pos[0] == -1:
        batch_cca_position_ids = position_ids
    else:
        batch_cca_position_ids = []

        # image input.
        for b_idx, img_token_pos in enumerate(batch_img_token_pos):
            # position_ids可能是[1,seq_len],也可能是[batch_size,seq_len]
            if b_idx > position_ids.shape[0] - 1:
                break

            if seq_len == 1:
                pos = past_key_values[0][0].shape[-2]
                cca_position_ids = torch.ones(1).to(torch.long).to(self.device) * (
                    pos - IMG_TOKEN_LEN + PRESET_IMG_TOKEN_LEN
                )
            else:
                ### CCA ###
                if CCA_FLAG:
                    # flatten concentric_pos, concat with text positions.
                    cca_position_ids = (
                        torch.cat(
                            [
                                torch.arange(0, img_token_pos),
                                concentric_pos.flatten()
                                + img_token_pos.to(concentric_pos.device),
                                torch.arange(
                                    img_token_pos + H // 2,
                                    seq_len - IMG_TOKEN_LEN + H // 2,
                                ),
                            ]
                        )
                        .to(torch.long)
                        .to(self.device)
                    )

                ### ONE ###
                elif ONE_FLAG:
                    cca_position_ids = (
                        torch.cat(
                            [
                                torch.arange(0, img_token_pos),
                                one_pos.flatten() + img_token_pos.to(one_pos.device),
                                torch.arange(
                                    img_token_pos + 1, seq_len - IMG_TOKEN_LEN + 1
                                ),
                            ]
                        )
                        .to(torch.long)
                        .to(self.device)
                    )

            cca_position_ids = cca_position_ids.unsqueeze(0)
            batch_cca_position_ids.append(cca_position_ids)

        batch_cca_position_ids = torch.cat(batch_cca_position_ids)

    # concentric causal masking.
    if seq_len == 1:
        cca_attention_mask = attention_mask
    else:
        if attention_mask is not None:
            cca_attention_mask = attention_mask.clone()
        else:
            # 上三角全为-inf，下三角全为0，相当于直接造一个causal mask
            cca_attention_mask = torch.triu(
                float("-inf")
                * torch.ones(batch_size, 1, seq_len, seq_len, device=self.device),
                diagonal=1,
            ).to(self.dtype)

        if batch_img_token_pos[0] == -1:
            pass
        else:
            # set causal mask according to cca posistions.
            if batch_cca_position_ids.shape[0] < batch_size:
                # 将batch_cca_position_ids扩展到batch_size
                repeat_batch_cca_position_ids = batch_cca_position_ids.repeat(
                    batch_size // batch_cca_position_ids.shape[0], 1
                )
            else:
                repeat_batch_cca_position_ids = batch_cca_position_ids

            for (b_idx, img_token_pos), cca_position_ids in zip(
                enumerate(batch_img_token_pos), repeat_batch_cca_position_ids
            ):
                cca_attention_mask[
                    b_idx,
                    :,
                    img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                    img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                ] = float("-inf") * torch.ones(
                    (IMG_TOKEN_LEN, IMG_TOKEN_LEN), device=self.device
                )
                # 将causal mask的image对应小于等于当前pos的（需要激活的）mask设置为0.0，因为后面attention_mask是加在attention_weight上的，则-inf的部分的weight会被直接遮掉，而0.0的部份就是激活的
                for pos in torch.arange(
                    img_token_pos, img_token_pos + PRESET_IMG_TOKEN_LEN
                ):
                    k_pos = torch.nonzero(
                        (cca_position_ids <= pos) * (cca_position_ids >= img_token_pos)
                    )[:, 0]
                    q_pos = torch.nonzero(
                        (cca_position_ids == pos) * (cca_position_ids >= img_token_pos)
                    )[:, 0]
                    m_pos = torch.cartesian_prod(q_pos, k_pos)
                    cca_attention_mask[b_idx, 0, m_pos[:, 0], m_pos[:, 1]] = 0.0

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                cca_attention_mask,
                batch_cca_position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=cca_attention_mask,
                position_ids=batch_cca_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def qwen2forcausallm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    batch_img_token_pos: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        batch_img_token_pos=batch_img_token_pos,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
