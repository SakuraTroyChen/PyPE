import torch
import torch.nn.functional as F
import math
from llava.constants import IMG_TOKEN_LEN
from typing import List, Optional, Tuple, Union
from transformers.utils import (
    logging,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from torch.nn import CrossEntropyLoss

from celery.contrib import rdb
from einops import rearrange

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"
LLAMA_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
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

H = W = int(math.sqrt(IMG_TOKEN_LEN))

one_pos = torch.full((H, W), 0, dtype=torch.int64)
concentric_pos = torch.zeros(
    H,
    W,
    dtype=torch.int64,
)

pos_pt = [H // 2 - 1, W // 2 - 1, H // 2, W // 2]

for pos in range(11, -1, -1):
    concentric_pos[pos_pt[0] : pos_pt[2] + 1, pos_pt[1]] = pos
    concentric_pos[pos_pt[0] : pos_pt[2] + 1, pos_pt[3]] = pos
    concentric_pos[pos_pt[0], pos_pt[1] : pos_pt[3] + 1] = pos
    concentric_pos[pos_pt[2], pos_pt[1] : pos_pt[3] + 1] = pos
    pos_pt = [pos_pt[0] - 1, pos_pt[1] - 1, pos_pt[2] + 1, pos_pt[3] + 1]


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def navigator_forward(
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

    if "pype" in self.config._name_or_path:
        return pype_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            batch_img_token_pos,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    elif "rev" in self.config._name_or_path:
        return rev_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            batch_img_token_pos,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
    elif "vanilla" in self.config._name_or_path:
        return llamamodel_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            batch_img_token_pos,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
    else:
        return cca_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            batch_img_token_pos,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def pype_forward(
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

    # assert self._use_sdpa

    model_name = self.config._name_or_path
    model_type = model_name.split("pype")[-1]
    jump_num = int(model_type.split("x")[0])

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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

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
        # 可能后面直接张量的boradcast机制会自动扩展，所以这里不需要repeat
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # rdb.set_trace()
    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
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

    ### PyPE ###
    preset_pype_top_img_token_len = 0
    layer_index = 0
    for decoder_layer in self.layers:

        ### pype{jump_num}x ###
        if layer_index % jump_num == 0 and preset_pype_top_img_token_len < H // 2:
            preset_pype_top_img_token_len += 1

        layer_index += 1

        pype_pos = torch.zeros(
            H,
            W,
            dtype=torch.int64,
        )

        preset_pype_img_token_len = H // 2 + 1 - preset_pype_top_img_token_len
        for pos in range(1, preset_pype_img_token_len):
            pype_pos[pos : H - pos, pos : W - pos] = pos

        if batch_img_token_pos[0] == -1:
            batch_pype_position_ids = position_ids
        else:
            batch_pype_position_ids = []

            # image input.
            for b_idx, img_token_pos in enumerate(batch_img_token_pos):
                # position_ids可能是[1,seq_len],也可能是[batch_size,seq_len]
                if b_idx > position_ids.shape[0] - 1:
                    break

                if seq_len == 1:
                    pos = past_key_values[0][0].shape[-2]
                    pype_position_ids = torch.ones(1).to(torch.long).to(
                        self.device
                    ) * (pos - IMG_TOKEN_LEN + preset_pype_img_token_len)
                else:
                    # flatten pype_pos, concat with text positions.
                    pype_position_ids = (
                        torch.cat(
                            [
                                torch.arange(0, img_token_pos),
                                pype_pos.flatten()
                                + img_token_pos.to(pype_pos.device),
                                torch.arange(
                                    img_token_pos + preset_pype_img_token_len,
                                    seq_len
                                    - IMG_TOKEN_LEN
                                    + preset_pype_img_token_len,
                                ),
                            ]
                        )
                        .to(torch.long)
                        .to(self.device)
                    )
                    # print("============= I AM USING pype!!!! =============")

                pype_position_ids = pype_position_ids.unsqueeze(0)
                batch_pype_position_ids.append(pype_position_ids)

            batch_pype_position_ids = torch.cat(batch_pype_position_ids)

        # pype causal masking.
        if seq_len == 1:
            pype_attention_mask = attention_mask
        else:
            if attention_mask is not None:
                pype_attention_mask = attention_mask.clone()
            else:
                # 上三角全为-inf，下三角全为0，相当于直接造一个causal mask
                pype_attention_mask = torch.triu(
                    float("-inf")
                    * torch.ones(batch_size, 1, seq_len, seq_len, device=self.device),
                    diagonal=1,
                ).to(self.dtype)

            if batch_img_token_pos[0] == -1:
                pass
            else:
                # set causal mask according to pype positions.
                if batch_pype_position_ids.shape[0] < batch_size:
                    # 将batch_pype_position_ids扩展到batch_size
                    repeat_batch_pype_position_ids = (
                        batch_pype_position_ids.repeat(
                            batch_size // batch_pype_position_ids.shape[0], 1
                        )
                    )
                else:
                    repeat_batch_pype_position_ids = batch_pype_position_ids

                for (b_idx, img_token_pos), pype_position_ids in zip(
                    enumerate(batch_img_token_pos), repeat_batch_pype_position_ids
                ):
                    pype_attention_mask[
                        b_idx,
                        :,
                        img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                        img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                    ] = float("-inf") * torch.ones(
                        (IMG_TOKEN_LEN, IMG_TOKEN_LEN), device=self.device
                    )
                    # 将causal mask的image对应小于等于当前pos的（需要激活的）mask设置为0.0，因为后面attention_mask是加在attention_weight上的，则-inf的部分的weight会被直接遮掉，而0.0的部份就是激活的
                    for pos in torch.arange(
                        img_token_pos, img_token_pos + preset_pype_img_token_len
                    ):
                        k_pos = torch.nonzero(
                            (pype_position_ids <= pos)
                            * (pype_position_ids >= img_token_pos)
                        )[:, 0]
                        q_pos = torch.nonzero(
                            (pype_position_ids == pos)
                            * (pype_position_ids >= img_token_pos)
                        )[:, 0]
                        m_pos = torch.cartesian_prod(q_pos, k_pos)
                        pype_attention_mask[b_idx, 0, m_pos[:, 0], m_pos[:, 1]] = 0.0

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                pype_attention_mask,
                batch_pype_position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=pype_attention_mask,
                position_ids=batch_pype_position_ids,
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


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def llamamodel_forward(
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

    # assert self._use_sdpa

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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

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
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
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

    for decoder_layer in self.layers:

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


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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

    # assert self._use_sdpa

    one_flag, cca_flag = False, False

    # ONE
    if "one" in self.config._name_or_path:
        one_flag = True
        preset_img_token_len = 1
    # CCA
    elif "cca" in self.config._name_or_path:
        cca_flag = True
        preset_img_token_len = H // 2

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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

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
        # 可能后面直接张量的boradcast机制会自动扩展，所以这里不需要repeat
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # rdb.set_trace()
    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
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
                    pos - IMG_TOKEN_LEN + preset_img_token_len
                )
            else:
                ### CCA ###
                if cca_flag:
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
                    # print("============= I AM USING CCA!!!! =============")

                ### ONE ###
                elif one_flag:
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
                    # print("============= I AM USING ONE!!!! =============")

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
                    img_token_pos, img_token_pos + preset_img_token_len
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



@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def rev_forward(
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

    # assert self._use_sdpa

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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

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
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    ### REV ###
    # meta info for one batch.
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    # reverse positions.
    # text only.
    if batch_img_token_pos[0] == -1:
        batch_rev_position_ids = position_ids
    else:
        batch_rev_position_ids = []

        # image input.
        for b_idx, img_token_pos in enumerate(batch_img_token_pos):
            if b_idx > position_ids.shape[0] - 1:
                break

            if seq_len == 1:
                pos = past_key_values[0][0].shape[-2]
                rev_position_ids = torch.ones(1).to(torch.long).to(self.device) * (pos)
            else:
                # flatten concentric_pos, concat with text positions.
                rev_position_ids = (
                    torch.cat(
                        [
                            position_ids[b_idx, 0:img_token_pos],
                            position_ids[
                                b_idx, img_token_pos : img_token_pos + IMG_TOKEN_LEN
                            ].flip(0),
                            position_ids[b_idx, img_token_pos + IMG_TOKEN_LEN :],
                        ]
                    )
                    .to(torch.long)
                    .to(self.device)
                )
                # print("============= I AM USING REV!!!! =============")

            rev_position_ids = rev_position_ids.unsqueeze(0)
            batch_rev_position_ids.append(rev_position_ids)

        batch_rev_position_ids = torch.cat(batch_rev_position_ids)

    # reverse causal masking.
    if seq_len == 1:
        rev_attention_mask = attention_mask
    else:
        if attention_mask is not None:
            rev_attention_mask = attention_mask.clone()
        else:
            rev_attention_mask = torch.triu(
                float("-inf")
                * torch.ones(batch_size, 1, seq_len, seq_len, device=self.device),
                diagonal=1,
            ).to(self.dtype)

        if batch_img_token_pos[0] == -1:
            pass
        else:
            # set rev causal mask.
            for b_idx, img_token_pos in enumerate(batch_img_token_pos):
                if b_idx > rev_attention_mask.shape[0] - 1:
                    break
                rev_attention_mask[
                    b_idx,
                    :,
                    img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                    img_token_pos : img_token_pos + IMG_TOKEN_LEN,
                ] = torch.tril(
                    float("-inf")
                    * torch.ones(
                        1, 1, IMG_TOKEN_LEN, IMG_TOKEN_LEN, device=self.device
                    ),
                    diagonal=-1,
                ).to(
                    self.dtype
                )

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
                rev_attention_mask,
                batch_rev_position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=rev_attention_mask,
                position_ids=batch_rev_position_ids,
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


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def llamaforcausallm_forward(
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
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
    else:
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
