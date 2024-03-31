from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import LlamaTokenizer, AutoTokenizer
from llm_nav.model.clip_xformer import CLIPVisionModel
from transformers import MptForCausalLM
# from transformers import LlamaForCausalLM
from llm_nav.model.llama_xformer import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from llm_nav.config import FlamingoConfig

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "RWForCausalLM": "transformer.h",
    "MptForCausalLM": "transformer.blocks",
    "MosaicGPT": "transformer.blocks",
}

MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj",
    "GPTNeoXForCausalLM": "gpt_neox",
    "MPTForCausalLM": "mpt",
    "MosaicGPT": "mpt",
}


def _infer_decoder_layers_attr_name(model: nn.Module):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls),
                         {})  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1:])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def exists(val):
    return val is not None


class FlamingoPerceiverBlock(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8, mult: int = 4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        ff_dim = dim * mult
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        residual_latents = latents
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        out = self.to_out(out) + residual_latents
        residual_out = out
        for layer in self.feed_forward:
            out = layer(out)
        return out + residual_out


class FlamingoPerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            depth: int = 6,
            dim_head: int = 64,
            heads: int = 8,
            num_latents: int = 64,
            max_num_media: Optional[int] = None,
            max_num_frames: Optional[int] = 128,
            ff_mult: int = 4,
            use_frame_embs: bool = True,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim))
        self.use_frame_embs = use_frame_embs

        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(FlamingoPerceiverBlock(dim=dim, dim_head=dim_head, heads=heads, mult=ff_mult))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if self.use_frame_embs:
            frame_embs = repeat(self.frame_embs[:T], "T d -> b T F v d", b=b, F=F, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for block in self.layers:
            latents = block(x, latents)
        return self.norm(latents)


class FlamingoMaskedCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            dim_visual: int,
            dim_head: int = 64,
            heads: int = 8,
            only_attend_immediate_media: bool = True,
            stride: int = 1,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media
        self.stride = stride

    def forward(
            self,
            x: torch.Tensor,
            media: torch.Tensor,
            media_locations: Optional[torch.BoolTensor] = None,
            attend_previous: bool = True,
            trunc_locations: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            attend_previous: bool
                If false, ignores immediately preceding image and starts attending when following image
            trunc_locations: boolean mask identifying the media sessions in x
                shape (B, T_txt)
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            # at each boolean of True, increment the time counter (relative to media time)
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(T_img, device=x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            for i in range(self.stride - 1):
                media_time_tmp = media_time + i + 1
                text_to_media_mask_tmp = mask_op(
                    rearrange(text_time, "b i -> b 1 i 1"),
                    repeat(media_time_tmp, "j -> 1 1 1 (j n)", n=n),
                )
                text_to_media_mask = torch.logical_or(text_to_media_mask, text_to_media_mask_tmp)
            if exists(trunc_locations):
                trunc_time = trunc_locations * text_time
                trunc_time, _ = trunc_time.cummax(dim=-1)
                trunc_mask = torch.ge(repeat(media_time, "j -> 1 1 1 (j n)", n=n),
                                      rearrange(trunc_time, "b i -> b 1 i 1"))
                text_to_media_mask = torch.logical_and(text_to_media_mask, trunc_mask)

            text_to_media_mask = text_to_media_mask[:, :, -x.shape[-2]:]
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
            text_without_media_mask = text_without_media_mask[:, :, -x.shape[-2]:]
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FlamingoGatedCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            dim_visual: int,
            dim_head: int = 64,
            heads: int = 8,
            ff_mult: int = 4,
            only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = FlamingoMaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ff_mult, bias=False),
                nn.GELU(),
                nn.Linear(dim * ff_mult, dim, bias=False),
            ]
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
            self,
            x: torch.Tensor,
            media: torch.Tensor,
            media_locations: Optional[torch.BoolTensor] = None,
            attend_previous: bool = True,
            trunc_locations: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = (
                self.attn(
                    x,
                    media,
                    media_locations=media_locations,
                    attend_previous=attend_previous,
                    trunc_locations=trunc_locations
                )
                * self.attn_gate.tanh()
                + x
        )
        residual_x = x
        for ff in self.feed_forward:
            x = ff(x)
        x = x * self.ff_gate.tanh() + residual_x

        return x


class FlamingoLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer: nn.Module, decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        self.trunc_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x) -> None:
        self.vis_x = vis_x

    def condition_trunc_locations(self, trunc_locations) -> None:
        self.trunc_locations = trunc_locations

    def condition_media_locations(self, media_locations) -> None:
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
            self,
            lang_x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
            trunc_locations=self.trunc_locations,
        )
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
            self,
            media_token_id: int,
            vis_hidden_size: int,
            cross_attn_every_n_layers: int,
            only_attend_immediate_media: bool,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """

        gated_cross_attn_layers = nn.ModuleList(
            [
                FlamingoGatedCrossAttentionBlock(
                    dim=self.config.hidden_size,
                    dim_visual=vis_hidden_size,
                    only_attend_immediate_media=only_attend_immediate_media,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(gated_cross_attn_layer, decoder_layer)
                    for gated_cross_attn_layer, decoder_layer in
                    zip(gated_cross_attn_layers, self._get_decoder_layers())
                ]
            )
        )
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = False
        self.initialized_flamingo = True

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                trunc_counts: Optional[list] = None,
                **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError("Flamingo layers are not initialized. Please call `init_flamingo` first.")

        media_locations = input_ids == self.media_token_id
        if past_key_values is not None:
            input_ids = input_ids[:, past_key_values[0][0].shape[-2]:]

        attend_previous = True

        if trunc_counts is not None:
            trunc_locations = torch.zeros_like(media_locations)
            for b in range(media_locations.size(0)):
                indices = trunc_counts[b]
                ones_pos = (media_locations[b] == 1).nonzero(as_tuple=True)[0]
                for idx in indices:
                    if idx < len(ones_pos):
                        trunc_locations[b, ones_pos[idx]] = 1

        if self.__class__.__name__ == "LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                layer.condition_media_locations(media_locations)
                if trunc_counts is not None:
                    layer.condition_trunc_locations(trunc_locations)
                layer.condition_attend_previous(attend_previous)
        elif self.__class__.__name__ in ["MptForCausalLM", "MosaicGPT"]:
            for layer in self.transformer.blocks:
                layer.condition_media_locations(media_locations)
                if trunc_counts is not None:
                    layer.condition_trunc_locations(trunc_locations)
                layer.condition_attend_previous(attend_previous)
        else:
            print("inavaliable text encoder")

        return super().forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               past_key_values=past_key_values,
                               use_cache=use_cache,
                               **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_trunc_locations(None)
            layer.condition_attend_previous(None)


class FlamingoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlamingoConfig
    base_model_prefix = "flamingo"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlamingoPerceiverBlock", "CLIPEncoderLayer", "FlamingoLayer"]

    def _init_weights(self, module):
        """Flamingo requires no specific initialization"""
        return super()._init_weights(module)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value


# The following part will be released after acceptance
