import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from loguru import logger
from torch.nn.init import xavier_uniform_
from transformers import PreTrainedModel, ViTModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .configuration_smt import SMTFoundationConfig

_FLASH_ATTN_FUNC = None
_FLASH_ATTN_IMPORT_ATTEMPTED = False
_FLASH_ATTN_RUNTIME_DISABLED = False
_LOGGED_ATTENTION_BACKENDS = set()


def _load_flash_attn_func():
    global _FLASH_ATTN_FUNC, _FLASH_ATTN_IMPORT_ATTEMPTED
    if _FLASH_ATTN_RUNTIME_DISABLED:
        return None
    if not _FLASH_ATTN_IMPORT_ATTEMPTED:
        _FLASH_ATTN_IMPORT_ATTEMPTED = True
        try:
            from flash_attn import flash_attn_func

            _FLASH_ATTN_FUNC = flash_attn_func
        except (ImportError, OSError) as exc:
            logger.warning(f"FlashAttention 2 is unavailable; falling back to SDPA: {exc}")
    return _FLASH_ATTN_FUNC


def _log_attention_backend(backend):
    if backend not in _LOGGED_ATTENTION_BACKENDS:
        _LOGGED_ATTENTION_BACKENDS.add(backend)
        logger.info(f"Attention backend selected: {backend}")

# MuSViT (LSMT-MAE) vision backbones. Each entry maps the architecture name to
# [implementation class, encoder hidden size].
IMPL_DICT = {
    "ViTMAEModel": [ViTModel, 384],   # MuSViT-Small
    "ViTMAEBase": [ViTModel, 768],    # MuSViT-Base
}


def _normalize_i2w(i2w):
    if not isinstance(i2w, dict):
        raise TypeError(f"i2w must be a dictionary, got {type(i2w).__name__}")

    normalized = {}
    for raw_key, token in i2w.items():
        if isinstance(raw_key, bool):
            raise TypeError("i2w token ids must be integers or numeric strings, got bool")
        if isinstance(raw_key, (int, np.integer)):
            token_id = int(raw_key)
        elif isinstance(raw_key, str) and raw_key.isdecimal():
            token_id = int(raw_key)
        else:
            raise TypeError(
                f"i2w token ids must be integers or numeric strings, got {raw_key!r}"
            )
        if token_id < 0:
            raise ValueError(f"i2w token ids must be non-negative, got {token_id}")
        if not isinstance(token, str):
            raise TypeError(f"i2w[{raw_key!r}] must be a string, got {type(token).__name__}")
        if token_id in normalized and normalized[token_id] != token:
            raise ValueError(
                f"Conflicting i2w entries for token id {token_id}: "
                f"{normalized[token_id]!r} != {token!r}"
            )
        normalized[token_id] = token
    return normalized


class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max=None, w_max=None):
        super(PositionalEncoding2D, self).__init__()
        del h_max, w_max
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 or dim % 4 != 0:
            raise ValueError("2D positional encoding dimension must be a positive multiple of 4")
        self.dim = dim
        self.register_buffer("pe", None, persistent=False)

    def _build(self, h, w, device, dtype):
        calculation_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        div = torch.exp(
            -torch.arange(0, self.dim // 2, 2, device=device, dtype=calculation_dtype)
            / self.dim
            * torch.log(torch.tensor(10000.0, device=device, dtype=calculation_dtype))
        ).unsqueeze(1)
        h_pos = torch.arange(h, device=device, dtype=calculation_dtype).unsqueeze(0)
        w_pos = torch.arange(w, device=device, dtype=calculation_dtype).unsqueeze(0)
        pe = torch.zeros((1, self.dim, h, w), device=device, dtype=calculation_dtype)
        pe[:, :self.dim // 2:2] = torch.sin(div * h_pos).unsqueeze(0).unsqueeze(3).expand(-1, -1, -1, w)
        pe[:, 1:self.dim // 2:2] = torch.cos(div * h_pos).unsqueeze(0).unsqueeze(3).expand(-1, -1, -1, w)
        pe[:, self.dim // 2::2] = torch.sin(div * w_pos).unsqueeze(0).unsqueeze(2).expand(-1, -1, h, -1)
        pe[:, self.dim // 2 + 1::2] = torch.cos(div * w_pos).unsqueeze(0).unsqueeze(2).expand(-1, -1, h, -1)
        return pe.to(dtype=dtype)

    def _ensure_cache(self, h, w, device, dtype):
        expected_shape = (1, self.dim, h, w)
        if (
            self.pe is None
            or tuple(self.pe.shape) != expected_shape
            or self.pe.device != device
            or self.pe.dtype != dtype
        ):
            self.pe = self._build(h, w, device, dtype)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        self._ensure_cache(x.size(2), x.size(3), x.device, x.dtype)
        return x + self.pe

    def get_pe_by_size(self, h, w, device, dtype=torch.float32):
        self._ensure_cache(h, w, torch.device(device), dtype)
        return self.pe


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return x

class MHA(nn.Module):
    def __init__(self, embedding_dim, num_heads=None, dropout=0, proj_value=True, attention_backend="auto") -> None:
        super().__init__()

        if attention_backend not in {"auto", "flash_attention_2", "sdpa", "eager"}:
            raise ValueError(f"Unsupported attention backend: {attention_backend}")

        self.proj_value = proj_value
        self.attention_backend = attention_backend
        self.last_backend = None
        self.lq = nn.Linear(embedding_dim, embedding_dim)
        self.lk = nn.Linear(embedding_dim, embedding_dim)
        if proj_value:
            self.lv = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        # The pretrained decoder used unscaled QK scores; keep that behavior when
        # switching kernels so existing checkpoints remain numerically compatible.
        self.softmax_scale = 1.0
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _structured_allowed_mask(target_len, source_len, is_causal, window_size, device):
        query_positions = torch.arange(target_len, device=device) + source_len - target_len
        key_positions = torch.arange(source_len, device=device)
        allowed = torch.ones((target_len, source_len), dtype=torch.bool, device=device)
        if is_causal:
            allowed &= key_positions <= query_positions.unsqueeze(1)
        if window_size[0] >= 0:
            allowed &= key_positions >= query_positions.unsqueeze(1) - window_size[0]
        if window_size[1] >= 0:
            allowed &= key_positions <= query_positions.unsqueeze(1) + window_size[1]
        return allowed

    def _can_use_flash_attention_2(self, q, k, v, attn_mask, key_pad_mask, get_weights):
        if self.attention_backend not in {"auto", "flash_attention_2"} or get_weights:
            return False
        if attn_mask is not None or key_pad_mask is not None:
            return False
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return False
        if q.dtype not in {torch.float16, torch.bfloat16} or k.dtype != q.dtype or v.dtype != q.dtype:
            return False
        if self.head_dim > 256:
            return False
        return torch.cuda.get_device_capability(q.device)[0] >= 8

    def _eager_attention(self, q, k, v, attn_mask, key_pad_mask, is_causal, window_size):
        target_len = q.size(-2)
        source_len = k.size(-2)
        weights = torch.matmul(q, k.transpose(-2, -1))

        if is_causal or window_size != (-1, -1):
            allowed = self._structured_allowed_mask(
                target_len, source_len, is_causal, window_size, q.device
            )
            weights.masked_fill_(~allowed, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                weights.masked_fill_(attn_mask, float("-inf"))
            else:
                weights += attn_mask
        if key_pad_mask is not None:
            weights.masked_fill_(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        raw_weights = self.softmax(weights)
        return torch.matmul(self.dropout(raw_weights), v), raw_weights
            
    def forward(self, query, key, value, key_pad_mask=None, attn_mask=None, get_weights=True,
                is_causal=False, window_size=(-1, -1)):
        
        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value) if self.proj_value else value

        q = q.view(target_len, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(source_len, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(source_len, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        flash_attn_func = None
        if self._can_use_flash_attention_2(q, k, v, attn_mask, key_pad_mask, get_weights):
            flash_attn_func = _load_flash_attn_func()

        if flash_attn_func is not None:
            q_flash = q.transpose(1, 2).contiguous()
            k_flash = k.transpose(1, 2).contiguous()
            v_flash = v.transpose(1, 2).contiguous()
            try:
                attn_output = flash_attn_func(
                    q_flash,
                    k_flash,
                    v_flash,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=is_causal,
                    window_size=window_size,
                ).transpose(1, 2)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                    raise
                global _FLASH_ATTN_RUNTIME_DISABLED
                _FLASH_ATTN_RUNTIME_DISABLED = True
                logger.warning(f"FlashAttention 2 failed at runtime; falling back to SDPA: {exc}")
                flash_attn_func = None
            else:
                attn_output_weigths_raw = None
                self.last_backend = "flash_attention_2"

        if flash_attn_func is None and self.attention_backend != "eager" and not get_weights:
            sdpa_mask = None
            sdpa_is_causal = (
                is_causal
                and target_len == source_len
                and attn_mask is None
                and key_pad_mask is None
                and window_size == (-1, -1)
            )
            has_structured_mask = is_causal or window_size != (-1, -1)
            if not sdpa_is_causal and (attn_mask is not None or key_pad_mask is not None or has_structured_mask):
                if attn_mask is not None and attn_mask.dtype != torch.bool:
                    sdpa_mask = attn_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
                    sdpa_mask = sdpa_mask.expand(b, 1, target_len, source_len).clone()
                    if has_structured_mask:
                        allowed = self._structured_allowed_mask(
                            target_len, source_len, is_causal, window_size, q.device
                        )
                        sdpa_mask.masked_fill_(~allowed, float("-inf"))
                    if key_pad_mask is not None:
                        sdpa_mask.masked_fill_(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
                else:
                    sdpa_mask = torch.ones((b, 1, target_len, source_len), dtype=torch.bool, device=q.device)
                    if has_structured_mask:
                        sdpa_mask &= self._structured_allowed_mask(
                            target_len, source_len, is_causal, window_size, q.device
                        )
                    if attn_mask is not None:
                        sdpa_mask &= ~attn_mask.unsqueeze(0).unsqueeze(0)
                    if key_pad_mask is not None:
                        sdpa_mask &= ~key_pad_mask.unsqueeze(1).unsqueeze(2)

            try:
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=sdpa_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=sdpa_is_causal,
                    scale=self.softmax_scale,
                )
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                    raise
                logger.warning(f"SDPA failed at runtime; falling back to eager attention: {exc}")
                attn_output, attn_output_weigths_raw = self._eager_attention(
                    q, k, v, attn_mask, key_pad_mask, is_causal, window_size
                )
                self.last_backend = "eager"
            else:
                attn_output_weigths_raw = None
                self.last_backend = "sdpa"
        elif flash_attn_func is None:
            attn_output, attn_output_weigths_raw = self._eager_attention(
                q, k, v, attn_mask, key_pad_mask, is_causal, window_size
            )
            self.last_backend = "eager"

        _log_attention_backend(self.last_backend)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)
        
        if get_weights:
            return attn_output, attn_output_weigths_raw.mean(dim=1)
        
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)

class DecoderLayer(nn.Module):

    def __init__(self, d_model, dim_ff, attention_backend="auto") -> None:
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.ff = dim_ff

        self.input_attention = MHA(embedding_dim=self.d_model,
                             num_heads=4,
                             proj_value=True,
                             dropout=0.1,
                             attention_backend=attention_backend)
        
        self.norm1 = nn.LayerNorm(self.d_model)

        self.cross_attention = MHA(embedding_dim=self.d_model,
                             num_heads=4,
                             proj_value=True,
                             dropout=0.1,
                             attention_backend=attention_backend)

        self.ffNet = nn.Sequential(
            nn.Linear(self.d_model, self.ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ff, self.d_model)
        )

        self.dropout = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
    
    def set_lm_mode(self):
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad = False
        
        for parameter in self.norm2.parameters():
            parameter.requires_grad = False
    
    def set_transcription_mode(self):
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad = True
        
        for parameter in self.norm2.parameters():
            parameter.requires_grad = True

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                predict_n_last_only=None, need_weights=False, self_attention_is_causal=False,
                self_attention_window=(-1, -1)):
        
        if memory_value is None:
            memory_value = memory_key
        
        mha_q = tgt[-predict_n_last_only:] if predict_n_last_only else tgt

        attention_result = self.input_attention(
            mha_q,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_pad_mask=tgt_key_padding_mask,
            get_weights=need_weights,
            is_causal=self_attention_is_causal,
            window_size=self_attention_window,
        )
        if need_weights:
            tgt2, weights_input = attention_result
        else:
            tgt2, weights_input = attention_result, None
        tgt = mha_q + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        att_query = tgt

        attention_result = self.cross_attention(
            att_query,
            memory_key,
            memory_value,
            attn_mask=memory_mask,
            key_pad_mask=memory_key_padding_mask,
            get_weights=need_weights,
        )
        if need_weights:
            tgt2, weights_cross = attention_result
        else:
            tgt2, weights_cross = attention_result, None

        tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffNet(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, weights_input, weights_cross


class DecoderStack(nn.Module):

    def __init__(self, num_dec_layers, d_model, dim_ff, attention_backend="auto") -> None:
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, dim_ff=dim_ff, attention_backend=attention_backend)
            for _ in range(num_dec_layers)
        ])
    
    def set_lm_mode(self):
        for layer in self.layers:
            layer.set_lm_mode()
    
    def set_transcription_mode(self):
        for layer in self.layers:
            layer.set_transcription_mode()

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask, 
                memory_key_padding_mask, use_cache=False, cache=None, predict_last_n_only=False,
                keep_all_weights=False, self_attention_is_causal=True, self_attention_window=(-1, -1)):

        output = tgt
        cache_t = list() if use_cache else None
        all_weights = {
            "self": list(),
            "mix": list()
        } if keep_all_weights else None

        for i, dec_layer in enumerate(self.layers):
            output, weights_self, weights_cross = dec_layer(output, memory_key=memory_key,
                                        memory_value=memory_value,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        predict_n_last_only=predict_last_n_only,
                                        need_weights=keep_all_weights,
                                        self_attention_is_causal=self_attention_is_causal,
                                        self_attention_window=self_attention_window)

            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
        
            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights_cross)

        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, None, cache


class Decoder(nn.Module):
    def __init__(self, d_model, dim_ff, n_layers, maxlen, out_categories, attention_window=100,
                 attention_backend="auto") -> None:
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.dec_attn_win = attention_window
        self.positional_1D = PositionalEncoding1D(d_model, maxlen)

        self.decoder = DecoderStack(
            num_dec_layers=n_layers,
            d_model=d_model,
            dim_ff=dim_ff,
            attention_backend=attention_backend,
        )

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        self.end_relu = nn.ReLU()

        self.out_layer = nn.Conv1d(d_model, out_categories, kernel_size=1)
    
    def set_lm_mode(self):
        self.decoder.set_lm_mode()
    
    def set_transcription_mode(self):
        self.decoder.set_transcription_mode()

    def forward(self, raw_features_1D, enhanced_features_1D, tokens, 
                reduced_size, token_len, features_size, hidden_predict=None, num_pred=None, cache=None,
                keep_all_weights=False, use_cache=False):
        
        device = raw_features_1D.device
        
        pos_tokens = self.embedding(tokens).permute(0,2,1)

        pos_tokens = self.positional_1D(pos_tokens, start=0)
        pos_tokens = pos_tokens.permute(2,0,1).contiguous()

        if num_pred is None:
            num_pred = tokens.size(1)
        
        if not use_cache:
            cache = None
        elif self.dec_attn_win > 1 and cache is not None:
            cache = cache[:, -self.dec_attn_win-1:]
        
        num_tokens_to_keep = num_pred if self.dec_attn_win is None else min([num_pred + self.dec_attn_win - 1, pos_tokens.size(0), token_len[0]])
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        target_mask = None
        memory_mask = None

        key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)
        key_memory_mask = None#self.generate_enc_mask(reduced_size, features_size, device)

        if key_target_mask is not None:
            key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        if self.dec_attn_win in {None, 1} or self.dec_attn_win >= num_tokens_to_keep:
            self_attention_window = (-1, -1)
        else:
            self_attention_window = (self.dec_attn_win - 1, 0)

        output, weights, cache = self.decoder(pos_tokens, memory_key=enhanced_features_1D, memory_value=raw_features_1D, 
                                       tgt_mask=target_mask, memory_mask=memory_mask, tgt_key_padding_mask=key_target_mask, 
                                       memory_key_padding_mask=key_memory_mask, use_cache=use_cache, cache=cache,
                                       predict_last_n_only=num_pred, keep_all_weights=keep_all_weights,
                                       self_attention_is_causal=True, self_attention_window=self_attention_window)

        dpoutput = self.dropout(self.end_relu(output))

        predictions = self.out_layer(dpoutput.permute(1,2,0).contiguous())

        return output, predictions, hidden_predict, cache, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        if all(int(len_) >= len_mask for len_ in token_len):
            return None
        mask = torch.ones((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :int(len_)] = False
        
        return mask
    
    def generate_target_mask(self, target_len, device):
        if self.dec_attn_win == 1:
            return torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=1)
        else:
            return torch.logical_not(
                torch.logical_and(torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                                  torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=-self.dec_attn_win+1)))

class SMTOutput(CausalLMOutputWithCrossAttentions):
    """This is a nice output wrapper"""

class SMTFoundationModelForCausalLM(PreTrainedModel):
    config_class = SMTFoundationConfig

    def __init__(self, config:SMTFoundationConfig):
        super().__init__(config)
        
        if config.foundation_architecture == "ViTMAEBase":
            self.encoder = IMPL_DICT[config.foundation_architecture][0].from_pretrained(config.foundation_weights, mask_ratio=0.0)
        else:
            self.encoder = IMPL_DICT[config.foundation_architecture][0].from_pretrained(config.foundation_weights)

        self.encoder_type = config.foundation_architecture

        self.decoder = Decoder(d_model=config.d_model, dim_ff=config.dim_ff, n_layers=config.num_dec_layers,
                               maxlen=config.maxlen, out_categories=config.out_categories,
                               attention_window=config.maxlen + 1,
                               attention_backend=config.attention_backend)

        adaptor_in = IMPL_DICT[config.foundation_architecture][1]
        #self.weight_sum = nn.Parameter(torch.zeros(25, 1, 1, 1), requires_grad=True)
        self.adaptor = nn.Conv2d(in_channels=adaptor_in, out_channels=256, kernel_size=1, stride=1)
        #self.adaptor = nn.Sequential([
        #    nn.Conv2d(768, 256, 3, padding="same"), 
        #    nn.GeLU(), 
        #    nn.Conv2d(256, 256, 3, padding="same"), 
        #    nn.GeLU(), 
        #    nn.Conv2d(256, 256, 2, 2)])
        
        self.freeze_encoder()
        self.positional_2D = PositionalEncoding2D(config.d_model)

        self.padding_token = config.padding_token
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.w2i = config.w2i
        self.i2w = _normalize_i2w(config.i2w)
        self.maxlen = config.maxlen
        self.out_dir= config.out_dir
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward_encoder(self, x):
        output = self.encoder(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        return output
    
    def forward_decoder(self, encoder_output, y_pred, output_attentions=False, use_cache=False, cache=None):
        b, channels, ln = encoder_output.size()

        reduced_size = [s.shape[:2] for s in encoder_output]
        ylens = [len(sample) for sample in y_pred]

        offset = 1  # discard the encoder [CLS] token

        ln = ln - offset
        encoder_output = encoder_output[:, :, offset:]
        spatial_size = int(ln ** 0.5)
        if spatial_size * spatial_size != ln:
            raise ValueError(f"Encoder token count after CLS removal must be square, got {ln}")
        encoder_output = encoder_output.reshape((b, channels, spatial_size, spatial_size))
        encoder_output = self.adaptor(encoder_output)
        pos_features = self.positional_2D(encoder_output)
        features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2,0,1)
        enhanced_features = features
        enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2,0,1)
        
        output, predictions, _, _, weights = self.decoder(features, enhanced_features, y_pred[:, :], reduced_size, 
                                                           [max(ylens) for _ in range(b)], encoder_output.size(), 
                                                           cache=cache, keep_all_weights=output_attentions,
                                                           use_cache=use_cache)
        return SMTOutput(
            logits=predictions,
            hidden_states=output,
            attentions=weights["self"] if weights is not None else None,
            cross_attentions=weights["mix"] if weights is not None else None,
        )

    def forward(self, x, y_pred, labels=None, output_attentions=None, use_cache=False):
        if output_attentions is None:
            output_attentions = getattr(self.config, "output_attentions", False)
        x = self.forward_encoder(x)
        output = self.forward_decoder(
            x.permute(0,2,1).contiguous(),
            y_pred,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        if labels is not None:
            output.loss = self.loss(output.logits, labels[:, :-1])
        
        return output
    
    @torch.no_grad()    
    def predict(self, input, convert_to_str=False):
        if convert_to_str:
            warnings.warn(
                "convert_to_str is deprecated; vocabulary keys are normalized automatically",
                DeprecationWarning,
                stacklevel=2,
            )
        del convert_to_str
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(input.device).unsqueeze(0)
        encoder_output = self.forward_encoder(input).permute(0,2,1).contiguous()
        text_sequence = []
        for i in range(self.maxlen - predicted_sequence.shape[-1]):
            predictions = self.forward_decoder(
                encoder_output,
                predicted_sequence.long(),
                output_attentions=False,
                use_cache=False,
            )
            next_token = torch.argmax(predictions.logits[:, :, -1], dim=1, keepdim=True)
            predicted_token = int(next_token[0, 0].item())
            predicted_sequence = torch.cat([predicted_sequence, next_token], dim=1)
            try:
                predicted_text = self.i2w[predicted_token]
            except KeyError:
                raise KeyError(f"Unknown predicted token id {predicted_token}") from None
            if predicted_text == '<eos>':
                break
            text_sequence.append(predicted_text)
        
        return text_sequence, predictions
