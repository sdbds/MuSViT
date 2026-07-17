import unittest
from unittest.mock import patch

import torch

from experiments.full_page_omr.smt_foundation import modeling_smt
from experiments.full_page_omr.smt_foundation.configuration_smt import SMTFoundationConfig
from experiments.full_page_omr.smt_foundation.modeling_smt import (
    Decoder,
    DecoderLayer,
    MHA,
    SMTFoundationModelForCausalLM,
)


class MHABackendTests(unittest.TestCase):
    @staticmethod
    def _make_identity_attention(backend):
        attention = MHA(embedding_dim=2, num_heads=1, dropout=0.0, attention_backend=backend)
        with torch.no_grad():
            for projection in (attention.lq, attention.lk, attention.lv, attention.out_proj):
                projection.weight.copy_(torch.eye(2))
                projection.bias.zero_()
        attention.eval()
        return attention

    def test_sdpa_matches_eager_for_causal_attention(self):
        torch.manual_seed(7)
        eager = MHA(embedding_dim=8, num_heads=2, dropout=0.0, attention_backend="eager")
        sdpa = MHA(embedding_dim=8, num_heads=2, dropout=0.0, attention_backend="sdpa")
        sdpa.load_state_dict(eager.state_dict())
        eager.eval()
        sdpa.eval()

        hidden = torch.randn(5, 2, 8)
        eager_output = eager(hidden, hidden, hidden, get_weights=False, is_causal=True)
        sdpa_output = sdpa(hidden, hidden, hidden, get_weights=False, is_causal=True)

        torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-5, atol=1e-6)

    def test_projected_key_value_path_matches_regular_attention(self):
        torch.manual_seed(19)
        attention = MHA(
            embedding_dim=8,
            num_heads=2,
            dropout=0.0,
            attention_backend="eager",
        )
        attention.eval()
        query = torch.randn(2, 1, 8)
        key = torch.randn(5, 1, 8)
        value = torch.randn(5, 1, 8)

        projected = attention.project_key_value(key, value)
        actual = attention.forward_projected(
            query,
            projected,
            get_weights=False,
        )
        expected = attention(query, key, value, get_weights=False)

        self.assertEqual(projected.key.shape, torch.Size([1, 2, 5, 4]))
        self.assertEqual(projected.value.shape, torch.Size([1, 2, 5, 4]))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    def test_sdpa_uses_bottom_right_causal_sliding_window(self):
        attention = self._make_identity_attention("sdpa")
        query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        key = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]], [[0.0, 2.0]]]
        )
        value = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[2.0, 2.0]], [[4.0, 0.0]], [[0.0, 4.0]]]
        )

        output = attention(
            query,
            key,
            value,
            get_weights=False,
            is_causal=True,
            window_size=(2, 0),
        )

        q = query[:, 0]
        k = key[:, 0]
        v = value[:, 0]
        scores = q @ k.transpose(0, 1)
        q_position = torch.arange(query.size(0)) + key.size(0) - query.size(0)
        k_position = torch.arange(key.size(0))
        allowed = (k_position <= q_position[:, None]) & (k_position >= q_position[:, None] - 2)
        expected = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ v

        torch.testing.assert_close(output[:, 0], expected, rtol=1e-5, atol=1e-6)

    def test_sdpa_uses_bottom_right_noncausal_sliding_window(self):
        attention = self._make_identity_attention("sdpa")
        query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        key = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]], [[0.0, 2.0]]]
        )
        value = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[2.0, 2.0]], [[4.0, 0.0]], [[0.0, 4.0]]]
        )

        output = attention(query, key, value, get_weights=False, window_size=(1, 1))

        q = query[:, 0]
        k = key[:, 0]
        v = value[:, 0]
        scores = q @ k.transpose(0, 1)
        q_position = torch.arange(query.size(0)) + key.size(0) - query.size(0)
        k_position = torch.arange(key.size(0))
        allowed = (k_position >= q_position[:, None] - 1) & (k_position <= q_position[:, None] + 1)
        expected = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ v

        torch.testing.assert_close(output[:, 0], expected, rtol=1e-5, atol=1e-6)

    def test_auto_dispatches_eligible_inputs_to_flash_attention_2(self):
        attention = self._make_identity_attention("auto")
        query = torch.randn(3, 1, 2)
        key = torch.randn(3, 1, 2)
        calls = []

        def fake_flash_attention(q, k, v, **kwargs):
            calls.append((q.shape, k.shape, v.shape, kwargs))
            return torch.full_like(q, 3.0)

        with patch.object(attention, "_can_use_flash_attention_2", return_value=True), patch.object(
            modeling_smt, "_load_flash_attn_func", return_value=fake_flash_attention
        ):
            output = attention(query, key, key, get_weights=False, is_causal=True)

        self.assertEqual(attention.last_backend, "flash_attention_2")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], torch.Size([1, 3, 1, 2]))
        self.assertTrue(calls[0][3]["causal"])
        self.assertEqual(calls[0][3]["softmax_scale"], 1.0)
        torch.testing.assert_close(output, torch.full_like(output, 3.0))

    def test_flash_attention_runtime_failure_falls_back_to_sdpa(self):
        attention = self._make_identity_attention("auto")
        reference = self._make_identity_attention("sdpa")
        hidden = torch.randn(4, 1, 2)

        def broken_flash_attention(*args, **kwargs):
            raise RuntimeError("no compatible flash attention kernel")

        with patch.object(attention, "_can_use_flash_attention_2", return_value=True), patch.object(
            modeling_smt, "_load_flash_attn_func", return_value=broken_flash_attention
        ):
            output = attention(hidden, hidden, hidden, get_weights=False, is_causal=True)

        expected = reference(hidden, hidden, hidden, get_weights=False, is_causal=True)
        self.assertEqual(attention.last_backend, "sdpa")
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_flash_attention_out_of_memory_is_not_swallowed(self):
        attention = self._make_identity_attention("auto")
        hidden = torch.randn(4, 1, 2)

        def out_of_memory(*args, **kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory")

        with patch.object(attention, "_can_use_flash_attention_2", return_value=True), patch.object(
            modeling_smt, "_load_flash_attn_func", return_value=out_of_memory
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                attention(hidden, hidden, hidden, get_weights=False, is_causal=True)

    def test_requesting_attention_weights_uses_eager_backend(self):
        attention = self._make_identity_attention("auto")
        hidden = torch.randn(4, 2, 2)

        output, weights = attention(hidden, hidden, hidden, get_weights=True, is_causal=True)

        self.assertEqual(attention.last_backend, "eager")
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(weights.shape, torch.Size([2, 4, 4]))

    def test_sdpa_matches_eager_for_additive_and_padding_masks(self):
        torch.manual_seed(11)
        eager = MHA(embedding_dim=8, num_heads=2, dropout=0.0, attention_backend="eager")
        sdpa = MHA(embedding_dim=8, num_heads=2, dropout=0.0, attention_backend="sdpa")
        sdpa.load_state_dict(eager.state_dict())
        eager.eval()
        sdpa.eval()
        query = torch.randn(3, 2, 8)
        key = torch.randn(4, 2, 8)
        value = torch.randn(4, 2, 8)
        attention_mask = torch.zeros(3, 4)
        attention_mask[0, 3] = float("-inf")
        attention_mask[1, 0] = -2.5
        padding_mask = torch.tensor([[False, False, False, False], [False, False, False, True]])

        eager_output = eager(
            query,
            key,
            value,
            attn_mask=attention_mask,
            key_pad_mask=padding_mask,
            get_weights=False,
        )
        sdpa_output = sdpa(
            query,
            key,
            value,
            attn_mask=attention_mask,
            key_pad_mask=padding_mask,
            get_weights=False,
        )

        torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-5, atol=1e-6)

    def test_sdpa_runtime_failure_falls_back_to_eager(self):
        attention = self._make_identity_attention("sdpa")
        reference = self._make_identity_attention("eager")
        hidden = torch.randn(4, 1, 2)

        with patch.object(
            modeling_smt.F,
            "scaled_dot_product_attention",
            side_effect=RuntimeError("sdpa kernel unavailable"),
        ):
            output = attention(hidden, hidden, hidden, get_weights=False, is_causal=True)

        expected = reference(hidden, hidden, hidden, get_weights=False, is_causal=True)
        self.assertEqual(attention.last_backend, "eager")
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_sdpa_out_of_memory_is_not_swallowed(self):
        attention = self._make_identity_attention("sdpa")
        hidden = torch.randn(4, 1, 2)

        with patch.object(
            modeling_smt.F,
            "scaled_dot_product_attention",
            side_effect=torch.OutOfMemoryError("CUDA out of memory"),
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                attention(hidden, hidden, hidden, get_weights=False, is_causal=True)


class DecoderAttentionIntegrationTests(unittest.TestCase):
    def test_decoder_layer_skips_attention_weights_when_not_requested(self):
        layer = DecoderLayer(d_model=8, dim_ff=8, attention_backend="eager")
        layer.eval()
        target = torch.randn(4, 1, 8)
        memory = torch.randn(6, 1, 8)

        output, self_weights, cross_weights = layer(
            target,
            memory,
            need_weights=False,
            self_attention_is_causal=True,
        )

        self.assertEqual(output.shape, target.shape)
        self.assertIsNone(self_weights)
        self.assertIsNone(cross_weights)

    def test_decoder_defaults_to_fused_outputs_without_weights_or_cache(self):
        decoder = Decoder(
            d_model=8,
            dim_ff=8,
            n_layers=2,
            maxlen=16,
            out_categories=12,
            attention_window=17,
            attention_backend="auto",
        )
        decoder.eval()
        memory = torch.randn(6, 1, 8)
        tokens = torch.tensor([[1, 2, 3, 4]])

        output, predictions, _, cache, weights = decoder(
            memory,
            memory,
            tokens,
            reduced_size=[(2, 3)],
            token_len=[4],
            features_size=torch.Size([1, 8, 2, 3]),
        )

        self.assertEqual(output.shape, torch.Size([4, 1, 8]))
        self.assertEqual(predictions.shape, torch.Size([1, 12, 4]))
        self.assertIsNone(cache)
        self.assertIsNone(weights)
        for layer in decoder.decoder.layers:
            self.assertEqual(layer.input_attention.last_backend, "sdpa")
            self.assertEqual(layer.cross_attention.last_backend, "sdpa")

    def test_model_decoder_disables_weights_and_cache_by_default(self):
        class RecordingDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.call_kwargs = None

            def forward(self, raw_features, enhanced_features, tokens, *args, **kwargs):
                self.call_kwargs = kwargs
                token_count = tokens.size(1)
                output = torch.zeros(token_count, tokens.size(0), raw_features.size(-1))
                predictions = torch.zeros(tokens.size(0), 5, token_count)
                return output, predictions, None, None, None

        model = SMTFoundationModelForCausalLM.__new__(SMTFoundationModelForCausalLM)
        torch.nn.Module.__init__(model)
        model.adaptor = torch.nn.Identity()
        model.positional_2D = torch.nn.Identity()
        model.decoder = RecordingDecoder()
        encoder_output = torch.randn(1, 4, 5)
        tokens = torch.tensor([[1, 2, 3]])

        output = model.forward_decoder(encoder_output, tokens, output_attentions=False, use_cache=False)

        self.assertFalse(model.decoder.call_kwargs["keep_all_weights"])
        self.assertFalse(model.decoder.call_kwargs["use_cache"])
        self.assertIsNone(output.attentions)
        self.assertIsNone(output.cross_attentions)

    def test_attention_backend_is_serialized_in_model_config(self):
        config = SMTFoundationConfig(attention_backend="sdpa")
        self.assertEqual(config.attention_backend, "sdpa")

    def test_decoder_can_still_return_attention_weights_for_diagnostics(self):
        decoder = Decoder(
            d_model=8,
            dim_ff=8,
            n_layers=2,
            maxlen=16,
            out_categories=12,
            attention_window=17,
            attention_backend="auto",
        )
        decoder.eval()
        memory = torch.randn(6, 1, 8)
        tokens = torch.tensor([[1, 2, 3, 4]])

        _, _, _, cache, weights = decoder(
            memory,
            memory,
            tokens,
            reduced_size=[(2, 3)],
            token_len=[4],
            features_size=torch.Size([1, 8, 2, 3]),
            keep_all_weights=True,
        )

        self.assertIsNone(cache)
        self.assertEqual(len(weights["self"]), 2)
        self.assertEqual(len(weights["mix"]), 2)
        self.assertEqual(weights["self"][0].shape, torch.Size([1, 4, 4]))
        self.assertEqual(weights["mix"][0].shape, torch.Size([1, 4, 6]))
        for layer in decoder.decoder.layers:
            self.assertEqual(layer.input_attention.last_backend, "eager")
            self.assertEqual(layer.cross_attention.last_backend, "eager")

    def test_decoder_builds_padding_mask_only_for_padded_batches(self):
        decoder = Decoder(
            d_model=8,
            dim_ff=8,
            n_layers=1,
            maxlen=16,
            out_categories=12,
            attention_backend="auto",
        )

        self.assertIsNone(decoder.generate_token_mask([4, 4], (2, 4), torch.device("cpu")))
        mask = decoder.generate_token_mask([2, 4], (2, 4), torch.device("cpu"))

        expected = torch.tensor([[False, False, True, True], [False, False, False, False]])
        torch.testing.assert_close(mask, expected)

    def test_incremental_decoder_matches_full_prefix_logits(self):
        torch.manual_seed(23)
        decoder = Decoder(
            d_model=8,
            dim_ff=8,
            n_layers=2,
            maxlen=16,
            out_categories=12,
            attention_window=17,
            attention_backend="eager",
        )
        decoder.eval()
        raw_memory = torch.randn(6, 1, 8)
        enhanced_memory = torch.randn(6, 1, 8)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])

        memory = decoder.prepare_generation_memory(raw_memory, enhanced_memory)
        state = decoder.init_generation_state(memory)
        for position in range(tokens.size(1)):
            _, step_logits, state = decoder.decode_step(
                memory,
                tokens[:, position:position + 1],
                state,
            )
            _, full_logits, _, _, _ = decoder(
                raw_memory,
                enhanced_memory,
                tokens[:, :position + 1],
                reduced_size=[(2, 3)],
                token_len=[position + 1],
                features_size=torch.Size([1, 8, 2, 3]),
            )

            torch.testing.assert_close(
                step_logits,
                full_logits[:, :, -1:],
                rtol=1e-5,
                atol=1e-5,
            )
            self.assertEqual(state.position, position + 1)
            for layer_state in state.layers:
                self.assertEqual(layer_state.self_kv.key.size(2), position + 1)

    def test_incremental_decoder_reuses_cross_projection_and_honors_finite_window(self):
        torch.manual_seed(29)
        decoder = Decoder(
            d_model=8,
            dim_ff=8,
            n_layers=2,
            maxlen=16,
            out_categories=12,
            attention_window=4,
            attention_backend="eager",
        )
        decoder.eval()
        raw_memory = torch.randn(6, 1, 8)
        enhanced_memory = torch.randn(6, 1, 8)
        key_projections = [
            patch.object(layer.cross_attention.lk, "forward", wraps=layer.cross_attention.lk.forward)
            for layer in decoder.decoder.layers
        ]
        value_projections = [
            patch.object(layer.cross_attention.lv, "forward", wraps=layer.cross_attention.lv.forward)
            for layer in decoder.decoder.layers
        ]

        with key_projections[0] as key_0, key_projections[1] as key_1, \
                value_projections[0] as value_0, value_projections[1] as value_1:
            memory = decoder.prepare_generation_memory(raw_memory, enhanced_memory)
            state = decoder.init_generation_state(memory)
            prefix = []
            expected_projection_calls = 1
            for token_id in range(1, 8):
                prefix.append(token_id)
                _, step_logits, state = decoder.decode_step(
                    memory,
                    torch.tensor([[token_id]]),
                    state,
                )
                self.assertEqual(
                    [key_0.call_count, key_1.call_count],
                    [expected_projection_calls, expected_projection_calls],
                )
                self.assertEqual(
                    [value_0.call_count, value_1.call_count],
                    [expected_projection_calls, expected_projection_calls],
                )
                _, full_logits, _, _, _ = decoder(
                    raw_memory,
                    enhanced_memory,
                    torch.tensor([prefix]),
                    reduced_size=[(2, 3)],
                    token_len=[len(prefix)],
                    features_size=torch.Size([1, 8, 2, 3]),
                )
                torch.testing.assert_close(
                    step_logits,
                    full_logits[:, :, -1:],
                    rtol=1e-5,
                    atol=1e-5,
                )
                expected_projection_calls += 1
                for layer_state in state.layers:
                    self.assertLessEqual(layer_state.self_kv.key.size(2), 3)

        self.assertEqual([key_0.call_count, key_1.call_count], [8, 8])
        self.assertEqual([value_0.call_count, value_1.call_count], [8, 8])


if __name__ == "__main__":
    unittest.main()
