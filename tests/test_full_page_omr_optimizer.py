import unittest

import torch
from torch import nn

from experiments.full_page_omr.optimization import (
    AdamWWSDConfig,
    build_adamw,
    build_adamw_parameter_groups,
)


class TinyOMRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.adaptor = nn.Conv2d(4, 4, kernel_size=1)
        self.decoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.in_proj_bias = nn.Parameter(torch.zeros(4))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False


class AdamWWSDParameterGroupTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyOMRModel()
        self.config = AdamWWSDConfig()

    def test_default_config_matches_locked_protocol(self):
        self.assertEqual(self.config.max_steps, 4_000_000)
        self.assertEqual(self.config.stable_steps, 3_590_000)
        self.assertEqual(self.config.task_learning_rate, 1e-4)
        self.assertEqual(self.config.encoder_learning_rate, 1e-5)

    def test_each_parameter_appears_once_and_frozen_encoder_is_included(self):
        groups = build_adamw_parameter_groups(self.model, self.config)
        grouped = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len(list(self.model.parameters())))
        self.assertEqual(len({id(parameter) for parameter in grouped}), len(grouped))
        encoder_ids = {id(parameter) for parameter in self.model.encoder.parameters()}
        self.assertTrue(encoder_ids.issubset({id(parameter) for parameter in grouped}))

    def test_bias_and_layer_norm_are_no_decay(self):
        groups = build_adamw_parameter_groups(self.model, self.config)
        by_id = {
            id(parameter): group["weight_decay"]
            for group in groups
            for parameter in group["params"]
        }
        for module in self.model.modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                expected_no_decay = "bias" in parameter_name.lower() or isinstance(module, nn.LayerNorm)
                self.assertEqual(by_id[id(parameter)] == 0.0, expected_no_decay)

    def test_adamw_group_lrs_and_decay_match_protocol(self):
        optimizer = build_adamw(self.model, self.config)
        actual = {
            group["name"]: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups
        }
        self.assertEqual(actual["encoder_decay"], (1e-5, 0.01))
        self.assertEqual(actual["encoder_no_decay"], (1e-5, 0.0))
        self.assertEqual(actual["task_decay"], (1e-4, 0.01))
        self.assertEqual(actual["task_no_decay"], (1e-4, 0.0))
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertFalse(optimizer.defaults["amsgrad"])

    def test_encoder_parameter_updates_after_it_is_unfrozen(self):
        optimizer = build_adamw(self.model, self.config)
        parameter = next(self.model.encoder.parameters())
        before = parameter.detach().clone()
        parameter.requires_grad = True
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertFalse(torch.equal(parameter, before))
