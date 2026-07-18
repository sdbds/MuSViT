import copy
import torch
import random
import wandb
import lightning.pytorch as L
from lightning.pytorch.trainer.states import TrainerFn
from loguru import logger

from .eval.eval_functions import (
    canonical_text,
    canonicalize_prediction_ids,
    canonicalize_target_ids,
    compute_canonical_metrics,
)

from .optimization import (
    AdamWWSDConfig,
    build_adamw,
    build_wsd_scheduler,
    optimizer_protocol_metadata,
    same_typed_value,
    validate_adamw_wsd_resume_state,
)


ENCODER_TRAINING_MODES = frozenset({"fine_tune", "linear_probe"})
SAMPLES_SEEN_CHECKPOINT_KEY = "full_page_omr_samples_seen"


def _validate_non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class SMTPP_Trainer(L.LightningModule):
    def __init__(self, smt_config, smt_model, encoder_training_mode="fine_tune",
                 encoder_unfreeze_step=None, curriculum_step_offset=0,
                 batch_size=1, accumulate_grad_batches=1,
                 enforce_checkpoint_protocol=True,
                 run_protocol_version="full_page_omr_adamw_wsd_4m_v1",
                 optimizer_protocol="adamw_wsd_v1",
                 task_learning_rate=1e-4,
                 encoder_learning_rate=1e-5,
                 weight_decay=0.01,
                 max_steps=4_000_000,
                 wsd_warmup_steps=10_000,
                 wsd_decay_steps=400_000,
                 wsd_warmup_type="linear",
                 wsd_decay_type="cosine",
                 wsd_min_lr_ratio=0.0,
                 protocol_snapshot=None):
        super().__init__()
        if (
            not isinstance(encoder_training_mode, str)
            or encoder_training_mode not in ENCODER_TRAINING_MODES
        ):
            raise ValueError(
                f"encoder_training_mode must be one of {sorted(ENCODER_TRAINING_MODES)}, "
                f"got {encoder_training_mode!r}"
            )
        if encoder_unfreeze_step is not None and (
            isinstance(encoder_unfreeze_step, bool)
            or not isinstance(encoder_unfreeze_step, int)
            or encoder_unfreeze_step < 0
        ):
            raise ValueError("encoder_unfreeze_step must be a non-negative integer or None")
        if (
            isinstance(curriculum_step_offset, bool)
            or not isinstance(curriculum_step_offset, int)
            or curriculum_step_offset < 0
        ):
            raise ValueError("curriculum_step_offset must be a non-negative integer")
        batch_size = _validate_positive_integer(batch_size, "batch_size")
        accumulate_grad_batches = _validate_positive_integer(
            accumulate_grad_batches,
            "accumulate_grad_batches",
        )

        self.model = smt_model
        self.padding_token = smt_config.padding_token
        self.encoder_training_mode = encoder_training_mode
        self.encoder_unfreeze_step = encoder_unfreeze_step
        self.curriculum_step_offset = curriculum_step_offset
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.enforce_checkpoint_protocol = enforce_checkpoint_protocol
        self.samples_seen = 0
        self.optimizer_config = AdamWWSDConfig(
            protocol=optimizer_protocol,
            task_learning_rate=task_learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            weight_decay=weight_decay,
            max_steps=max_steps,
            warmup_steps=wsd_warmup_steps,
            decay_steps=wsd_decay_steps,
            warmup_type=wsd_warmup_type,
            decay_type=wsd_decay_type,
            min_lr_ratio=wsd_min_lr_ratio,
        )
        if protocol_snapshot is not None and not isinstance(protocol_snapshot, dict):
            raise TypeError("protocol_snapshot must be a dictionary")
        self.protocol_snapshot = copy.deepcopy(protocol_snapshot)

        self.preds = []
        self.grtrs = []

        self.save_hyperparameters(ignore=["smt_model", "enforce_checkpoint_protocol"])
        self.hparams["protocol_snapshot"] = copy.deepcopy(self.protocol_snapshot)
        self.hparams.update({
            "optimizer": "AdamW",
            "optimizer_implementation": "torch.optim.AdamW",
            "torch_version": str(torch.__version__),
            "optimizer_betas": list(self.optimizer_config.betas),
            "optimizer_eps": self.optimizer_config.eps,
            "optimizer_amsgrad": self.optimizer_config.amsgrad,
            "wsd_num_cycles": self.optimizer_config.num_cycles,
        })
        self.model.freeze_encoder()
        self.unfrozen_vision = False

    def configure_optimizers(self):
        optimizer = build_adamw(self.model, self.optimizer_config)
        scheduler = build_wsd_scheduler(optimizer, self.optimizer_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "wsd",
            },
        }

    def optimizer_protocol_metadata(self):
        return optimizer_protocol_metadata(self.model, self.optimizer_config)

    def _validate_optimizer_checkpoint_identity(self, hyper_parameters):
        if self.protocol_snapshot is None:
            raise ValueError(
                "full resume requires a complete expected protocol_snapshot"
            )
        saved_snapshot = hyper_parameters.get("protocol_snapshot")
        if saved_snapshot is None:
            raise ValueError(
                "checkpoint is missing protocol_snapshot evidence required for full resume"
            )
        if not same_typed_value(saved_snapshot, self.protocol_snapshot):
            raise ValueError(
                "checkpoint protocol snapshot mismatch: "
                f"saved={saved_snapshot!r}, expected={self.protocol_snapshot!r}"
            )

    def _validate_optimizer_checkpoint_state(self, checkpoint):
        validate_adamw_wsd_resume_state(
            checkpoint,
            self.model,
            self.optimizer_config,
        )

    def _is_evaluation_checkpoint_load(self):
        trainer = self._trainer
        return trainer is not None and trainer.state.fn in {
            TrainerFn.TESTING,
            TrainerFn.VALIDATING,
            TrainerFn.PREDICTING,
        }
    
    def forward(self, input, last_preds):
        return self.model(input, last_preds)

    @property
    def curriculum_step(self):
        return self.curriculum_step_offset + self.samples_seen

    def _sync_encoder_trainability(self):
        curriculum_step = self.curriculum_step
        should_unfreeze = (
            self.encoder_training_mode == "fine_tune"
            and self.encoder_unfreeze_step is not None
            and curriculum_step >= self.encoder_unfreeze_step
        )
        if should_unfreeze and not self.unfrozen_vision:
            self.model.unfreeze_encoder()
            self.unfrozen_vision = True
            logger.info(
                "Unfroze vision encoder after {} consumed samples "
                "(curriculum step: {}, boundary: {})",
                self.samples_seen,
                curriculum_step,
                self.encoder_unfreeze_step,
            )
        elif not should_unfreeze and self.unfrozen_vision:
            self.model.freeze_encoder()
            self.unfrozen_vision = False

    def on_load_checkpoint(self, checkpoint):
        if not self.enforce_checkpoint_protocol:
            return

        if self._is_evaluation_checkpoint_load():
            return

        hyper_parameters = checkpoint.get("hyper_parameters", {})
        if not isinstance(hyper_parameters, dict):
            raise ValueError("checkpoint hyper_parameters must be a dictionary")
        self._validate_optimizer_checkpoint_identity(hyper_parameters)
        self._validate_optimizer_checkpoint_state(checkpoint)

        checkpoint_mode = hyper_parameters.get("encoder_training_mode")
        if checkpoint_mode is None:
            if self.encoder_training_mode != "fine_tune":
                raise ValueError(
                    "legacy checkpoints may only be resumed with "
                    "encoder_training_mode='fine_tune'"
                )
            logger.warning(
                "Legacy checkpoint has no encoder_training_mode; using explicit mode {!r} "
                "with boundary {!r}",
                self.encoder_training_mode,
                self.encoder_unfreeze_step,
            )
        else:
            if checkpoint_mode != self.encoder_training_mode:
                raise ValueError(
                    "Checkpoint encoder_training_mode mismatch: "
                    f"checkpoint={checkpoint_mode!r}, requested={self.encoder_training_mode!r}"
                )

            if "encoder_unfreeze_step" not in hyper_parameters:
                raise ValueError("Checkpoint is missing encoder_unfreeze_step metadata")

            checkpoint_boundary = hyper_parameters["encoder_unfreeze_step"]
            if checkpoint_boundary != self.encoder_unfreeze_step:
                raise ValueError(
                    "Checkpoint encoder_unfreeze_step mismatch: "
                    f"checkpoint={checkpoint_boundary!r}, requested={self.encoder_unfreeze_step!r}"
                )

            checkpoint_offset = hyper_parameters.get("curriculum_step_offset", 0)
            if checkpoint_offset != self.curriculum_step_offset:
                raise ValueError(
                    "Checkpoint curriculum_step_offset mismatch: "
                    f"checkpoint={checkpoint_offset!r}, requested={self.curriculum_step_offset!r}"
                )

        if SAMPLES_SEEN_CHECKPOINT_KEY in checkpoint:
            self.samples_seen = _validate_non_negative_integer(
                checkpoint[SAMPLES_SEEN_CHECKPOINT_KEY],
                SAMPLES_SEEN_CHECKPOINT_KEY,
            )
            return

        if self.batch_size != 1 or self.accumulate_grad_batches != 1:
            raise ValueError(
                "Legacy checkpoint has no samples_seen counter; migration is only exact "
                "with batch_size=1 and accumulate_grad_batches=1"
            )
        if "global_step" not in checkpoint:
            return
        self.samples_seen = _validate_non_negative_integer(
            checkpoint.get("global_step"),
            "legacy checkpoint global_step used for samples_seen",
        )
        logger.warning(
            "Legacy checkpoint has no samples_seen counter; inferred {} from global_step "
            "under the batch_size=1, accumulate_grad_batches=1 contract",
            self.samples_seen,
        )

    def on_save_checkpoint(self, checkpoint):
        if self.protocol_snapshot is None:
            raise ValueError("checkpoint save requires a complete protocol_snapshot")
        checkpoint[SAMPLES_SEEN_CHECKPOINT_KEY] = _validate_non_negative_integer(
            self.samples_seen,
            SAMPLES_SEEN_CHECKPOINT_KEY,
        )
        hyper_parameters = checkpoint.setdefault("hyper_parameters", {})
        if not isinstance(hyper_parameters, dict):
            raise ValueError("checkpoint hyper_parameters must be a dictionary")
        hyper_parameters["protocol_snapshot"] = copy.deepcopy(
            self.protocol_snapshot
        )

    def training_step(self, batch):
        self._sync_encoder_trainability()

        x, di, y = batch[:3]
        outputs = self.model(x, di[:, :-1], labels=y)
        loss = outputs.loss
        self.log('loss', loss, on_step=False, on_epoch=True, batch_size=x.shape[0], prog_bar=True)
        self.samples_seen += int(x.shape[0])
        
        return loss
        
    
    def validation_step(self, val_batch):
        x, dec_in, y = val_batch
        del dec_in
        generation = self.model.generate_token_ids(input=x)
        self.preds.append(
            canonicalize_prediction_ids(
                generation.token_ids,
                self.model.i2w,
                maxlen=self.model.maxlen,
            )
        )
        self.grtrs.append(
            canonicalize_target_ids(y.squeeze(0).tolist(), self.model.i2w)
        )
        
    def on_validation_epoch_end(self, metric_name="val"):
        cer, ser, ler = compute_canonical_metrics(self.preds, self.grtrs)
        
        random_index = random.randint(0, len(self.preds)-1)
        predtoshow = canonical_text(self.preds[random_index])
        gttoshow = canonical_text(self.grtrs[random_index])
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")
        
        self.log(f'{metric_name}_CER_v2', cer, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_SER_v2', ser, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_LER_v2', ler, on_epoch=True, prog_bar=True)
        
        self.preds = []
        self.grtrs = []
        
        return ser
        
    
    def test_step(self, test_batch):
        return self.validation_step(test_batch)
    
    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end("test")
