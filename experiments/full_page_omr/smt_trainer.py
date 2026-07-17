import torch
import random
import wandb
import lightning.pytorch as L
from loguru import logger

from .eval.eval_functions import (
    canonical_text,
    canonicalize_prediction_ids,
    canonicalize_target_ids,
    compute_canonical_metrics,
)

from . import _globals


ENCODER_TRAINING_MODES = frozenset({"fine_tune", "linear_probe"})


class SMTPP_Trainer(L.LightningModule):
    def __init__(self, smt_config, smt_model, encoder_training_mode="fine_tune",
                 encoder_unfreeze_step=None, curriculum_step_offset=0,
                 enforce_checkpoint_protocol=True):
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

        self.model = smt_model
        self.padding_token = smt_config.padding_token
        self.encoder_training_mode = encoder_training_mode
        self.encoder_unfreeze_step = encoder_unfreeze_step
        self.curriculum_step_offset = curriculum_step_offset
        self.enforce_checkpoint_protocol = enforce_checkpoint_protocol

        self.preds = []
        self.grtrs = []

        self.save_hyperparameters(ignore=["smt_model", "enforce_checkpoint_protocol"])
        self.model.freeze_encoder()
        self.unfrozen_vision = False
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.parameters()), lr=_globals.learning_rate, amsgrad=False)
    
    def forward(self, input, last_preds):
        return self.model(input, last_preds)

    def _sync_encoder_trainability(self, step):
        curriculum_step = step + self.curriculum_step_offset
        should_unfreeze = (
            self.encoder_training_mode == "fine_tune"
            and self.encoder_unfreeze_step is not None
            and curriculum_step >= self.encoder_unfreeze_step
        )
        if should_unfreeze and not self.unfrozen_vision:
            self.model.unfreeze_encoder()
            self.unfrozen_vision = True
            logger.info(
                "Unfroze vision encoder at global step {} "
                "(curriculum step: {}, boundary: {})",
                step,
                curriculum_step,
                self.encoder_unfreeze_step,
            )
        elif not should_unfreeze and self.unfrozen_vision:
            self.model.freeze_encoder()
            self.unfrozen_vision = False

    def on_load_checkpoint(self, checkpoint):
        if not self.enforce_checkpoint_protocol:
            return

        hyper_parameters = checkpoint.get("hyper_parameters", {})
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
            return
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

    def training_step(self, batch):
        self._sync_encoder_trainability(int(self.global_step))

        x, di, y, = batch
        outputs = self.model(x, di[:, :-1], labels=y)
        loss = outputs.loss
        self.log('loss', loss, on_step=False, on_epoch=True, batch_size=x.shape[0], prog_bar=True)
        
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
