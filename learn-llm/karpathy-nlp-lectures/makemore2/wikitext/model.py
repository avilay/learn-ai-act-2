import torch as t
import lightning as L
from typing import Any
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT


class LangModel(t.nn.Module):
    def __init__(
        self, *, context_len: int, vocab_len: int, hidden_len: int, emb_dim: int
    ) -> None:
        super().__init__()
        self.emb = t.nn.Embedding(num_embeddings=vocab_len, embedding_dim=emb_dim)
        self.fc1 = t.nn.Linear(
            in_features=context_len * emb_dim, out_features=hidden_len
        )
        self.relu = t.nn.ReLU()
        self.fc2 = t.nn.Linear(in_features=hidden_len, out_features=vocab_len)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: 2D matrix of int64s of shape m x context_len
        return: m-element 1-D tensor array of logits
        """
        x = self.emb(x)
        dim = self.fc1.in_features
        x = x.view(-1, dim)
        # x = x.view(-1, t.prod(t.tensor(x.shape[1:])))
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits.squeeze()


class LangNet(L.LightningModule):
    def __init__(
        self,
        *,
        context_len: int,
        vocab_len: int,
        hidden_len: int,
        emb_dim: int,
        learning_rate: float,
        learning_rate_exp_decay: float,
        **other_hparams: Any
    ) -> None:
        super().__init__()
        self.model = LangModel(
            context_len=context_len,
            vocab_len=vocab_len,
            hidden_len=hidden_len,
            emb_dim=emb_dim,
        )
        self.loss_fn = t.nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self._losses: list[float] = []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = t.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore
        return {
            "optimizer": optim,
            "lr_scheduler": t.optim.lr_scheduler.ExponentialLR(
                optimizer=optim, gamma=self.hparams.learning_rate_exp_decay  # type: ignore
            ),
        }

    def training_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        self._losses.append(loss.detach().item())
        self.log("loss", loss.detach().item())
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = t.tensor(self._losses).mean()
        self.log("avg_loss", avg_loss)
