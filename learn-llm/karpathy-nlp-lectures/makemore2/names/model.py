import torch as t
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


class CharLangModel(t.nn.Module):
    def __init__(
        self, *, context_len: int, vocab_len: int, hidden_dim: int, emb_dim: int
    ) -> None:
        super().__init__()
        self.emb = t.nn.Embedding(vocab_len, emb_dim)
        self.fc1 = t.nn.Linear(
            in_features=context_len * emb_dim, out_features=hidden_dim
        )
        self.fc2 = t.nn.Linear(in_features=hidden_dim, out_features=vocab_len)

    def forward(self, x) -> t.Tensor:
        x1 = self.emb(x)
        x2 = x1.view(-1, self.fc1.in_features)
        x3 = self.fc1(x2)
        x4 = t.tanh(x3)
        logits = self.fc2(x4)
        return logits


class CharLangNet(L.LightningModule):
    def __init__(
        self,
        *,
        context_len: int,
        vocab_len: int,
        hidden_dim: int,
        emb_dim: int,
        learning_rate: float,
        **other_hparams
    ) -> None:
        super().__init__()
        self.model = CharLangModel(
            context_len=context_len,
            vocab_len=vocab_len,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
        )
        self.loss_fn = t.nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self._losses: list[float] = []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return t.optim.SGD(params=self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def training_step(self, batch: t.Tensor, batch_idx: int) -> STEP_OUTPUT:
        inputs, targets = batch
        logits = self.model.forward(inputs)
        loss = self.loss_fn(logits, targets)
        self._losses.append(loss.detach().item())
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = t.tensor(self._losses).mean()
        self.log("avg_loss", avg_loss)
