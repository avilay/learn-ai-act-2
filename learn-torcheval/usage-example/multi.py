import click
import torch as t
import torch.distributed as dist
import torcheval.metrics as teval
from torcheval.metrics.toolkit import sync_and_compute
import os

N_EPOCHS = 2
STEPS_PER_EPOCH = 10
BATCH_SIZE = 3


def gen_processed_batch(batch_size: int) -> tuple[t.Tensor, t.Tensor]:
    outputs = t.randint(low=0, high=4, size=(batch_size,))
    targets = t.randint(low=0, high=4, size=(batch_size,))
    return outputs, targets


@click.command()
@click.option("--rank", type=int, help="Rank of the trainer")
def main(rank: int):
    os.environ["RANK"] = str(rank)
    print("Starting process group")
    dist.init_process_group("gloo")
    metric = teval.MulticlassAccuracy()

    print("Starting training")
    for epoch in range(N_EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            outputs, targets = gen_processed_batch(BATCH_SIZE)
            metric.update(outputs, targets)

    # Compute end of epoch metrics - run on all ranks
    acc = sync_and_compute(metric)
    # Get the metric ready for the next epoch - runs on all ranks
    metric.reset()

    if dist.get_rank() == 0:
        print(f"Epoch {epoch}: Accuracy is {acc:.3f}")


if __name__ == "__main__":
    main()
