import torch as t
import torcheval.metrics as teval

N_EPOCHS = 2
STEPS_PER_EPOCH = 10
BATCH_SIZE = 3


def gen_processed_batch(batch_size: int) -> tuple[t.Tensor, t.Tensor]:
    outputs = t.randint(low=0, high=4, size=(batch_size,))
    targets = t.randint(low=0, high=4, size=(batch_size,))
    return outputs, targets


def main():
    metric = teval.MulticlassAccuracy()
    for epoch in range(N_EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            outputs, targets = gen_processed_batch(BATCH_SIZE)
            metric.update(outputs, targets)

    # Calculate the epoch level metric
    acc = metric.compute()
    print(f"Epoch {epoch}: Accuracy is {acc:.3f}")
    # Get the metric ready for the next epoch
    metric.reset()


if __name__ == "__main__":
    main()
