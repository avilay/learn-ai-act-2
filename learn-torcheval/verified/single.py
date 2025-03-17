import torch as t
import torcheval.metrics as metrics

N_EPOCHS = 2
STEPS_PER_EPOCH = 10
BATCH_SIZE = 3


def gen_processed_batch(batch_size: int) -> tuple[t.Tensor, t.Tensor]:
    outputs = t.randint(low=0, high=4, size=(batch_size,))
    targets = t.randint(low=0, high=4, size=(batch_size,))
    return outputs, targets


def main():
    all_outputs, all_targets = [], []
    metric = metrics.MulticlassAccuracy()
    for epoch in range(N_EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            outputs, targets = gen_processed_batch(BATCH_SIZE)
            all_outputs.append(outputs)
            all_targets.append(targets)
            metric.update(outputs, targets)

        # Calculate the epoch level metric
        acc = metric.compute()
        print(f"Epoch {epoch}: Accuracy is {acc:.2f}")
        # Get the metric ready for the next epoch
        metric.reset()

        outputs = t.concatenate(all_outputs)
        targets = t.concatenate(all_targets)
        exp_acc = t.sum(outputs == targets) / outputs.shape[0]
        print(f"Epoch {epoch}: Expected Accuracy is {exp_acc:.8f}")
        all_outputs, all_targets = [], []


if __name__ == "__main__":
    main()
