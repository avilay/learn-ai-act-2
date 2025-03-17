"""Run Instructions on M1 MacOS using docker

Start the master node -
```shell
$ docker container run \
  --platform linux/amd64 \
  --mount type=bind,source="$(pwd)",target=/workspace/app,readonly \
  -dit \
  --name master \
  --env-file ./docker_rank_0.env \
  pytorch/pytorch
```

Get the IP address of the master node -
```shell
docker network inspect bridge | jq '.[0].Containers | to_entries | .[] | select(.value.Name == "master")'
```

Replace the ??? with this IP address in docker_rank_1.env.

Start the trainer node -
```shell
$ docker container run \
  --platform linux/amd64 \
  --mount type=bind,source="$(pwd)",target=/workspace/app,readonly \
  -dit \
  --name trainer \
  --env-file ./docker_rank_1.env \
  pytorch/pytorch
```

Log into the master/trainer and start the program-
```shell
docker attach master
# pip install torcheval
# cd app
# python multi.py
```
"""

import torch as t
import torch.distributed as dist
import torcheval.metrics as metrics
from torcheval.metrics.toolkit import sync_and_compute

N_EPOCHS = 2
STEPS_PER_EPOCH = 10
BATCH_SIZE = 3


def gen_processed_batch(batch_size: int) -> tuple[t.Tensor, t.Tensor]:
    outputs = t.randint(low=0, high=4, size=(batch_size,))
    targets = t.randint(low=0, high=4, size=(batch_size,))
    return outputs, targets


def main():
    print("Starting process group")
    dist.init_process_group("gloo")
    metric = metrics.MulticlassAccuracy()
    all_outputs, all_targets = [], []

    print("Starting training")
    for epoch in range(N_EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            outputs, targets = gen_processed_batch(BATCH_SIZE)
            all_outputs.append(outputs)
            all_targets.append(targets)
            metric.update(outputs, targets)

        my_outputs = t.concatenate(all_outputs)
        my_targets = t.concatenate(all_targets)

        if dist.get_rank() == 0:
            trainer_outputs = t.empty_like(my_outputs)
            trainer_targets = t.empty_like(my_targets)
            dist.recv(trainer_outputs, src=1)
            dist.recv(trainer_targets, src=1)
            outputs = t.concatenate((my_outputs, trainer_outputs))
            targets = t.concatenate((my_targets, trainer_targets))
            exp_acc = t.sum(outputs == targets) / outputs.shape[0]
            print(f"Epoch {epoch}: Expected accuracy is {exp_acc:.3f}")
        else:
            dist.send(my_outputs, dst=0)
            dist.send(my_targets, dst=0)

        all_outputs, all_targets = [], []

        # Compute end of epoch metrics on all the ranks
        acc = sync_and_compute(metric)
        # Get the metric ready for the next epoch on all the ranks
        metric.reset()

        if dist.get_rank() == 0:
            print(f"Epoch {epoch}: Accuracy is {acc:.3f}")


if __name__ == "__main__":
    main()
