# Checkpointing
A training checkpoint should have the following states saved -
  * random generator - verified.
  * model - verified.
  * data loader - does not work, tried just checkpointing the rng state and then the sampler, but none of that works.
  * optimizer
  * learning rate scheduler
  * current epoch and local step number
  * global step number

And the following, though not strictly states, are useful to have around when resuming from a checkpoint
  * precision used in mixed precision training
  * hyperparameters

torchsnapshot does not work on M1 because for some reason it has a hard depepdency on c10 even though I am not using torch.distributed anywhere in my code. I can work around this by writing my own single device snapshotter and a factory that will instantiate the correct snapshotter at runtime. That way my code will be portable across M1 and other platforms. The other method was to go fix torchsnapshot, but it is not worth my time.

## Experiment Design
For all these experiments set the manual seed first. It will be good to do these experiments where the artifacts are saved in one process but are restored in a different process, so notebooks are out.

### Random Generator

Just pickle the random generator state when saving the checkpoint and then compare it with the restored random generator.

### Model

Take a small model, say a $3 \times 1$ linear layer and add a tensor to it. Pickle the tensor (not the full layer) and compare it with the restored layer.

### Data Loader

Create a simple dataset that returns index numbers as tensor data, and then instantiate a data loader with this dataset and shuffle set to true. Iterate a couple of times and then save the checkpoint. Continue to iterate the saved data loader and note the order in which the indexes/tensors appear. Restore the checkpointed data loader and start iterating. It should start from the same point it was saved.