# TorchTNT

### Utils

* all_gather_tensors, barrier, get_global_rank, get_local_rank, get_process_group_backend_from_device, get_world_size, copy_data_to_device, ==get_device_from_env==, record_data_in_stream, sync_bool
* ==attach_oom_observer, is_out_of_cpu_memory, is_out_of_cuda_memory, is_out_of_memory_error, log_memory_snapshot, measure_rss_deltas==
* close_progress_bar, create_progress_bar, update_progress_bar
* get_filesystem, is_windows, days_to_secs
* ==get_module_summary, ModuleSummary, get_summary_table, prune_module_summary, get_tensor_size_bytes==
* ==get_nvidia_smi_gpu_stats, GPUStats, get_psutil_cpu_stats, CPUStats==
* get_python_version, get_torch_version
* ==get_timer_summary, Timer, log_elapsed_time==
* ==init_from_env==, prepare_ddp, prepare_fsdp, maybe_enable_tf32, seed
* rank_zero_critical, rank_zero_debug, rank_zero_error, rank_zero_info, rank_zero_print, rank_zero_warn
* transfer_batch_norm_stats, transfer_weights
* Meta only: ==get_runtime_env==, 

### Loggers

* CSVLogger
* InMemoryLogger
* JSONLogger
* TensorBoardLogger

### Callbacks

* BaseCSVWriter - predict callback
* GarbageCollector
* Lambda
* LearningRateMonitor
* ModuleSummary
* PyTorchProfiler
* SystemResourcesMonitor
* TensorBoardParameterMonitor
* TorchSnapshotSaver
* TQDMProgressBar
* TrainProgressMonitor

### Units

#### TrainUnit

* `on_train_start(state: State) -> None`
* `on_train_epoch_start(state: State) -> None`
* `ABSTRACT train_step(state: State, data, TTrainData) -> Any`
* `on_train_epoch_end(state: State) -> None`
* `on_train_end(state: State) -> None`

#### EvalUnit

* `on_eval_start(state: State) -> None`
* `on_eval_epoch_start(state: State) -> None`
* `ABSTRACT eval_step(state: State, data: TEvalData) -> Any`
* `on_eval_epoch_end(state: State) -> None`
* `on_eval_end(state: State) -> None`

#### PredictUnit

* `on_predict_start(state: State) -> None`
* `on_predict_epoch_start(state: State) -> None`
* `ABSTRACT predict_step(state: State, data: TPredictData) -> Any`
* `on_predict_epoch_end(state: State) -> None`
* `on_predict_end(state: State) -> None`

#### AutoUnit

* `ABSTRACT compute_loss(state: State, data: TData) -> Tuple[Tensor, Any]`
* `ABSTRACT configure_optimizers_and_lr_schedulers(module: Module) -> Tuple[Optimizer, Optional[LRScheduler]]`
* `move_data_to_device(state: State, data: TData, non_blocking: bool) -> TData`



* `train_step(state: State, data: Iterator[TData]) -> Tuple[Tensor, Any]`

  Main implementation here. No point overriding this.

  

* `on_train_step_end(state: State, data: TData, step: int, loss: Tensor, outputs: Any) -> None`

​	Empty implementation. Can override if I want.



* `on_train_epoch_end(state: State) -> None`

​	Some implementation only in case of Stochastic Weight Averaged (SWA) model. Otherwise is noon.



* `eval_step(state: State, data: TData) -> Tuple[Tensor, Any]`

​	Main implementation here. No point overriding this.



* `on_eval_step_end(state: State, data: TData, step: int, loss: Tensor, outputs: Any) -> None`

​	Empty implementation. Can override if I want.



* `on_train_end(state: State) -> None`

​	Some implementation only in case of Stochastic Weight Averaged (SWA) model. Otherwise is noon.

#### AutoPredictUnit

Useless



## Improvements

* Better support for metrics in `AutoUnit`
* Better support for progress bar in `AutoUnit`
* `restore_from_latest` is confusing
