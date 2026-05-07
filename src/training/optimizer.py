import torch


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: dict, max_epochs: int, iters_per_epoch: int | None = None):
    """
    Build SGD optimizer and polynomial LR scheduler.

    If `iters_per_epoch` is provided, the scheduler will operate per-iteration
    across `max_epochs * iters_per_epoch` steps. Otherwise it will step per-epoch.
    Returns (optimizer, scheduler, per_iteration_flag)
    """
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 3e-5)
    momentum = cfg.get("momentum", 0.99)
    poly_power = cfg.get("poly_power", 1.0)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=cfg.get("nesterov", True),
    )

    per_iteration = iters_per_epoch is not None and iters_per_epoch > 0

    if per_iteration:
        total_steps = int(max_epochs * iters_per_epoch)

        def poly_lr_step(current_step: int) -> float:
            progress = min(max(current_step / max(total_steps, 1), 0.0), 1.0)
            return (1.0 - progress) ** poly_power

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_step)
    else:
        def poly_lr_epoch(epoch: int) -> float:
            progress = min(max(epoch / max(max_epochs, 1), 0.0), 1.0)
            return (1.0 - progress) ** poly_power

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_epoch)

    return optimizer, scheduler, per_iteration
