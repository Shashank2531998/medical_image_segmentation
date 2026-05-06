import torch


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: dict, max_epochs: int):
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 3e-5)
    momentum = cfg.get("momentum", 0.99)
    poly_power = cfg.get("poly_power", 0.9)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=cfg.get("nesterov", True),
    )

    def poly_lr(epoch: int) -> float:
        progress = min(max(epoch / max(max_epochs, 1), 0.0), 1.0)
        return (1.0 - progress) ** poly_power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)
    return optimizer, scheduler
