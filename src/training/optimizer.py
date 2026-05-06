import torch


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: dict):
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.1))
    return optimizer, scheduler
