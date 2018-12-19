import torch.optim as optim


def get_optimizer(model, config):
    # optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['decay'])
    # # optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.decay, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, threshold=1e-4)

    if config['bias_lr'] is None:
        optimizer = getattr(optim, config['type'])(model.parameters(), **config['optimizer_param'])
    else:
        param_groups = []
        for name, param in model.named_parameters():
            if 'bias' in name:
                param_groups += [{'params': param, 'lr': config['bias_lr']}]
            elif 'w_output' in name:
                param_groups += [{'params': param, 'lr': config['w_output_lr']}]
            elif 'w_inter' in name:
                param_groups += [{'params': param, 'lr': config['w_inter_lr']}]
            else:
                param_groups += [{'params': param}]
        optimizer = getattr(optim, config['type'])(param_groups, **config['optimizer_param'])

    lr_scheduler = None
    if config['lr_scheduler']:
        lr_scheduler = getattr(
            optim.lr_scheduler,
            config['lr_scheduler'], None)(optimizer, **config['lr_scheduler_param'])

    return optimizer, lr_scheduler
