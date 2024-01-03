import torch

def train(model, device, train_loader, optimizer, criterion, metric, running_norms=None):
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)

    # compute train acc
    train_metric = metric(target.detach().cpu().numpy(), output.detach().cpu().numpy())
    
    # compute train loss
    loss = criterion(output, target)
    train_loss = loss.item()
    loss.backward()

    if running_norms is not None:
        gradient_norms = optimizer.step(running_norms)
        gradient_norms_sq = gradient_norms * gradient_norms
        return train_loss, train_metric, gradient_norms_sq
    
    else:
        optimizer.step()
        return train_loss, train_metric
    

def test(model, device, test_loader, criterion, metric):
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)

        test_metric = metric(target.detach().cpu().numpy(), output.detach().cpu().numpy())

        test_loss = criterion(output, target).item()

    return test_loss, test_metric
