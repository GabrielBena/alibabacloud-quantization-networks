import torch.nn.functional as F
import torch


def train(model, device, train_loader, optimizer, epoch, **kwargs):
    model.train()
    train_losses, train_accs = [], []
    if hasattr(model, "quant"):
        model.set_training(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if kwargs.get("adjust_T", False):
            model.set_temperature(
                kwargs.get("Ts")[(epoch - 1) * len(train_loader) + batch_idx]
            )

        desc = str(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )

        train_losses.append(loss.item())
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        train_accs.append(correct / len(data))

        if kwargs.get("pbar", None):
            kwargs["pbar"].set_description(desc)

    return train_losses, train_accs, desc


def test(model, device, test_loader, pbar=None):
    model.eval()
    if hasattr(model, "quant"):
        model.set_training(False)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    desc = str(
        "| Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    if pbar is not None:
        pbar.set_description(desc)

    return test_loss, correct / len(test_loader.dataset), desc
