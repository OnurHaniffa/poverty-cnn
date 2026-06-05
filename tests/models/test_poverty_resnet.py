import torch
from poverty_cnn.models.poverty_resnet import PovertyResNet


def test_forward_shape():
    model = PovertyResNet(in_channels=8)
    x = torch.randn(2, 8, 224, 224)
    out = model(x)
    assert out.shape == (2,)          # scalar regression per sample


def test_dropout_present_for_mc_dropout():
    # a Dropout layer must exist (kept active at inference for MC-dropout later)
    model = PovertyResNet(in_channels=8, dropout=0.2)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_overfit_tiny_batch():
    # the model must be able to drive loss toward 0 on a handful of samples;
    # if it can't, gradients/wiring are broken — catch it before any real run.
    torch.manual_seed(0)
    model = PovertyResNet(in_channels=8)
    x = torch.randn(8, 8, 224, 224)
    y = torch.randn(8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    model.train()
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.25 * first   # loss collapsed -> it learns
