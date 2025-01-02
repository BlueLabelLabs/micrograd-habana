from micrograd.nn import MLP
from micrograd.utils import get_device_initial


def test_device():
    device = get_device_initial()

    model = MLP(2, [16, 16, 1], device=device)

    # check if the device is set correctly
    for p in model.parameters():  # check every parameter
        match device:
            case "cpu":
                assert p.device is None
            case _:
                assert p.device == device
