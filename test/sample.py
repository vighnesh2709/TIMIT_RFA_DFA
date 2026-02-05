import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function


model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

prof = profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_flops=True
)


for i in range(5):
    if i == 0:
        prof.start()

    model(inputs)

    if i == 0:
        prof.stop()

print(prof.key_averages().table(sort_by = "flops"))
