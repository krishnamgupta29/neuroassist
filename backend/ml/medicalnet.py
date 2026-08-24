import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import logging
import os

logger = logging.getLogger(__name__)

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
        device=out.device, dtype=out.dtype
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, sample_input_D=128, sample_input_H=128, sample_input_W=128, num_seg_classes=3, shortcut_type='B'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_seg_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


# Set to True once a checkpoint is actually loaded. Everything downstream keys
# off this to decide whether a prediction may be presented as a finding.
WEIGHTS_LOADED = False

# Absolute or backend-relative path to a state_dict saved with torch.save().
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "neuroassist_resnet10.pth")
WEIGHTS_PATH = os.getenv("NEUROASSIST_WEIGHTS", DEFAULT_WEIGHTS)


def get_multiclass_model():
    """Build the 3-class ResNet-10 and load trained weights if configured.

    Without NEUROASSIST_WEIGHTS the returned network is randomly initialised.
    It still runs, but its softmax output carries no diagnostic meaning, so
    WEIGHTS_LOADED stays False and the API labels the result as a demo.
    """
    global WEIGHTS_LOADED
    model = resnet10(num_seg_classes=3)

    if not WEIGHTS_PATH:
        logger.warning(
            "NEUROASSIST_WEIGHTS is not set - running an UNTRAINED ResNet-10. "
            "Predictions are meaningless and are flagged as demo output."
        )
        return model

    if not os.path.exists(WEIGHTS_PATH):
        logger.error("NEUROASSIST_WEIGHTS=%s does not exist - staying untrained.", WEIGHTS_PATH)
        return model

    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        # Accept either a bare state_dict or a training checkpoint wrapping one.
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Checkpoint loaded with missing=%s unexpected=%s", missing, unexpected)
        WEIGHTS_LOADED = True
        logger.info("Loaded trained weights from %s", WEIGHTS_PATH)
    except Exception:
        logger.exception("Failed to load %s - staying untrained.", WEIGHTS_PATH)

    return model


def weights_loaded() -> bool:
    return WEIGHTS_LOADED
