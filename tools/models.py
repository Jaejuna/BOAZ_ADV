import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MovePredictModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise Exception("Only ResNet series available.")

        self.fc_layer = nn.Sequential(
            nn.Linear(1000, 512), 
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layer(x)
        return x

class RGBDepthFusionNet(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", pretrained=True):
        super(RGBDepthFusionNet, self).__init__()
        if backbone == "resnet18":
            self.rgb_resnet = models.resnet18(pretrained=pretrained)
            self.depth_resnet = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.rgb_resnet = models.resnet34(pretrained=pretrained)
            self.depth_resnet = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.rgb_resnet = models.resnet50(pretrained=pretrained)
            self.depth_resnet = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.rgb_resnet = models.resnet101(pretrained=pretrained)
            self.depth_resnet = models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            self.rgb_resnet = models.resnet152(pretrained=pretrained)
            self.depth_resnet = models.resnet152(pretrained=pretrained)
        else:
            raise Exception("Only ResNet series available.")

        # 마지막 분류 레이어 제거 (특징만 추출하기 위함)
        self.rgb_resnet = nn.Sequential(*(list(self.rgb_resnet.children())[:-2]))
        self.depth_resnet = nn.Sequential(*(list(self.depth_resnet.children())[:-2]))

        # 융합을 위한 간단한 컨볼루션 레이어 수정
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1),  # 입력 채널을 4096으로 변경
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 완전 연결 계층 추가
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb_image, depth_image):
        # 각 이미지에 대한 특징 추출
        rgb_feature = self.rgb_resnet(rgb_image)
        depth_feature = self.depth_resnet(depth_image)

        # 특징들을 채널 차원을 따라 결합
        combined_feature = torch.cat((rgb_feature, depth_feature), dim=1)

        # 융합된 특징을 컨볼루션 레이어에 통과시켜 최종 특징 추출
        fused_feature = self.fusion_conv(combined_feature)

        # 완전 연결 계층을 통과시켜 최종 출력 생성
        fused_feature = fused_feature.view(fused_feature.size(0), -1)  # 배치 차원을 유지하며 나머지 차원을 flatten
        output = self.fc_layer(fused_feature)
        return output