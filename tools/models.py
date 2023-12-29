import torch
import torch.nn as nn
import torch.nn.init as init
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
        
        # 위치 및 방향 정보 처리를 위한 완전 연결 계층
        self.position_fc = nn.Sequential(
            nn.Linear(3, 64),  # 위치 및 방향 정보를 위한 입력 레이어 (x, y, yaw)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        # 융합을 위한 간단한 컨볼루션 레이어 수정
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 완전 연결 계층 추가
        self.fc_layer = nn.Sequential(
            nn.Linear(640, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # 모든 모듈을 순회하며 He 초기화 적용
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)

    def forward(self, rgb_image, depth_image, position):
        # 각 이미지에 대한 특징 추출
        rgb_feature = self.rgb_resnet(rgb_image)
        depth_feature = self.depth_resnet(depth_image)

        # 특징들을 채널 차원을 따라 결합
        cat_feature = torch.cat((rgb_feature, depth_feature), dim=1)

        # 융합된 특징을 컨볼루션 레이어에 통과시켜 최종 특징 추출
        fused_feature = self.fusion_conv(cat_feature)

        # 완전 연결 계층을 통과시켜 최종 출력 생성
        fused_feature = fused_feature.view(fused_feature.size(0), -1)
        
        # 위치 및 방향 정보 처리
        position_feature = self.position_fc(position)

        # 위치 정보와 이미지 특징 결합
        combined_feature = torch.cat((fused_feature, position_feature), dim=1)

        # 최종 출력 생성
        output = self.fc_layer(combined_feature)
        return output
    
class LowerComplexityRGBDepthFusionNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LowerComplexityRGBDepthFusionNet, self).__init__()

        # ResNet 모델 선택
        base_model = models.resnet18(pretrained=pretrained)

        # RGB 및 깊이 이미지를 위한 ResNet 복사본 생성
        self.rgb_resnet = nn.Sequential(*(list(base_model.children())[:-2]))
        self.depth_resnet = nn.Sequential(*(list(base_model.children())[:-2]))

        # 위치 정보 처리를 위한 완전 연결 계층
        self.position_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        # 융합을 위한 컨볼루션 레이어
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_model.fc.in_features * 2, 1024, kernel_size=1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 최종 분류를 위한 완전 연결 계층
        self.fc_layer = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # He 초기화 적용
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)

    def forward(self, rgb_image, depth_image, position):
        # RGB 및 깊이 이미지 특징 추출
        rgb_feature = self.rgb_resnet(rgb_image)
        depth_feature = self.depth_resnet(depth_image)

        # 특징 융합
        cat_feature = torch.cat((rgb_feature, depth_feature), dim=1)
        fused_feature = self.fusion_conv(cat_feature)

        # 위치 정보 처리
        position_feature = self.position_fc(position)

        # 특징 결합 및 최종 출력 생성
        fused_feature = fused_feature.view(fused_feature.size(0), -1)
        combined_feature = torch.cat((fused_feature, position_feature), dim=1)
        output = self.fc_layer(combined_feature)
        return output

class AttentionFusionRGBDepthFusionNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(AttentionFusionRGBDepthFusionNet, self).__init__()
        # 고급 ResNet 모델 사용 (예: ResNet101 또는 ResNet152)
        self.rgb_resnet = models.resnet152(pretrained=pretrained)
        self.depth_resnet = models.resnet152(pretrained=pretrained)

        # 마지막 분류 레이어 제거
        self.rgb_resnet = nn.Sequential(*(list(self.rgb_resnet.children())[:-2]))
        self.depth_resnet = nn.Sequential(*(list(self.depth_resnet.children())[:-2]))

        # 위치 정보 처리를 위한 완전 연결 계층
        self.position_fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # 고급 융합 전략 (예: 주의 메커니즘 사용)
        self.fusion_layer = AttentionFusion(2048 * 2, 256)  # 주의 메커니즘 적용

        # 최종 분류를 위한 완전 연결 계층
        self.fc_layer = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 과적합 방지를 위한 드롭아웃
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # He 초기화 적용
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)

    def forward(self, rgb_image, depth_image, position):
        # RGB 및 깊이 이미지 특징 추출
        rgb_feature = self.rgb_resnet(rgb_image)
        depth_feature = self.depth_resnet(depth_image)

        # 특징 융합
        fused_feature = self.fusion_layer(rgb_feature, depth_feature)

        # 위치 정보 처리
        position_feature = self.position_fc(position)

        # 특징 결합 및 최종 출력 생성
        combined_feature = torch.cat((fused_feature, position_feature), dim=1)
        output = self.fc_layer(combined_feature)
        return output

class AttentionFusion(nn.Module):
    """주의 메커니즘을 이용한 특징 융합 레이어"""
    def __init__(self, input_dim, output_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1)
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, rgb_feature, depth_feature):
        # 특징 맵 평탄화
        rgb_feature = rgb_feature.view(rgb_feature.size(0), -1)
        depth_feature = depth_feature.view(depth_feature.size(0), -1)

        # 특징 결합
        combined_feature = torch.cat((rgb_feature, depth_feature), dim=1)

        # 주의 가중치 계산
        attention_weights = torch.softmax(self.attention(combined_feature), dim=1)

        # 가중치 적용 및 특징 융합
        fused_feature = self.fc(combined_feature)
        fused_feature = fused_feature * attention_weights

        return fused_feature