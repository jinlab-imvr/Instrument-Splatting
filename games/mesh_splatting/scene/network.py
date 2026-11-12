import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def camera_to_ndc(X_c, K, W, H, z_near=0.010, z_far=0.3):  # unit: m
    # X_c: (B, N, 3) or (B, 3)
    # K: (B, 3, 3)
    # W, H: int
    # Output: (B, N, 3) or (B, 3)
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    Z_c = X_c[..., 2]
    X = X_c[..., 0]
    Y = X_c[..., 1]

    # Ensure fx, fy, cx, cy have the right shape for broadcasting
    while fx.dim() < X.dim():
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)

    u = fx * (X / Z_c) + cx
    v = fy * (Y / Z_c) + cy

    x_ndc = 2 * u / W - 1
    y_ndc = 1 - 2 * v / H
    z_ndc = (2 * Z_c - (z_far + z_near)) / (z_far - z_near)

    return torch.stack([x_ndc, y_ndc, z_ndc], dim=-1)

def ndc_to_camera(ndc, K, W, H, z_near=0.010, z_far=0.3):  # unit: m
    # ndc: (B, N, 3) or (B, 3)
    # K: (B, 3, 3)
    # W, H: int
    # Output: (B, N, 3) or (B, 3)
    x_ndc = ndc[..., 0]
    y_ndc = ndc[..., 1]
    z_ndc = ndc[..., 2]

    u = (x_ndc + 1) * W / 2
    v = (1 - y_ndc) * H / 2

    Z_c = 0.5 * ((z_ndc * (z_far - z_near)) + (z_far + z_near))

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    # Ensure fx, fy, cx, cy have the right shape for broadcasting
    while fx.dim() < u.dim():
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)

    X_c = (u - cx) * Z_c / fx
    Y_c = (v - cy) * Z_c / fy

    return torch.stack([X_c, Y_c, Z_c], dim=-1)

class KeypointUpSample(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        #nn.init.uniform_(self.kps_score_lowres.weight)
        #nn.init.uniform_(self.kps_score_lowres.bias)
        self.up_scale = 1
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return torch.nn.functional.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
        )


class SpatialSoftArgmax(nn.Module):
    """
    The spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, h, device=device),
                    torch.linspace(-1, 1, w, device=device),
                    indexing='ij',
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, h, device=device),
                torch.arange(0, w, device=device),
                indexing='ij',
            )
        )

    def forward(self, x, mask=None):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        if mask is not None:
            # resize the mask to match the input shape
            mask = F.interpolate(mask, size=(h, w), mode='nearest')
            
        # use relu to process x
        x = F.relu(x)
        if mask is not None:
            x = x * mask
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        yc, xc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C, 2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2), softmax
    
class KeyPointNet(nn.Module):
    def __init__(self, n_kp, height, width, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(KeyPointNet, self).__init__()

        self.lim = lim
        self.height = height
        self.width = width
        k = n_kp

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)
        deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes

        self.backbone = torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])

        self.read_out = KeypointUpSample(2048, k)

        self.spatialsoftargmax = SpatialSoftArgmax()


    def forward(self, img, mask=None):
        input_shape = img.shape[-2:]
        

        resnet_out = self.backbone(img)['out']  # (B, 2048, H//8, W//8)
        

        # keypoint prediction branch
        heatmap = self.read_out(resnet_out) # (B, k, H//4, W//4)
        # heatmap = F.interpolate(heatmap, (self.height, self.width), mode='bilinear', align_corners=False)
        keypoints, softmax = self.spatialsoftargmax(heatmap, mask)
        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = resnet_out.device)
        scale = torch.tensor([self.width // 2, self.height // 2], device = resnet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        return keypoints, softmax
    
class PoseRegressionHead(nn.Module):
    def __init__(self, in_channels=2048, out_dim=10):
        super(PoseRegressionHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 2048, kernel_size=3, stride=2, padding=1),  # (B, 1024, H/2, W/2)
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1),  # (B, 512, H/4, W/4)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim)  # Output: (B, out_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    
class KeyPointPoseNet(nn.Module):
    def __init__(self, n_kp, height, width, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(KeyPointPoseNet, self).__init__()

        self.lim = lim
        self.height = height
        self.width = width
        self.pose_head = PoseRegressionHead(in_channels=2048, out_dim=10)
        self.rotation_activation = torch.nn.functional.normalize
        self.joint_activation = torch.nn.functional.sigmoid
        k = n_kp

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)
        deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes

        self.backbone = torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])

        self.read_out = KeypointUpSample(2048, k)

        self.spatialsoftargmax = SpatialSoftArgmax()

    def forward(self, img, mask=None):
        input_shape = img.shape[-2:]
        
        resnet_out = self.backbone(img)['out']  # (B, 2048, H//8, W//8)
        output = self.pose_head(resnet_out)
        quat = self.rotation_activation(output[:, :4], dim=1)  # Normalize the quaternion vector
        joint_angles = self.joint_activation(output[:, -3:])  # Normalize joint angles
        joint_angles = joint_angles * 3.141592653589793 - 3.141592653589793 / 2  # Scale to [-pi/2, pi/2]
        
        translation = self.joint_activation(output[:, 4:7])  # Extract translation vector
        translation = translation * 2 - 1   # NDC coordinates in [-1, 1]
        pose_output = torch.cat([quat, translation, joint_angles], dim=1)  # Concatenate quaternion, translation, and joint angles
        # keypoint prediction branch 
        heatmap = self.read_out(resnet_out) # (B, k, H//4, W//4)
        # heatmap = F.interpolate(heatmap, (self.height, self.width), mode='bilinear', align_corners=False)
        keypoints, softmax = self.spatialsoftargmax(heatmap, mask)
        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = resnet_out.device)
        scale = torch.tensor([self.width // 2, self.height // 2], device = resnet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        return keypoints, softmax, pose_output

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = ResidualConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ResidualConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ResidualConvBlock(128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        dec2 = self.upconv2(enc3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)