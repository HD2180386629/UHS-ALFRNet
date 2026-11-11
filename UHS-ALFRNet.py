import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. DropPath
# =========================================================

class DropPath(nn.Module):
    """
    Stochastic Depth.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# =========================================================
# 2. LayerNorm2d
# =========================================================

class LayerNorm2d(nn.Module):
    """
    LayerNorm for feature maps with shape [B, C, H, W].
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


# =========================================================
# 3. GRN: Global Response Normalization
# =========================================================

class GRN(nn.Module):
    """
    Global Response Normalization.
    It enhances inter-channel response competition.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


# =========================================================
# 4. Depthwise Separable Projection
# =========================================================

class DSConvProj(nn.Module):
    """
    Depthwise separable projection.
    Used as the unified input projection in each stem.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)

        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.norm = LayerNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# =========================================================
# 5. Prototype Attention
# =========================================================

class PrototypeAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4, proto_size=4, drop_path=0.0):
        super().__init__()
        assert channels % num_heads == 0, (f"channels={channels} must be divisible by num_heads={num_heads}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.proto_size = proto_size

        self.norm = LayerNorm2d(channels)

        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool2d((proto_size, proto_size))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x_norm = self.norm(x)

        b, c, h, w = x_norm.shape
        n = h * w


        # Query: [B, heads, N, head_dim]
        q = self.q(x_norm)
        q = q.flatten(2).transpose(1, 2)
        q = q.reshape(b, n, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # Prototype tokens for Key and Value: [B, heads, P, head_dim]
        proto = self.pool(x_norm)

        p = proto.shape[-2] * proto.shape[-1]

        kv = self.kv(proto)
        kv = kv.flatten(2).transpose(1, 2)
        kv = kv.reshape(b, p, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, n, c)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        out = self.proj(out)
        out = identity + self.drop_path(out)
        return out


# =========================================================
# 6. Unified Hybrid Stem Block
# =========================================================

class UnifiedHybridStemBlock(nn.Module):
    def __init__(self, channels, num_heads=4, proto_size=4, expansion=2, drop_path=0.0):
        super().__init__()
        hidden_ch = channels * expansion

        # CNN local branch
        self.local_norm = LayerNorm2d(channels)
        self.local_pw1 = nn.Conv2d(channels, hidden_ch, kernel_size=1, bias=False)
        self.local_dw = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=hidden_ch, bias=False)
        self.local_grn = GRN(hidden_ch)
        self.local_pw2 = nn.Conv2d(hidden_ch, channels, kernel_size=1, bias=False)

        # Transformer global branch
        self.global_attn = PrototypeAttention2D(channels=channels, num_heads=num_heads, proto_size=proto_size,
            drop_path=drop_path)

        # Lightweight FFN
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn_pw1 = nn.Conv2d(channels, hidden_ch, kernel_size=1, bias=False)
        self.ffn_dw = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=hidden_ch, bias=False)
        self.ffn_pw2 = nn.Conv2d(hidden_ch, channels, kernel_size=1, bias=False)

        # CNN-Transformer fusion
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False), LayerNorm2d(channels),
            nn.GELU())

        # Attention residual gate
        ca_hidden = max(channels // 8, 4)
        self.channel_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, ca_hidden, kernel_size=1, bias=False), nn.GELU(),
            nn.Conv2d(ca_hidden, channels, kernel_size=1, bias=False), nn.Sigmoid())

        self.spatial_attn = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x

        # CNN local branch
        local = self.local_norm(x)
        local = self.local_pw1(local)
        local = self.local_dw(local)
        local = F.gelu(local)
        local = self.local_grn(local)
        local = self.local_pw2(local)

        # Transformer branch
        global_feat = self.global_attn(x)

        # FFN after attention
        ffn_identity = global_feat
        global_feat = self.ffn_norm(global_feat)
        global_feat = self.ffn_pw1(global_feat)
        global_feat = self.ffn_dw(global_feat)
        global_feat = F.gelu(global_feat)
        global_feat = self.ffn_pw2(global_feat)
        global_feat = ffn_identity + self.drop_path(global_feat)

        # Fusion
        fused = torch.cat([local, global_feat], dim=1)
        fused = self.fuse(fused)

        # Attention residual mechanism
        ca = self.channel_attn(fused)
        fused_ca = fused * ca

        avg_out = torch.mean(fused_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(fused_ca, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))

        # Residual attention, avoiding over-suppression of thin cracks.
        fused = fused + fused_ca * sa

        out = identity + self.drop_path(fused)
        return out


# =========================================================
# 7. Unified Hybrid Stem
# =========================================================

class UnifiedHybridStem(nn.Module):
    """
    Unified hybrid stem used in the main encoder and decoder.
    """

    def __init__(self, in_ch, out_ch, stride=1, depth=1, num_heads=4, proto_size=4, expansion=2, drop_path=0.0):
        super().__init__()
        self.proj = DSConvProj(in_ch=in_ch, out_ch=out_ch, stride=stride)

        blocks = []
        for _ in range(depth):
            blocks.append(UnifiedHybridStemBlock(channels=out_ch, num_heads=num_heads, proto_size=proto_size,
                    expansion=expansion, drop_path=drop_path))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        return x


# =========================================================
# 8. ALFR: Adaptive Learnable Feature Reconstruction
# =========================================================

class AdaptiveLearnableFeatureReconstruction(nn.Module):
    """
    ALFR: Adaptive Learnable Feature Reconstruction.

    It is specifically used for skip feature reconstruction.

    Paths:
        1. Local structure reconstruction
        2. High-frequency detail reconstruction
        3. Learnable basis reconstruction
        4. Adaptive residual weighting
    """

    def __init__(self, in_ch, out_ch, num_bases=8, expansion=2, drop_path=0.0):
        super().__init__()
        self.out_ch = out_ch
        self.num_bases = num_bases
        hidden_ch = out_ch * expansion

        # Channel alignment
        self.align = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), LayerNorm2d(out_ch), nn.GELU())

        # Local structure reconstruction
        self.local_reconstruction = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch,
                bias=False), nn.Conv2d(out_ch, hidden_ch, kernel_size=1, bias=False), nn.GELU(), GRN(hidden_ch),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False))

        # High-frequency detail reconstruction
        self.detail_norm = LayerNorm2d(out_ch)
        self.detail_dwconv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False)

        self.detail_gate = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False), LayerNorm2d(out_ch),
            nn.GELU(), nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True), nn.Sigmoid())

        # Learnable basis reconstruction
        self.basis = nn.Parameter(torch.randn(num_bases, out_ch) * 0.02)

        self.query_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.reconstruct_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

        # Adaptive residual weighting
        weight_hidden = max(out_ch // 8, 4)
        self.adaptive_weight = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, weight_hidden, kernel_size=1, bias=False), nn.GELU(),
            nn.Conv2d(weight_hidden, 3, kernel_size=1, bias=True), nn.Sigmoid())

        self.out_norm = LayerNorm2d(out_ch)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = self.align(x)
        identity = x

        # 1. Local reconstruction
        local_feat = self.local_reconstruction(x)

        # 2. High-frequency reconstruction
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - smooth

        detail_feat = self.detail_norm(high_freq)
        detail_feat = self.detail_dwconv(detail_feat)

        detail_gate = self.detail_gate(torch.cat([x, detail_feat], dim=1))
        detail_feat = detail_feat * detail_gate

        # 3. Learnable basis reconstruction
        b, c, h, w = x.shape

        query = self.query_proj(x)
        query = query.flatten(2).transpose(1, 2)  # [B, H*W, C]

        basis = self.basis.unsqueeze(0).expand(b, -1, -1)  # [B, K, C]

        affinity = torch.matmul(query, basis.transpose(1, 2)) * (c ** -0.5)
        affinity = affinity.softmax(dim=-1)

        basis_feat = torch.matmul(affinity, basis)  # [B, H*W, C]
        basis_feat = basis_feat.transpose(1, 2).reshape(b, c, h, w)
        basis_feat = self.reconstruct_proj(basis_feat)

        # 4. Adaptive fusion
        weights = self.adaptive_weight(x)
        w_local = weights[:, 0:1]
        w_detail = weights[:, 1:2]
        w_basis = weights[:, 2:3]

        reconstructed = (w_local * local_feat + w_detail * detail_feat + w_basis * basis_feat)

        reconstructed = self.out_norm(reconstructed)
        out = identity + self.drop_path(reconstructed)
        return out


# =========================================================
# 9. Decoder Stage with ALFR
# =========================================================

class DecoderStage(nn.Module):

    def __init__(self, dec_ch, skip_ch, out_ch, depth=1, num_heads=4, proto_size=4, num_bases=8, drop_path=0.0):
        super().__init__()
        self.dec_stem = UnifiedHybridStem(in_ch=dec_ch, out_ch=out_ch, stride=1, depth=1, num_heads=num_heads,
            proto_size=proto_size, expansion=2, drop_path=drop_path)

        self.skip_reconstructor = AdaptiveLearnableFeatureReconstruction(in_ch=skip_ch, out_ch=out_ch,
            num_bases=num_bases, expansion=2, drop_path=drop_path)

        self.fuse_gate = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False), LayerNorm2d(out_ch),
            nn.GELU(), nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True), nn.Sigmoid())

        self.refine_stem = UnifiedHybridStem(in_ch=out_ch, out_ch=out_ch, stride=1, depth=depth, num_heads=num_heads,
            proto_size=proto_size, expansion=2, drop_path=drop_path)

    def forward(self, dec_feat, skip_feat):
        dec_feat = F.interpolate(dec_feat, size=skip_feat.shape[2:], mode="bilinear", align_corners=False)

        dec_feat = self.dec_stem(dec_feat)
        skip_feat = self.skip_reconstructor(skip_feat)

        gate = self.fuse_gate(torch.cat([dec_feat, skip_feat], dim=1))
        x = dec_feat + gate * skip_feat
        x = self.refine_stem(x)
        return x


# =========================================================
# 10. UHS-ALFRNet
# =========================================================

class UHSALFRNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, dims=(32, 64, 128, 320, 768), depths=(1, 1, 2, 1, 1),
            heads=(1, 2, 4, 8, 12), proto_sizes=(9, 9, 9, 7, 7), num_bases=(8, 8, 8, 8), drop_path=0.05):
        super().__init__()
        d0, d1, d2, d3, d4 = dims

        # Encoder
        self.stem = UnifiedHybridStem(in_ch=in_ch, out_ch=d0, stride=1, depth=depths[0], num_heads=heads[0],
            proto_size=proto_sizes[0], expansion=2, drop_path=drop_path)

        self.enc1 = UnifiedHybridStem(in_ch=d0, out_ch=d1, stride=2, depth=depths[1], num_heads=heads[1],
            proto_size=proto_sizes[1], expansion=2, drop_path=drop_path)

        self.enc2 = UnifiedHybridStem(in_ch=d1, out_ch=d2, stride=2, depth=depths[2], num_heads=heads[2],
            proto_size=proto_sizes[2], expansion=2, drop_path=drop_path)

        self.enc3 = UnifiedHybridStem(in_ch=d2, out_ch=d3, stride=2, depth=depths[3], num_heads=heads[3],
            proto_size=proto_sizes[3], expansion=2, drop_path=drop_path)

        self.bridge = UnifiedHybridStem(in_ch=d3, out_ch=d4, stride=2, depth=depths[4], num_heads=heads[4],
            proto_size=proto_sizes[4], expansion=2, drop_path=drop_path)

        # Decoder
        self.dec3 = DecoderStage(dec_ch=d4, skip_ch=d3, out_ch=d3, depth=1, num_heads=heads[3],
            proto_size=proto_sizes[3], num_bases=num_bases[3], drop_path=drop_path)

        self.dec2 = DecoderStage(dec_ch=d3, skip_ch=d2, out_ch=d2, depth=1, num_heads=heads[2],
            proto_size=proto_sizes[2], num_bases=num_bases[2], drop_path=drop_path)

        self.dec1 = DecoderStage(dec_ch=d2, skip_ch=d1, out_ch=d1, depth=1, num_heads=heads[1],
            proto_size=proto_sizes[1], num_bases=num_bases[1], drop_path=drop_path)

        self.dec0 = DecoderStage(dec_ch=d1, skip_ch=d0, out_ch=d0, depth=1, num_heads=heads[0],
            proto_size=proto_sizes[0], num_bases=num_bases[0], drop_path=drop_path)

        # Segmentation head
        self.seg_head = nn.Sequential(DSConvProj(in_ch=d0, out_ch=d0, stride=1), nn.Dropout2d(0.05),
            nn.Conv2d(d0, num_classes, kernel_size=1))

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)       # [B, d0, H, W]
        x1 = self.enc1(x0)      # [B, d1, H/2, W/2]
        x2 = self.enc2(x1)      # [B, d2, H/4, W/4]
        x3 = self.enc3(x2)      # [B, d3, H/8, W/8]
        x4 = self.bridge(x3)    # [B, d4, H/16, W/16]

        # Decoder
        y3 = self.dec3(x4, x3)
        y2 = self.dec2(y3, x2)
        y1 = self.dec1(y2, x1)
        y0 = self.dec0(y1, x0)

        out = self.seg_head(y0)
        return out


# =========================================================
# 11. Loss Functions
# =========================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


# =========================================================
# 12. Metrics
# =========================================================

@torch.no_grad()
def binary_seg_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.flatten()
    targets = targets.flatten()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"precision": precision.item(), "recall": recall.item(), "f1": f1.item(), "iou": iou.item()}


# =========================================================
# 13. Parameter and MACs Estimation
# =========================================================

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_macs(model, input_size=(1, 3, 512, 512), device="cpu"):
    macs = {"total": 0}
    hooks = []

    def conv_hook(module, inputs, output):
        x = inputs[0]
        batch_size = x.shape[0]
        out_h = output.shape[2]
        out_w = output.shape[3]

        kernel_ops = (module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups))

        mac = (batch_size * out_h * out_w * module.out_channels * kernel_ops)
        macs["total"] += mac

    def prototype_attention_hook(module, inputs, output):
        x = inputs[0]
        b, c, h, w = x.shape
        n = h * w
        p = module.proto_size * module.proto_size

        # QK + AV
        mac = 2 * b * n * p * c
        macs["total"] += mac

    def alfr_hook(module, inputs, output):
        x = output
        b, c, h, w = x.shape
        n = h * w
        k = module.num_bases

        # Query-basis affinity + basis reconstruction
        mac = 2 * b * n * k * c
        macs["total"] += mac

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, PrototypeAttention2D):
            hooks.append(m.register_forward_hook(prototype_attention_hook))
        elif isinstance(m, AdaptiveLearnableFeatureReconstruction):
            hooks.append(m.register_forward_hook(alfr_hook))

    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()
    return macs["total"]


# =========================================================
# 14. Forward Test
# =========================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UHSALFRNet(in_ch=3, num_classes=1, dims=(32, 64, 128, 320, 768), depths=(1, 1, 2, 1, 1),
        heads=(1, 2, 4, 8, 12), proto_sizes=(2, 2, 4, 4, 4), num_bases=(8, 8, 8, 8), drop_path=0.05).to(device)

    x = torch.randn(1, 3, 640, 320).to(device)

    with torch.no_grad():
        y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    params = count_trainable_params(model)
    print(f"Trainable Params: {params / 1e6:.3f} M")

    macs = estimate_macs(model, input_size=(1, 3, 640, 320), device=device)

    print(f"MACs: {macs / 1e9:.3f} G")
    print(f"FLOPs approximately: {2 * macs / 1e9:.3f} G")

    # Loss test
    target = torch.randint(0, 2, y.shape).float().to(device)
    criterion = BCEDiceLoss()
    loss = criterion(y, target)
    print("Loss test:", loss.item())
