import torch
import torch.nn as nn
from einops import rearrange
import timm
from timm.layers import DropPath


# ==========================================
# 1. FUNZIONE HELPER: CREAZIONE ViT CON TIMM
# ==========================================
def create_custom_vit(depth=6, dropout=0.10, attention_dropout=0.1, drop_path_rate=0.2, pretrained=True):
    """
    Crea una ViT basata su DinoBloom-S COMPLETAMENTE CONGELATA.

    MODIFICA: TUTTO il feature extractor e' congelato (0 parametri trainabili).
    Questo riduce i parametri trainabili da 4.56M a ~3.77M.
    """
    print(f"Creazione Custom DinoBloom-S ViT (Profondita: {depth} blocchi, Estrazione Feature Spaziali: 384)")

    model = timm.create_model(
        'hf_hub:1aurent/vit_small_patch14_224.dinobloom',
        pretrained=True,
        num_classes=0,
        img_size=224,
        global_pool='',
        drop_rate=dropout,
        attn_drop_rate=attention_dropout,
        drop_path_rate=drop_path_rate,
    )

    if len(model.blocks) > depth:
        model.blocks = nn.Sequential(*list(model.blocks[:depth]))
        print(f"Modello alleggerito: ridotto a {len(model.blocks)} blocchi di attenzione.")

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"DinoBloom-S: TUTTO il feature extractor e' CONGELATO ({n_train:,}/{n_total:,} parametri trainabili).")

    return model


# ==========================================
# 2. MODULO GATED CROSS-ATTENTION
# ==========================================
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.25):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x, context):
        b, n, _ = x.shape
        x_norm = self.norm(x)
        context_norm = self.norm(context)

        q = self.to_q(x_norm)
        kv = self.to_kv(context_norm).chunk(2, dim=-1)
        k, v = kv

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        fused_info = self.to_out(out)

        g = self.gate(fused_info)
        return (g * fused_info) + (0.1 * fused_info)


# ==========================================
# 3. BLOCCO TRANSFORMER CON DROP-PATH
# ==========================================
class TransformerBlockWithDropPath(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_head=64, mlp_ratio=4.0, drop_path=0.1, dropout=0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(dim=embed_dim, heads=num_heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, context=None):
        if context is not None:
            x = x + self.drop_path(self.attn(x, context))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==========================================
# 4. TUMOR CLASSIFIER CON CROSS-ATTENTION BIDIREZIONALE
# ==========================================
class TumorClassifierViT(nn.Module):
    def __init__(self, num_classes=5):
        super(TumorClassifierViT, self).__init__()
        self.feature_dim = 384

        # Branch 1: Immagini Originali
        self.vit_orig = create_custom_vit(
            depth=6,
            dropout=0.15,
            attention_dropout=0.1,
            drop_path_rate=0.1
        )

        # Branch 2: Immagini Segmentate
        self.vit_seg = create_custom_vit(
            depth=6,
            dropout=0.15,
            attention_dropout=0.1,
            drop_path_rate=0.1
        )

        # Blocchi Transformer con DropPath (Cross-Attention bidirezionale)
        self.transformer_block_orig = TransformerBlockWithDropPath(
            embed_dim=self.feature_dim,
            num_heads=8,
            dim_head=64,
            mlp_ratio=4.0,
            drop_path=0.20,
            dropout=0.25
        )
        self.transformer_block_seg = TransformerBlockWithDropPath(
            embed_dim=self.feature_dim,
            num_heads=8,
            dim_head=64,
            mlp_ratio=4.0,
            drop_path=0.20,
            dropout=0.25
        )

        # ✅ RIMOSSO: Gate di affidabilità della segmentazione
        # self.ln_gate_seg = nn.LayerNorm(self.feature_dim)
        # self.gate_layer_seg = nn.Sequential(
        #     nn.Linear(384, 192),
        #     nn.ReLU(),
        #     nn.Linear(192, 384)
        # )

        # Classificatore finale MLP
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_orig, img_seg, return_features=False):
        """
        Forward pass del modello dual-stream con Cross-Attention bidirezionale.
        ✅ SENZA gate di affidabilità sulla segmentazione.

        Args:
            img_orig: Immagini originali (Batch, 3, H, W)
            img_seg: Immagini segmentate (Batch, 3, H, W)
            return_features: Se True, restituisce anche le feature intermedie

        Returns:
            Se return_features=False: logits (Batch, num_classes)
            Se return_features=True: (logits, features)
        """
        # Estrazione features dai patch (senza CLS token)
        feat_orig = self.vit_orig(img_orig)[:, 1:, :]
        feat_seg = self.vit_seg(img_seg)[:, 1:, :]

        # Cross-Attention bidirezionale
        fused_orig = self.transformer_block_orig(feat_orig, context=feat_seg)
        fused_seg = self.transformer_block_seg(feat_seg, context=feat_orig)

        # ✅ RIMOSSO: Gate di affidabilità della segmentazione
        # logit_seg = self.gate_layer_seg(self.ln_gate_seg(fused_seg))
        # seg_trust = torch.sigmoid(logit_seg)
        # final_orig = fused_orig * (1.0 - seg_trust)
        # final_seg = fused_seg * seg_trust
        # fused_total = final_orig + final_seg

        # ✅ NUOVO: Fusione semplice additiva (senza gate)
        fused_total = fused_orig + fused_seg  # [batch, 196, 384]

        # Pooling medio sui token per ottenere l'embedding immagine (B, 384)
        image_embedding = fused_total.mean(dim=1)  # [batch, 384]

        # Classificazione
        logits = self.classifier(image_embedding)

        if return_features:
            return logits, image_embedding

        return logits

    def extract_features(self, img_orig, img_seg):
        """Estrae solo le feature intermedie."""
        logits, features = self.forward(img_orig, img_seg, return_features=True)
        return features

    def forward_with_features(self, img_orig, img_seg):
        """Forward pass che restituisce sia logits che features."""
        return self.forward(img_orig, img_seg, return_features=True)
