import torch
from torch import nn
from transformers import AutoModel

BACKBONE = "xlm-roberta-base"
HIDDEN = 768


class AttentionCDDClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        freeze_backbone: bool = True,
        unfreeze_last: int = 0,
        hidden: int = HIDDEN,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE, attn_implementation="eager")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last > 0:
                for layer in self.backbone.encoder.layer[-unfreeze_last:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        self.attention = nn.MultiheadAttention(hidden, num_heads=1, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        key_padding = ~attention_mask.bool()
        attn_out, _ = self.attention(out, out, out, key_padding_mask=key_padding)
        attn_out = self.norm(attn_out + out)
        cls = out[:, 0]
        pooled = attn_out.mean(dim=1)
        features = torch.cat([cls, pooled], dim=-1)
        return self.head(features)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]