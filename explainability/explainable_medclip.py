import torch
import torch.nn.functional as F
from collections import defaultdict

class ViTGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register()

    def _register(self):
        def forward_hook(module, input, output):
            # SwinStage output: (hidden_states, hw_shape)
            if isinstance(output, tuple):
                output = output[0]  # (B, N, C)
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            grad = grad_output[0]
            if isinstance(grad, tuple):
                grad = grad[0]
            self.gradients = grad

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)



class ExplainableMedCLIP(torch.nn.Module):
    """
    A unified wrapper for MedCLIP with explainability support.
    """

    def __init__(
        self,
        medclip_model,
        vision_target_layer=None,
        enable_gradcam=True,
        enable_text_attention=True,
    ):
        super().__init__()
        self.model = medclip_model
        self.enable_gradcam = enable_gradcam
        self.enable_text_attention = enable_text_attention

        # -------- buffers for explainability --------
        self._text_attentions = None

        # -------- register hooks --------
        if enable_gradcam:
            self.gradcam = ViTGradCAM(
            self.model,
            vision_target_layer
            )

        if enable_text_attention:
            self._register_text_hooks()

    def _register_text_hooks(self):
        """
        Enable attention output from transformer.
        """
        self.model.text_model.model.config.output_attentions = True

    # ======================================================
    # Unified forward
    # ======================================================
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        return_loss=False,
    ):
        """
        Unified forward that always returns embeddings.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=return_loss,
        )
        img_emb = outputs["img_embeds"]
        txt_emb = outputs["text_embeds"]
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (img_emb @ txt_emb.T)

        return img_emb, txt_emb, logits

    # ======================================================
    # Explainability APIs
    # ======================================================
    def vit_gradcam(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        prompt_index=0
    ):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs["logits"]       # (B, num_prompts)
        print("logits:",logits)
        score = logits[0, prompt_index]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        acts = self.gradcam.activations   # (B, N, C)
        grads = self.gradcam.gradients    # (B, N, C)

        assert acts is not None, "No activations captured"
        assert grads is not None, "No gradients captured"

        weights = grads.mean(dim=-1)      # (B, N)
        cam = (weights.unsqueeze(-1) * acts).sum(dim=-1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)

        return cam



    @staticmethod
    def cam_to_heatmap(cam, img_size=224, patch_size=16):
        B, N = cam.shape
        h = w = int(N ** 0.5)  # 49 → 7×7

        cam = cam.reshape(B, h, w)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        )
        return cam.squeeze(1)

    def explain_text(self):
        """
        Return attention maps from text encoder.
        """
        assert self.enable_text_attention, "Text attention not enabled."

        outputs = self.model.text_model.model(
            output_attentions=True
        )
        return outputs.attentions  # tuple(layer, B, heads, seq, seq)

    # ======================================================
    # Responsibility & Traceability
    # ======================================================
    def responsibility_trace(
        self,
        image_cam,
        token_importance,
        top_k=5
    ):
        """
        Produce a human-readable responsibility summary.
        """
        return {
            "image_evidence_score": float(image_cam.mean()),
            "text_evidence_tokens": token_importance[:top_k]
        }

    # ======================================================
    # Bias / Fairness (placeholder)
    # ======================================================
    def bias_evaluation(
        self,
        predictions,
        labels,
        metadata,
        subgroup_key
    ):
        stats = defaultdict(list)
        for p, y, m in zip(predictions, labels, metadata):
            stats[m[subgroup_key]].append(int(p == y))

        return {
            k: sum(v) / len(v) for k, v in stats.items()
        }

    # ======================================================
    # Privacy / Federated Learning (placeholder)
    # ======================================================
    def apply_privacy_filter(self, embeddings):
        """
        Placeholder for DP / FL.
        """
        return embeddings
