import torch
import torch.nn as nn
import torchvision.transforms as T
import clip


class CLIP(nn.Module):
    def __init__(self, device, clipmodel="ViT-B/16", res=224):
        super().__init__()

        self.device = device

        if clipmodel == "ViT-L/14@336px":
            res = 336
        elif clipmodel == "RN50x4":
            res = 288
        elif clipmodel == "RN50x16":
            res = 384
        elif clipmodel == "RN50x64":
            res = 448
        elif clipmodel == "ViT-B/32":
            res = 224

        self.clip_model, self.clip_preprocess = clip.load(clipmodel, device=self.device, jit=False)

        # image augmentation
        self.clip_transform = T.Compose([
            T.Resize((res, res)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.augment_transform = T.Compose([
            T.RandomResizedCrop(res, scale=(1, 1)),
            T.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            self.clip_transform
        ])

        # self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))

    def get_text_embeds(self, prompt, negative_prompt):

        # NOTE: negative_prompt is ignored for CLIP.
        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)

        return text_z

    def train_step(self, text_z, pred_rgb):
        pred_rgb = self.clip_transform(pred_rgb)
        image_z = self.clip_model.encode_image(pred_rgb)
        loss = -torch.mean(torch.cosine_similarity(image_z, text_z))
        return loss
