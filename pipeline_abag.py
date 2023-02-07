from typing import List, Optional, Union, Tuple

import torch
import numpy as np
from diffusers.schedulers import LMSDiscreteScheduler
from torch.nn import functional as F

from pipeline_stable_diffusion import StableDiffusionPipeline
from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention

from utils.gaussian_smoothing import GaussianSmoothing

class ABaGPipeline(StableDiffusionPipeline):

    @staticmethod
    def _compute_max_attention_per_bbox_and_outofbbox(
            attention_maps: torch.Tensor,
            indices_to_alter: List[int],
            smooth_attentions: bool = False,
            sigma: float = 0.5,
            kernel_size: int = 3,
            bboxes: List = None) -> List[torch.Tensor]:

        attention_for_text = attention_maps[:, :, 1:-1]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values of bbox and out of bboxes
        bbox_max_indices_list = []
        out_of_bbox_max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            
            index = [bbox[0]-1 for bbox in bboxes].index(i)
            bbox = bboxes[index]
            
            # bbox内の大域的な特徴を取得するために平均値を使用しています。
            tmp_image = image.clone()
            bbox_max_indices_list.append(
                tmp_image[bbox[1]:bbox[2], bbox[3]:bbox[4]].mean()
            )

            # bbox外の最大値
            tmp_image = image.clone()
            tmp_image[bbox[1]:bbox[2], bbox[3]:bbox[4]] = 0
            out_of_bbox_max_indices_list.append(tmp_image.max())

        return bbox_max_indices_list, out_of_bbox_max_indices_list

    def _aggregate_and_get_max_attention_per_bbox_and_outofbbox(
            self, 
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            smooth_attentions: bool = False,
            sigma: float = 0.5,
            kernel_size: int = 3,
            bboxes: List = None):
        
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        
        bbox_max_indices_list, out_of_bbox_max_indices_list = self._compute_max_attention_per_bbox_and_outofbbox(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            bboxes=bboxes)
        return bbox_max_indices_list, out_of_bbox_max_indices_list

    @staticmethod
    def _compute_loss(
            bbox_max_indices_list: List[torch.Tensor],
            out_of_bbox_max_indices_list: List[torch.Tensor],
            return_loss: bool = False,
            lr: float = 0.6) -> torch.Tensor:
            
        in_bbox_losses = [max(0, 1. - curr_max) for curr_max in bbox_max_indices_list]
        out_bbox_losses = out_of_bbox_max_indices_list

        loss = (max(in_bbox_losses) + max(out_bbox_losses)) * lr

        if return_loss:
            return loss, in_bbox_losses, out_bbox_losses
        else:
            return loss
    
    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[int],
        loss: torch.Tensor,
        threshold: float,
        text_embeddings: torch.Tensor,
        text_input,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 16,
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        max_refinement_steps: int = 20,
        bboxes: list = None,
        lr: list = 0.6):
        
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            bbox_max_indices_list, out_of_bbox_max_indices_list = self._aggregate_and_get_max_attention_per_bbox_and_outofbbox(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                bboxes=bboxes)
            loss, in_bbox_losses, out_bbox_losses = self._compute_loss(
                bbox_max_indices_list=bbox_max_indices_list,
                out_of_bbox_max_indices_list=out_of_bbox_max_indices_list,
                return_loss=True,
                lr=lr)

            print(target_loss, loss)
            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)
            
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        bbox_max_indices_list, out_of_bbox_max_indices_list = self._aggregate_and_get_max_attention_per_bbox_and_outofbbox(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            bboxes=bboxes)
        
        loss, in_bbox_losses, out_bbox_losses = self._compute_loss(
            bbox_max_indices_list=bbox_max_indices_list,
            out_of_bbox_max_indices_list=out_of_bbox_max_indices_list,
            return_loss=True,
            lr=lr)

        return loss, latents, bbox_max_indices_list, out_of_bbox_max_indices_list
    
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            bboxes: list = None,
            lr: float = 0.6,
            **kwargs):
        
        text_embeddings, text_input, latents, do_classifier_free_guidance, extra_step_kwargs = self._setup_inference(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=generator,
            latents=latents,
            lr=lr,
            **kwargs
        )

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        
        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            with torch.enable_grad():

                latents = latents.clone().detach().requires_grad_(True)

                # Forward pass of denoising with text conditioning
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
                self.unet.zero_grad()

                 # Get max activation value for each subject bbox
                bbox_max_indices_list, out_of_bbox_max_indices_list = self._aggregate_and_get_max_attention_per_bbox_and_outofbbox(
                    attention_store=attention_store,
                    indices_to_alter=indices_to_alter,
                    attention_res=attention_res,
                    smooth_attentions=smooth_attentions,
                    sigma=sigma,
                    kernel_size=kernel_size,
                    bboxes=bboxes)
                
                if not run_standard_sd:
                    loss, in_bbox_losses, out_bbox_losses = self._compute_loss(
                        bbox_max_indices_list, 
                        out_of_bbox_max_indices_list,
                        return_loss=True,
                        lr=lr)

                    if i in thresholds.keys() and loss > 1. - thresholds[i]:
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        loss, latents, bbox_max_indices_list, out_of_bbox_max_indices_list = self._perform_iterative_refinement_step(
                            latents=latents,
                            indices_to_alter=indices_to_alter,
                            loss=loss,
                            threshold=thresholds[i],
                            text_embeddings=text_embeddings,
                            text_input=text_input,
                            attention_store=attention_store,
                            step_size=scale_factor * np.sqrt(scale_range[i]),
                            t=t,
                            attention_res=attention_res,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                            bboxes=bboxes,
                            lr=lr)
                    
                    if i < max_iter_to_alter:
                        loss = self._compute_loss(
                            bbox_max_indices_list, 
                            out_of_bbox_max_indices_list,
                            lr=lr)
                        if loss != 0:
                            latents = self._update_latent(
                                latents=latents,
                                loss=loss,
                                step_size=scale_factor * np.sqrt(scale_range[i]))

            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            if do_classifier_free_guidance:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        outputs = self._prepare_output(latents, output_type, return_dict)
        return outputs
