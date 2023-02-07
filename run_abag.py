import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from config import ABaGRunConfig
from pipeline_abag import ABaGPipeline
from utils.ptp_utils import AttentionStore
from utils import ptp_utils

def read_bbox_txt_file(file: str):
    with open(file) as f:
        lines = f.read().splitlines()
        bboxes = []
        for i, line in enumerate(lines):
            bboxes.append([int(s) for s in line.split(' ')])
            
    return bboxes

def load_model(config: ABaGRunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable = ABaGPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    return stable

def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def run_on_prompt(prompt: List[str],
                  model: ABaGPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  bboxes: List,
                  config: ABaGRunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    bboxes = bboxes,
                    lr=config.lr)
    image = outputs.images[0]
    return image

@pyrallis.wrap()
def main(config: ABaGRunConfig):
    stable = load_model(config)
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices
    bboxes = read_bbox_txt_file(config.bbox_txt_file)

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(
            prompt=config.prompt,
            model=stable,
            controller=controller,
            token_indices=token_indices,
            seed=g,
            bboxes=bboxes,
            config=config)

        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

if __name__ == '__main__':
    # python run_b_magic.py --prompt "a mouse and a red car" --seeds [0] --token_indices [2,6]
    main()