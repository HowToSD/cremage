# Technical note on SDXL
Note: This document is still work in progress and is a very early draft.

This document covers how SDXL works on Cremage.

# Code
Main flow is defined in:
./modules/sdxl/sdxl_pipeline/sdxl_image_generator.py

generate() is the entry point.

Major steps are the following:
1. Model instantiation (init_st)
1. Weight loading (load_state_dict_into_model)
    If refiner is specified, then it goes through model instantiation code as well.
1. Sampler (denoiser) instantiation (init_sampling)
1. Wildcards resolution (resolve_wildcards)
1. txt2img (run_txt2img) or img2img (run_img2img)
1. Refiner (apply_refiner)

# Model instantiation
Code: init_st (modules/sdxl/sdxl_pipeline/sdxl_image_generator_utils.py)

* Load LoRA models (load_loras)
* Instantiate the main model (instantiate_model_from_config, instantiate_from_config)

## Configs for model instantiation
Step 1: Get the config file name from the model key
VERSION2SPECS (modules/sdxl/sdxl_pipeline/sdxl_image_generator.py)
  key: model key ("SDXL-base-1.0", or "SDXL-refiner-1.0")
  value: config file name, et al

Step 2: Instantiate the model using the definitions in config
Config file location: modules/sdxl/configs/inference/sd_xl_base.yaml

Object to instantiate is sgm.models.diffusion.DiffusionEngine.
Shown below is the config used to instantiate this object:
```
model:
  target: sgm.models.diffusion.DiffusionEngine
  params:
    scale_factor: 0.13025
    disable_first_stage_autocast: True

    denoiser_config:
      target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
      params:
        num_idx: 1000

        scaling_config:
          target: sgm.modules.diffusionmodules.denoiser_scaling.EpsScaling
        discretization_config:
          target: sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization

    network_config:
      target: sgm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers

    conditioner_config:
      target: sgm.modules.GeneralConditioner
      params:
        emb_models:
          - is_trainable: False
            input_key: txt
            target: sgm.modules.encoders.modules.FrozenCLIPEmbedder
            params:
              layer: hidden
              layer_idx: 11

          - is_trainable: False
            input_key: txt
            target: sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder2
            params:
              arch: ViT-bigG-14
              version: laion2b_s39b_b160k
              freeze: True
              layer: penultimate
              always_return_pooled: True
              legacy: False

          - is_trainable: False
            input_key: original_size_as_tuple
            target: sgm.modules.encoders.modules.ConcatTimestepEmbedderND
            params:
              outdim: 256

          - is_trainable: False
            input_key: crop_coords_top_left
            target: sgm.modules.encoders.modules.ConcatTimestepEmbedderND
            params:
              outdim: 256

          - is_trainable: False
            input_key: target_size_as_tuple
            target: sgm.modules.encoders.modules.ConcatTimestepEmbedderND
            params:
              outdim: 256
        lora_ranks: None
        lora_weights: None
    first_stage_config:
      target: sgm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          attn_type: vanilla-xformers
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
```
## Instantiation of DiffusionEngine class 
Code: modules/sdxl/sgm/models/diffusion.py
Here are the main items that take place:
* Unet model instantiation
* Denoiser instantiation
* Sample instantiation if sampler config is not none
* Conditioner instantiation
* VAE (first stage) initialization
* Loss func initialization

### Conditioner
Object: sgm.modules.GeneralConditioner
Conditioners that are defined are:
* Text embedding 1 (FrozenCLIPEmbedder)
* Text embedding 2 (FrozenOpenCLIPEmbedder2) ViT-bigG-14, laion2b_s39b_b160k
* original_size_as_tuple (target: sgm.modules.encoders.modules.ConcatTimestepEmbedderND)
* crop_coords_top_left (target: sgm.modules.encoders.modules.ConcatTimestepEmbedderND)
* target_size_as_tuple (target: sgm.modules.encoders.modules.ConcatTimestepEmbedderND)

Note that options for some of the embedders are set in run_txt2img method as shown below:
```
def run_txt2img (sdxl_image_generator.py)
...
    W, H = opt.W, opt.H
...
    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        init_dict,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
    )
...
def init_embedder_options (sdxl_image_generator_utils.py)
...
      if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]  # 1024
            orig_height = init_dict["orig_height"]  # 1024
            
            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0
            
            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left
...
        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]
```
