"""Taken from https://github.com/runwayml/stable-diffusion"""
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        """
        
        Args:
            model (LatentDiffusion): Reference to the main LatentDiffusion model instance.
                Note that this model contains a reference to the actual UNet model internally.
        """
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if torch.cuda.is_available():
                if attr.device != torch.device("cuda"):
                    attr = attr.to(torch.device("cuda"))
            else:
                if attr.device != torch.device(os.environ.get("GPU_DEVICE", "cpu")):
                    attr = attr.half().to(os.environ.get("GPU_DEVICE", "cpu"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        if torch.cuda.is_available():
            to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        else:
            to_torch = lambda x: x.clone().detach().to(torch.float16).to(self.model.device)
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        # Cremage note:
        # ddim_sigmas contain:
        # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        # 0., 0.], dtype=torch.float64)

        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        logger.debug(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            if torch.cuda.is_available():
                img = torch.randn(shape, device=device)
            else:
                img = torch.randn(shape, device=device).half()
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1  # descending starting with 49
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        Defines a single step of diffusion.

        Args:
            index (int): Decreasing index so that alpha value will go up as we iterate.

        Note:
        self.model.alphas_cumprod is using:
        tensor([
        0.9990, 0.9985, 0.9976, 0.9966, 0.9956, 0.9946, 0.9941, 0.9932, 0.9922,
        0.9912, 0.9902, 0.9897, 0.9888, 0.9878, 0.9868, 0.9858, 0.9849, 0.9839,
        0.9834, 0.9824, 0.9814, 0.9805, 0.9795, 0.9785, 0.9775, 0.9766, 0.9756,
        0.9746, 0.9736, 0.9727, 0.9717, 0.9707, 0.9697, 0.9688, 0.9678, 0.9668,
        0.9658, 0.9648, 0.9639, 0.9629, 0.9619, 0.9609, 0.9600, 0.9590, 0.9580,
        0.9565, 0.9556, 0.9546, 0.9536, 0.9526, 0.9517, 0.9507, 0.9497, 0.9482,
        0.9473, 0.9463, 0.9453, 0.9443, 0.9429, 0.9419, 0.9409, 0.9399, 0.9385,
        0.9375, 0.9365, 0.9355, 0.9341, 0.9331, 0.9321, 0.9312, 0.9297, 0.9287,
        0.9277, 0.9263, 0.9253, 0.9243, 0.9229, 0.9219, 0.9204, 0.9194, 0.9185,
        0.9170, 0.9160, 0.9146, 0.9136, 0.9126, 0.9111, 0.9102, 0.9087, 0.9077,
        0.9062, 0.9053, 0.9038, 0.9028, 0.9014, 0.9004, 0.8989, 0.8979, 0.8965,
        0.8955, 0.8940, 0.8931, 0.8916, 0.8906, 0.8892, 0.8882, 0.8867, 0.8853,
        0.8843, 0.8828, 0.8818, 0.8804, 0.8789, 0.8779, 0.8765, 0.8750, 0.8740,
        0.8726, 0.8711, 0.8701, 0.8687, 0.8672, 0.8662, 0.8647, 0.8633, 0.8623,
        0.8608, 0.8594, 0.8579, 0.8569, 0.8555, 0.8540, 0.8525, 0.8516, 0.8501,
        0.8486, 0.8472, 0.8457, 0.8447, 0.8433, 0.8418, 0.8403, 0.8389, 0.8374,
        0.8364, 0.8350, 0.8335, 0.8320, 0.8306, 0.8291, 0.8276, 0.8262, 0.8252,
        0.8237, 0.8223, 0.8208, 0.8193, 0.8179, 0.8164, 0.8149, 0.8135, 0.8120,
        0.8105, 0.8091, 0.8076, 0.8062, 0.8047, 0.8032, 0.8018, 0.8003, 0.7988,
        0.7974, 0.7959, 0.7944, 0.7930, 0.7915, 0.7900, 0.7886, 0.7871, 0.7856,
        0.7842, 0.7827, 0.7812, 0.7798, 0.7783, 0.7769, 0.7749, 0.7734, 0.7720,
        0.7705, 0.7690, 0.7676, 0.7661, 0.7646, 0.7627, 0.7612, 0.7598, 0.7583,
        0.7568, 0.7554, 0.7539, 0.7520, 0.7505, 0.7490, 0.7476, 0.7461, 0.7441,
        0.7427, 0.7412, 0.7397, 0.7383, 0.7363, 0.7349, 0.7334, 0.7319, 0.7305,
        0.7285, 0.7271, 0.7256, 0.7241, 0.7222, 0.7207, 0.7192, 0.7173, 0.7158,
        0.7144, 0.7129, 0.7109, 0.7095, 0.7080, 0.7065, 0.7046, 0.7031, 0.7017,
        0.6997, 0.6982, 0.6968, 0.6948, 0.6934, 0.6919, 0.6899, 0.6885, 0.6870,
        0.6851, 0.6836, 0.6821, 0.6802, 0.6787, 0.6772, 0.6753, 0.6738, 0.6724,
        0.6704, 0.6689, 0.6670, 0.6655, 0.6641, 0.6621, 0.6606, 0.6592, 0.6572,
        0.6558, 0.6538, 0.6523, 0.6509, 0.6489, 0.6475, 0.6455, 0.6440, 0.6426,
        0.6406, 0.6392, 0.6372, 0.6357, 0.6343, 0.6323, 0.6309, 0.6289, 0.6274,
        0.6255, 0.6240, 0.6226, 0.6206, 0.6191, 0.6172, 0.6157, 0.6138, 0.6123,
        0.6108, 0.6089, 0.6074, 0.6055, 0.6040, 0.6021, 0.6006, 0.5991, 0.5972,
        0.5957, 0.5938, 0.5923, 0.5903, 0.5889, 0.5869, 0.5854, 0.5840, 0.5820,
        0.5806, 0.5786, 0.5771, 0.5752, 0.5737, 0.5718, 0.5703, 0.5688, 0.5669,
        0.5654, 0.5635, 0.5620, 0.5601, 0.5586, 0.5566, 0.5552, 0.5532, 0.5518,
        0.5503, 0.5483, 0.5469, 0.5449, 0.5435, 0.5415, 0.5400, 0.5381, 0.5366,
        0.5352, 0.5332, 0.5317, 0.5298, 0.5283, 0.5264, 0.5249, 0.5234, 0.5215,
        0.5200, 0.5181, 0.5166, 0.5146, 0.5132, 0.5117, 0.5098, 0.5083, 0.5063,
        0.5049, 0.5029, 0.5015, 0.4998, 0.4980, 0.4966, 0.4949, 0.4932, 0.4915,
        0.4897, 0.4883, 0.4866, 0.4849, 0.4832, 0.4814, 0.4800, 0.4783, 0.4766,
        0.4749, 0.4734, 0.4717, 0.4700, 0.4683, 0.4668, 0.4651, 0.4634, 0.4617,
        0.4602, 0.4585, 0.4568, 0.4553, 0.4536, 0.4519, 0.4504, 0.4487, 0.4470,
        0.4456, 0.4438, 0.4421, 0.4407, 0.4390, 0.4373, 0.4358, 0.4341, 0.4326,
        0.4309, 0.4292, 0.4277, 0.4260, 0.4246, 0.4229, 0.4214, 0.4197, 0.4180,
        0.4165, 0.4148, 0.4133, 0.4116, 0.4102, 0.4084, 0.4070, 0.4053, 0.4038,
        0.4023, 0.4006, 0.3992, 0.3975, 0.3960, 0.3943, 0.3928, 0.3914, 0.3896,
        0.3882, 0.3867, 0.3850, 0.3835, 0.3818, 0.3804, 0.3789, 0.3774, 0.3757,
        0.3743, 0.3728, 0.3711, 0.3696, 0.3682, 0.3667, 0.3650, 0.3635, 0.3621,
        0.3606, 0.3591, 0.3574, 0.3560, 0.3545, 0.3530, 0.3516, 0.3501, 0.3486,
        0.3469, 0.3455, 0.3440, 0.3425, 0.3411, 0.3396, 0.3381, 0.3367, 0.3352,
        0.3337, 0.3323, 0.3308, 0.3293, 0.3279, 0.3264, 0.3250, 0.3235, 0.3223,
        0.3208, 0.3193, 0.3179, 0.3164, 0.3149, 0.3135, 0.3123, 0.3108, 0.3093,
        0.3079, 0.3064, 0.3052, 0.3037, 0.3022, 0.3008, 0.2996, 0.2981, 0.2966,
        0.2954, 0.2939, 0.2925, 0.2913, 0.2898, 0.2886, 0.2871, 0.2856, 0.2844,
        0.2830, 0.2817, 0.2803, 0.2791, 0.2776, 0.2764, 0.2749, 0.2737, 0.2725,
        0.2710, 0.2698, 0.2683, 0.2671, 0.2659, 0.2644, 0.2632, 0.2620, 0.2605,
        0.2593, 0.2581, 0.2566, 0.2554, 0.2542, 0.2529, 0.2515, 0.2502, 0.2490,
        0.2478, 0.2466, 0.2452, 0.2440, 0.2428, 0.2416, 0.2402, 0.2390, 0.2378,
        0.2366, 0.2354, 0.2341, 0.2329, 0.2317, 0.2305, 0.2292, 0.2281, 0.2269,
        0.2257, 0.2245, 0.2233, 0.2222, 0.2209, 0.2197, 0.2186, 0.2174, 0.2163,
        0.2151, 0.2140, 0.2128, 0.2117, 0.2104, 0.2094, 0.2083, 0.2070, 0.2059,
        0.2048, 0.2036, 0.2025, 0.2014, 0.2003, 0.1992, 0.1981, 0.1970, 0.1959,
        0.1948, 0.1937, 0.1926, 0.1915, 0.1904, 0.1893, 0.1884, 0.1873, 0.1862,
        0.1851, 0.1841, 0.1830, 0.1820, 0.1809, 0.1798, 0.1788, 0.1777, 0.1768,
        0.1758, 0.1747, 0.1737, 0.1727, 0.1716, 0.1707, 0.1697, 0.1687, 0.1676,
        0.1666, 0.1656, 0.1647, 0.1637, 0.1627, 0.1617, 0.1608, 0.1598, 0.1588,
        0.1578, 0.1570, 0.1560, 0.1550, 0.1541, 0.1532, 0.1522, 0.1512, 0.1504,
        0.1494, 0.1486, 0.1476, 0.1467, 0.1458, 0.1449, 0.1439, 0.1431, 0.1422,
        0.1412, 0.1404, 0.1395, 0.1387, 0.1378, 0.1368, 0.1360, 0.1351, 0.1343,
        0.1334, 0.1326, 0.1317, 0.1309, 0.1300, 0.1293, 0.1284, 0.1276, 0.1267,
        0.1259, 0.1251, 0.1243, 0.1235, 0.1227, 0.1218, 0.1210, 0.1202, 0.1194,
        0.1187, 0.1179, 0.1171, 0.1163, 0.1155, 0.1148, 0.1140, 0.1132, 0.1125,
        0.1118, 0.1110, 0.1102, 0.1095, 0.1088, 0.1080, 0.1072, 0.1065, 0.1058,
        0.1051, 0.1044, 0.1036, 0.1029, 0.1022, 0.1015, 0.1008, 0.1001, 0.0994,
        0.0987, 0.0980, 0.0974, 0.0967, 0.0960, 0.0953, 0.0947, 0.0940, 0.0933,
        0.0927, 0.0920, 0.0913, 0.0907, 0.0900, 0.0894, 0.0887, 0.0881, 0.0875,
        0.0869, 0.0862, 0.0856, 0.0850, 0.0844, 0.0837, 0.0831, 0.0825, 0.0819,
        0.0813, 0.0807, 0.0801, 0.0795, 0.0789, 0.0784, 0.0778, 0.0772, 0.0766,
        0.0760, 0.0755, 0.0750, 0.0743, 0.0738, 0.0732, 0.0727, 0.0721, 0.0716,
        0.0710, 0.0705, 0.0700, 0.0695, 0.0689, 0.0684, 0.0679, 0.0673, 0.0668,
        0.0663, 0.0658, 0.0653, 0.0648, 0.0643, 0.0638, 0.0633, 0.0628, 0.0623,
        0.0618, 0.0613, 0.0608, 0.0604, 0.0599, 0.0594, 0.0589, 0.0585, 0.0580,
        0.0575, 0.0571, 0.0566, 0.0562, 0.0557, 0.0553, 0.0548, 0.0544, 0.0540,
        0.0535, 0.0531, 0.0526, 0.0522, 0.0518, 0.0514, 0.0509, 0.0505, 0.0501,
        0.0497, 0.0493, 0.0489, 0.0485, 0.0481, 0.0477, 0.0473, 0.0469, 0.0465,
        0.0461, 0.0457, 0.0453, 0.0450, 0.0446, 0.0442, 0.0438, 0.0435, 0.0431,
        0.0427, 0.0424, 0.0420, 0.0416, 0.0413, 0.0409, 0.0406, 0.0402, 0.0399,
        0.0396, 0.0392, 0.0388, 0.0385, 0.0382, 0.0378, 0.0375, 0.0372, 0.0369,
        0.0366, 0.0362, 0.0359, 0.0356, 0.0353, 0.0350, 0.0347, 0.0344, 0.0341,
        0.0338, 0.0334, 0.0331, 0.0328, 0.0326, 0.0323, 0.0320, 0.0317, 0.0314,
        0.0311, 0.0308, 0.0305, 0.0303, 0.0300, 0.0297, 0.0294, 0.0292, 0.0289,
        0.0287, 0.0284, 0.0281, 0.0279, 0.0276, 0.0274, 0.0271, 0.0269, 0.0266,
        0.0264, 0.0261, 0.0259, 0.0256, 0.0254, 0.0251, 0.0249, 0.0247, 0.0244,
        0.0242, 0.0240, 0.0237, 0.0235, 0.0233, 0.0231, 0.0229, 0.0226, 0.0224,
        0.0222, 0.0220, 0.0218, 0.0216, 0.0214, 0.0211, 0.0210, 0.0208, 0.0205,
        0.0203, 0.0201, 0.0199, 0.0198, 0.0196, 0.0194, 0.0192, 0.0190, 0.0188,
        0.0186, 0.0184, 0.0182, 0.0181, 0.0179, 0.0177, 0.0175, 0.0174, 0.0172,
        0.0170, 0.0168, 0.0167, 0.0165, 0.0163, 0.0162, 0.0160, 0.0159, 0.0157,
        0.0155, 0.0154, 0.0152, 0.0151, 0.0149, 0.0147, 0.0146, 0.0145, 0.0143,
        0.0142, 0.0140, 0.0139, 0.0137, 0.0136, 0.0134, 0.0133, 0.0132, 0.0130,
        0.0129, 0.0127, 0.0126, 0.0125, 0.0123, 0.0122, 0.0121, 0.0120, 0.0118,
        0.0117, 0.0116, 0.0115, 0.0113, 0.0112, 0.0111, 0.0110, 0.0109, 0.0107,
        0.0106, 0.0105, 0.0104, 0.0103, 0.0102, 0.0101, 0.0100, 0.0098, 0.0097,
        0.0096, 0.0095, 0.0094, 0.0093, 0.0092, 0.0091, 0.0090, 0.0089, 0.0088,
        0.0087, 0.0086, 0.0085, 0.0084, 0.0083, 0.0082, 0.0082, 0.0081, 0.0080,
        0.0079, 0.0078, 0.0077, 0.0076, 0.0075, 0.0075, 0.0074, 0.0073, 0.0072,
        0.0071, 0.0070, 0.0070, 0.0069, 0.0068, 0.0067, 0.0066, 0.0066, 0.0065,
        0.0064, 0.0063, 0.0063, 0.0062, 0.0061, 0.0061, 0.0060, 0.0059, 0.0058,
        0.0058, 0.0057, 0.0056, 0.0056, 0.0055, 0.0054, 0.0054, 0.0053, 0.0053,
        0.0052, 0.0051, 0.0051, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0047,
        0.0047], device='cuda:0', dtype=torch.float16)

        Note that at high-level to remember, this goes from 1 to 0.

        alphas is set to:
            tensor([0.9985, 0.9805, 0.9609, 0.9399, 0.9170, 0.8931, 0.8672, 0.8403, 0.8120,
            0.7827, 0.7520, 0.7207, 0.6885, 0.6558, 0.6226, 0.5889, 0.5552, 0.5215,
            0.4883, 0.4553, 0.4229, 0.3914, 0.3606, 0.3308, 0.3022, 0.2749, 0.2490,
            0.2245, 0.2014, 0.1798, 0.1598, 0.1412, 0.1243, 0.1088, 0.0947, 0.0819,
            0.0705, 0.0604, 0.0514, 0.0435, 0.0366, 0.0305, 0.0254, 0.0210, 0.0172,
            0.0140, 0.0113, 0.0091, 0.0073, 0.0058]
        
        Values for index, t, a_t and sigma_t:
            index: 49
            t: [981]
            a_t: [[[[0.0058]]]]
            sigma_t: [[[[0.]]]]
            index: 48
            t: [961]
            a_t: [[[[0.0073]]]]
            sigma_t: [[[[0.]]]]
            index: 47
            t: [941]
            a_t: [[[[0.0091]]]]
            sigma_t: [[[[0.]]]]
            index: 46
            t: [921]
            a_t: [[[[0.0113]]]]
            sigma_t: [[[[0.]]]]
            index: 45
            t: [901]
            a_t: [[[[0.0140]]]]
            sigma_t: [[[[0.]]]]
            index: 44
            t: [881]
            a_t: [[[[0.0172]]]]
            sigma_t: [[[[0.]]]]
            index: 43
            t: [861]
            a_t: [[[[0.0210]]]]
            sigma_t: [[[[0.]]]]
            index: 42
            t: [841]
            a_t: [[[[0.0254]]]]
            sigma_t: [[[[0.]]]]
            index: 41
            t: [821]
            a_t: [[[[0.0305]]]]
            sigma_t: [[[[0.]]]]
            index: 40
            t: [801]
            a_t: [[[[0.0366]]]]
            sigma_t: [[[[0.]]]]
            index: 39
            t: [781]
            a_t: [[[[0.0435]]]]
            sigma_t: [[[[0.]]]]
            index: 38
            t: [761]
            a_t: [[[[0.0514]]]]
            sigma_t: [[[[0.]]]]
            index: 37
            t: [741]
            a_t: [[[[0.0604]]]]
            sigma_t: [[[[0.]]]]
            index: 36
            t: [721]
            a_t: [[[[0.0705]]]]
            sigma_t: [[[[0.]]]]
            index: 35
            t: [701]
            a_t: [[[[0.0819]]]]
            sigma_t: [[[[0.]]]]
            index: 34
            t: [681]
            a_t: [[[[0.0947]]]]
            sigma_t: [[[[0.]]]]
            index: 33
            t: [661]
            a_t: [[[[0.1088]]]]
            sigma_t: [[[[0.]]]]
            index: 32
            t: [641]
            a_t: [[[[0.1243]]]]
            sigma_t: [[[[0.]]]]
            index: 31
            t: [621]
            a_t: [[[[0.1412]]]]
            sigma_t: [[[[0.]]]]
            index: 30
            t: [601]
            a_t: [[[[0.1598]]]]
            sigma_t: [[[[0.]]]]
            index: 29
            t: [581]
            a_t: [[[[0.1798]]]]
            sigma_t: [[[[0.]]]]
            index: 28
            t: [561]
            a_t: [[[[0.2014]]]]
            sigma_t: [[[[0.]]]]
            index: 27
            t: [541]
            a_t: [[[[0.2245]]]]
            sigma_t: [[[[0.]]]]
            index: 26
            t: [521]
            a_t: [[[[0.2490]]]]
            sigma_t: [[[[0.]]]]
            index: 25
            t: [501]
            a_t: [[[[0.2749]]]]
            sigma_t: [[[[0.]]]]
            index: 24
            t: [481]
            a_t: [[[[0.3022]]]]
            sigma_t: [[[[0.]]]]
            index: 23
            t: [461]
            a_t: [[[[0.3308]]]]
            sigma_t: [[[[0.]]]]
            index: 22
            t: [441]
            a_t: [[[[0.3606]]]]
            sigma_t: [[[[0.]]]]
            index: 21
            t: [421]
            a_t: [[[[0.3914]]]]
            sigma_t: [[[[0.]]]]
            index: 20
            t: [401]
            a_t: [[[[0.4229]]]]
            sigma_t: [[[[0.]]]]
            index: 19
            t: [381]
            a_t: [[[[0.4553]]]]
            sigma_t: [[[[0.]]]]
            index: 18
            t: [361]
            a_t: [[[[0.4883]]]]
            sigma_t: [[[[0.]]]]
            index: 17
            t: [341]
            a_t: [[[[0.5215]]]]
            sigma_t: [[[[0.]]]]
            index: 16
            t: [321]
            a_t: [[[[0.5552]]]]
            sigma_t: [[[[0.]]]]
            index: 15
            t: [301]
            a_t: [[[[0.5889]]]]
            sigma_t: [[[[0.]]]]
            index: 14
            t: [281]
            a_t: [[[[0.6226]]]]
            sigma_t: [[[[0.]]]]
            index: 13
            t: [261]
            a_t: [[[[0.6558]]]]
            sigma_t: [[[[0.]]]]
            index: 12
            t: [241]
            a_t: [[[[0.6885]]]]
            sigma_t: [[[[0.]]]]
            index: 11
            t: [221]
            a_t: [[[[0.7207]]]]
            sigma_t: [[[[0.]]]]
            index: 10
            t: [201]
            a_t: [[[[0.7520]]]]
            sigma_t: [[[[0.]]]]
            index: 9
            t: [181]
            a_t: [[[[0.7827]]]]
            sigma_t: [[[[0.]]]]
            index: 8
            t: [161]
            a_t: [[[[0.8120]]]]
            sigma_t: [[[[0.]]]]
            index: 7
            t: [141]
            a_t: [[[[0.8403]]]]
            sigma_t: [[[[0.]]]]
            index: 6
            t: [121]
            a_t: [[[[0.8672]]]]
            sigma_t: [[[[0.]]]]
            index: 5
            t: [101]
            a_t: [[[[0.8931]]]]
            sigma_t: [[[[0.]]]]
            index: 4
            t: [81]
            a_t: [[[[0.9170]]]]
            sigma_t: [[[[0.]]]]
            index: 3
            t: [61]
            a_t: [[[[0.9399]]]]
            sigma_t: [[[[0.]]]]
            index: 2
            t: [41]
            a_t: [[[[0.9609]]]]
            sigma_t: [[[[0.]]]]
            index: 1
            t: [21]
            a_t: [[[[0.9805]]]]
            sigma_t: [[[[0.]]]]
            index: 0
            t: [1]
            a_t: [[[[0.9985]]]]
            sigma_t: [[[[0.]]]]
        """
        b, *_, device = *x.shape, x.device

        # Steps for this block
        # * Calls UNet
        # * Apply CFG to the result
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # Pass both negative and positive prompt embedding and get inference result.
            # Split the prediction to the noise for negative and the noise for positive.
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            # Update the noise prediction
            # This is a vector that originates from e_t_uncond.
            # This vector is pointed toward e_t from there.
            # The length of the vector is determined by CFG.
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # End of model inference & applying CFG

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # Cremage note: use_original_steps is off.
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas  # 1 towards 0
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # Predict x_0 using the reverse of forward diffusion formula
        # Cremage note on derivation:
        # Using forward diffusion formula where a_t is cumprod_a_t:
        # x_t = sqrt(a_t) * x0 + sqrt(1-a_t) * eps
        # Move eps term to LHS
        # x_t - sqrt(1-a_t) * eps = sqrt(a_t) * x0
        # Swap sides
        # sqrt(a_t) * x0 = x_t - sqrt(1-a_t) * eps
        # Divide both sides by sqrt(a_t)
        # x0 = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # direction pointing to x_t
        # Cremage note: This should be deterministic
        # Note: for forward, equivalent is sqrt(1 - a_t) * eps
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        # Cremage note: Add some noise
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # Cremage note:
        # pred_x0 part is the same as forward diffusion formula
        # noise part is different from forward diffusion
        #     dir_xt : deterministic
        #     noise  : stochastic
        # However, given that sigmas are all 0, this is equivalent of:
        # x_prev = a_prev.sqrt() * pred_x0 + (1. - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        """
        Add noise to the image using the forward diffusion algorithm.
        The formula is
        x_t = sqrt(cum_prod_alpha_t) * x_0 + sqrt(1 - cum_prod_alpha_t) * N

        Args:
            x0: Original image
            t: Time step to specify to control the amount of noise to add.
               This is used as an index to a cumprod alpha array.
               Value of t depends on whether user_original_steps is set or not.
               if use_original_steps is set to True, then DDPM steps are used.
               Number of DDPM steps is 1000, so t needs to point to a number in [0, 999]
               If use_original_steps is set to False, then DDIM steps are used.
               Since the number of DDIM steps is normally set to 50,
               t needs to point to a number in [0, 49].
               t is computed by by the total number of steps multiplied by the denoising ratio.
               For example, if denoising ratio is 0.5 and if DDIM is used,
               t is 25 (50 * 0.5).
            use_original_steps: Controls if t should point to the DDPM 1000 element
               alpha array or DDIM alpha array.
               If True, t will point to DDPM 1000-element alpha array.
               If False, t will point to DDIM 50-element alpha array. Note that
               the DDIM array size may be different.
        """
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:  # DDPM 1000-element alpha array
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:  # DDIM 50-element alpha array
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        # Apply forward diffusion formula to add noise
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
