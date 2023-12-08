"""
For ControlNet, we need more hijack
"""
from typing import Optional
import torch
from modules import shared
from modules.shared import opts
from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.extensions import active
import modules.devices as devices

from deepcache import DeepCacheParams, DeepCacheSession
from scripts.controlnet_temp import (
    UnetHook, ControlModelType, mark_prompt_context, unmark_prompt_context, predict_q_sample, predict_start_from_noise, predict_noise_from_start, timestep_embedding, cond_cast_unet, AutoMachine, aligned_adding, blur,
    lowvram, register_schedule,
    UNetModel, AbstractLowScaleModel, torch_dfs, BasicTransformerBlock, BasicTransformerBlockSGM, th
)
def is_controlnet_loaded():
    return hasattr(opts, "control_net_modules_path") #some attribute that is only set when controlnet is loaded

class UNetHookModified(DeepCacheSession):
    
    @staticmethod
    def patch_controlnet(params:DeepCacheParams, p:StableDiffusionProcessing):
        """
        Patches the controlnet. Returns original hook if successful, otherwise returns False.
        """
        from modules.scripts import scripts_txt2img, ScriptRunner
        scripts_txt2img:ScriptRunner = scripts_txt2img
        script_components:scripts.Script = None
        found_controlnet = False
        for script_components in scripts_txt2img.alwayson_scripts:
            if script_components.title() == "ControlNet":
                found_controlnet = True
                break
        if not found_controlnet:
            print("ControlNet not found")
            return False
        count = script_components.get_enabled_units(p)
        if count == 0:
            print("Controlnet not enabled")
            return False
        # controlnet-dependent scripts
        from scripts.hook import UnetHook
        original_hook = UnetHook.hook
        UnetHook.hook = UNetHookModified.generate_controlnet_hook(params)
        return (UnetHook, original_hook) # return hook and original hook

    @staticmethod
    def generate_controlnet_hook(params:DeepCacheParams):
        """
        Creates a hook for the given params.
        """
        session = DeepCacheSession()
        session.load_params(params)
        cache_enable_step = params.cache_enable_step
        full_run_step_rate = params.full_run_step_rate # '5' means run full model every 5 steps
        CACHE_LAST = session.CACHE_LAST
        session.enumerated_timestep["value"] = -1
        valid_cache_timestep_range = 50 # total 1000, 50
        def get_hook(self, model, sd_ldm, control_params, process):
            """
            Hook for the model. The script should be executed before controlnet.hook, then redirect the hook to this function
            """
            print("ControlNet hooked with deepcache")
            nonlocal session
            self.model = model
            self.sd_ldm = sd_ldm
            self.control_params = control_params
            valid_caching_in_level = min(caching_level, len(self.model.input_blocks) - 1)
            valid_caching_out_level = min(valid_caching_in_level, len(self.model.output_blocks) - 1)
            caching_level = valid_caching_out_level
            model_is_sdxl = getattr(self.sd_ldm, 'is_sdxl', False)
            cache_cond = lambda : session.enumerated_timestep["value"] % full_run_step_rate == 0 or session.enumerated_timestep["value"] > cache_enable_step
            use_cache_cond = lambda : shared.opts.deepcache_enable and session.enumerated_timestep["value"] > cache_enable_step and session.enumerated_timestep["value"] % full_run_step_rate != 0
            outer = self
                
                # =========== Deepcache functions ==========
            def put_cache(h:torch.Tensor, timestep:int, real_timestep:float):
                """
                Registers cache
                """
                CACHE_LAST = session.CACHE_LAST
                CACHE_LAST["timestep"].add(timestep)
                assert h is not None, "Cannot cache None"
                # maybe move to cpu and load later for low vram?
                CACHE_LAST["last"] = h
                CACHE_LAST[f"timestep_{timestep}"] = h
                CACHE_LAST["real_timestep"] = real_timestep
            def get_cache(current_timestep:int, real_timestep:float) -> Optional[torch.Tensor]:
                """
                Returns the cached tensor for the given timestep and cache key.
                """
                if current_timestep < cache_enable_step:
                    session.fail_reasons['disabled'] += 1
                    session.cache_fail_count += 1
                    return None
                elif full_run_step_rate < 1:
                    session.fail_reasons['full_run_step_rate_disabled'] += 1
                    session.cache_fail_count += 1
                    return None
                elif current_timestep % full_run_step_rate == 0:
                    if f"timestep_{current_timestep}" in CACHE_LAST:
                        session.cache_success_count += 1
                        session.success_reasons['cached_exact'] += 1
                        CACHE_LAST["last"] = CACHE_LAST[f"timestep_{current_timestep}"] # update last
                        return CACHE_LAST[f"timestep_{current_timestep}"]
                    session.fail_reasons['full_run_step_rate_division'] += 1
                    session.cache_fail_count += 1
                    return None
                elif CACHE_LAST.get("real_timestep", 0) + valid_cache_timestep_range < real_timestep:
                    session.fail_reasons['cache_outdated'] += 1
                    session.cache_fail_count += 1
                    return None
                # check if cache exists
                if "last" in CACHE_LAST:
                    session.success_reasons['cached_last'] += 1
                    session.cache_success_count += 1
                    return CACHE_LAST["last"]
                session.fail_reasons['not_cached'] += 1
                session.cache_fail_count += 1
                return None

            def process_sample(*args, **kwargs):
                # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
                # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
                # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
                # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
                # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
                # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
                # After you mark the prompts, the mismatch errors will disappear.
                mark_prompt_context(kwargs.get('conditioning', []), positive=True)
                mark_prompt_context(kwargs.get('unconditional_conditioning', []), positive=False)
                mark_prompt_context(getattr(process, 'hr_c', []), positive=True)
                mark_prompt_context(getattr(process, 'hr_uc', []), positive=False)
                return process.sample_before_CN_hack(*args, **kwargs)
            # session can be used inside forward now
            def forward(self:UNetModel, x, timesteps=None, context=None, y=None, **kwargs):
                print("ControlNet forward")
                is_sdxl = y is not None and model_is_sdxl
                total_t2i_adapter_embedding = [0.0] * 4
                if is_sdxl:
                    total_controlnet_embedding = [0.0] * 10
                else:
                    total_controlnet_embedding = [0.0] * 13
                require_inpaint_hijack = False
                is_in_high_res_fix = False
                batch_size = int(x.shape[0])

                # Handle cond-uncond marker
                cond_mark, outer.current_uc_indices, context = unmark_prompt_context(context)
                outer.model.cond_mark = cond_mark
                # logger.info(str(cond_mark[:, 0, 0, 0].detach().cpu().numpy().tolist()) + ' - ' + str(outer.current_uc_indices))

                # Revision
                if is_sdxl:
                    revision_y1280 = 0

                    for param in outer.control_params:
                        if param.guidance_stopped:
                            continue
                        if param.control_model_type == ControlModelType.ReVision:
                            if param.vision_hint_count is None:
                                k = torch.Tensor([int(param.preprocessor['threshold_a'] * 1000)]).to(param.hint_cond).long().clip(0, 999)
                                param.vision_hint_count = outer.revision_q_sampler.q_sample(param.hint_cond, k)
                            revision_emb = param.vision_hint_count
                            if isinstance(revision_emb, torch.Tensor):
                                revision_y1280 += revision_emb * param.weight

                    if isinstance(revision_y1280, torch.Tensor):
                        y[:, :1280] = revision_y1280 * cond_mark[:, :, 0, 0]
                        if any('ignore_prompt' in param.preprocessor['name'] for param in outer.control_params) \
                                or (getattr(process, 'prompt', '') == '' and getattr(process, 'negative_prompt', '') == ''):
                            context = torch.zeros_like(context)

                # High-res fix
                for param in outer.control_params:
                    # select which hint_cond to use
                    if param.used_hint_cond is None:
                        param.used_hint_cond = param.hint_cond
                        param.used_hint_cond_latent = None
                        param.used_hint_inpaint_hijack = None

                    # has high-res fix
                    if isinstance(param.hr_hint_cond, torch.Tensor) and x.ndim == 4 and param.hint_cond.ndim == 4 and param.hr_hint_cond.ndim == 4:
                        _, _, h_lr, w_lr = param.hint_cond.shape
                        _, _, h_hr, w_hr = param.hr_hint_cond.shape
                        _, _, h, w = x.shape
                        h, w = h * 8, w * 8
                        if abs(h - h_lr) < abs(h - h_hr):
                            is_in_high_res_fix = False
                            if param.used_hint_cond is not param.hint_cond:
                                param.used_hint_cond = param.hint_cond
                                param.used_hint_cond_latent = None
                                param.used_hint_inpaint_hijack = None
                        else:
                            is_in_high_res_fix = True
                            if param.used_hint_cond is not param.hr_hint_cond:
                                param.used_hint_cond = param.hr_hint_cond
                                param.used_hint_cond_latent = None
                                param.used_hint_inpaint_hijack = None

                self.is_in_high_res_fix = is_in_high_res_fix
                no_high_res_control = is_in_high_res_fix and shared.opts.data.get("control_net_no_high_res_fix", False)

                # Convert control image to latent
                for param in outer.control_params:
                    if param.used_hint_cond_latent is not None:
                        continue
                    if param.control_model_type not in [ControlModelType.AttentionInjection] \
                            and 'colorfix' not in param.preprocessor['name'] \
                            and 'inpaint_only' not in param.preprocessor['name']:
                        continue
                    param.used_hint_cond_latent = outer.call_vae_using_process(process, param.used_hint_cond, batch_size=batch_size)

                # vram
                for param in outer.control_params:
                    if getattr(param.control_model, 'disable_memory_management', False):
                        continue

                    if param.control_model is not None:
                        if outer.lowvram and is_sdxl and hasattr(param.control_model, 'aggressive_lowvram'):
                            param.control_model.aggressive_lowvram()
                        elif hasattr(param.control_model, 'fullvram'):
                            param.control_model.fullvram()
                        elif hasattr(param.control_model, 'to'):
                            param.control_model.to(devices.get_device_for("controlnet"))

                # handle prompt token control
                for param in outer.control_params:
                    if no_high_res_control:
                        continue

                    if param.guidance_stopped:
                        continue

                    if param.control_model_type not in [ControlModelType.T2I_StyleAdapter]:
                        continue

                    control = param.control_model(x=x, hint=param.used_hint_cond, timesteps=timesteps, context=context)
                    control = torch.cat([control.clone() for _ in range(batch_size)], dim=0)
                    control *= param.weight
                    control *= cond_mark[:, :, :, 0]
                    context = torch.cat([context, control.clone()], dim=1)

                # handle ControlNet / T2I_Adapter
                for param in outer.control_params:
                    if no_high_res_control:
                        continue

                    if param.guidance_stopped:
                        continue

                    if param.control_model_type not in [ControlModelType.ControlNet, ControlModelType.T2I_Adapter]:
                        continue

                    # inpaint model workaround
                    x_in = x
                    control_model = param.control_model.control_model

                    if param.control_model_type == ControlModelType.ControlNet:
                        if x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9:
                            # inpaint_model: 4 data + 4 downscaled image + 1 mask
                            x_in = x[:, :4, ...]
                            require_inpaint_hijack = True

                    assert param.used_hint_cond is not None, f"Controlnet is enabled but no input image is given"

                    hint = param.used_hint_cond

                    # ControlNet inpaint protocol
                    if hint.shape[1] == 4:
                        c = hint[:, 0:3, :, :]
                        m = hint[:, 3:4, :, :]
                        m = (m > 0.5).float()
                        hint = c * (1 - m) - m

                    control = param.control_model(x=x_in, hint=hint, timesteps=timesteps, context=context, y=y)

                    if is_sdxl:
                        control_scales = [param.weight] * 10
                    else:
                        control_scales = [param.weight] * 13

                    if param.cfg_injection or param.global_average_pooling:
                        if param.control_model_type == ControlModelType.T2I_Adapter:
                            control = [torch.cat([c.clone() for _ in range(batch_size)], dim=0) for c in control]
                        control = [c * cond_mark for c in control]

                    high_res_fix_forced_soft_injection = False

                    if is_in_high_res_fix:
                        if 'canny' in param.preprocessor['name']:
                            high_res_fix_forced_soft_injection = True
                        if 'mlsd' in param.preprocessor['name']:
                            high_res_fix_forced_soft_injection = True

                    # if high_res_fix_forced_soft_injection:
                    #     logger.info('[ControlNet] Forced soft_injection in high_res_fix in enabled.')

                    if param.soft_injection or high_res_fix_forced_soft_injection:
                        # important! use the soft weights with high-res fix can significantly reduce artifacts.
                        if param.control_model_type == ControlModelType.T2I_Adapter:
                            control_scales = [param.weight * x for x in (0.25, 0.62, 0.825, 1.0)]
                        elif param.control_model_type == ControlModelType.ControlNet:
                            control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]

                    if is_sdxl and param.control_model_type == ControlModelType.ControlNet:
                        control_scales = control_scales[:10]

                    if param.advanced_weighting is not None:
                        control_scales = param.advanced_weighting

                    control = [c * scale for c, scale in zip(control, control_scales)]
                    if param.global_average_pooling:
                        control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]

                    for idx, item in enumerate(control):
                        target = None
                        if param.control_model_type == ControlModelType.ControlNet:
                            target = total_controlnet_embedding
                        if param.control_model_type == ControlModelType.T2I_Adapter:
                            target = total_t2i_adapter_embedding
                        if target is not None:
                            target[idx] = item + target[idx]

                # Replace x_t to support inpaint models
                for param in outer.control_params:
                    if not isinstance(param.used_hint_cond, torch.Tensor):
                        continue
                    if param.used_hint_cond.shape[1] != 4:
                        continue
                    if x.shape[1] != 9:
                        continue
                    if param.used_hint_inpaint_hijack is None:
                        mask_pixel = param.used_hint_cond[:, 3:4, :, :]
                        image_pixel = param.used_hint_cond[:, 0:3, :, :]
                        mask_pixel = (mask_pixel > 0.5).to(mask_pixel.dtype)
                        masked_latent = outer.call_vae_using_process(process, image_pixel, batch_size, mask=mask_pixel)
                        mask_latent = torch.nn.functional.max_pool2d(mask_pixel, (8, 8))
                        if mask_latent.shape[0] != batch_size:
                            mask_latent = torch.cat([mask_latent.clone() for _ in range(batch_size)], dim=0)
                        param.used_hint_inpaint_hijack = torch.cat([mask_latent, masked_latent], dim=1)
                        param.used_hint_inpaint_hijack.to(x.dtype).to(x.device)
                    x = torch.cat([x[:, :4, :, :], param.used_hint_inpaint_hijack], dim=1)

                # vram
                for param in outer.control_params:
                    if param.control_model is not None:
                        if outer.lowvram:
                            param.control_model.to('cpu')

                # A1111 fix for medvram.
                if shared.cmd_opts.medvram or (getattr(shared.cmd_opts, 'medvram_sdxl', False) and is_sdxl):
                    try:
                        # Trigger the register_forward_pre_hook
                        outer.sd_ldm.model()
                    except:
                        pass

                # Clear attention and AdaIn cache
                for module in outer.attn_module_list:
                    module.bank = []
                    module.style_cfgs = []
                for module in outer.gn_module_list:
                    module.mean_bank = []
                    module.var_bank = []
                    module.style_cfgs = []

                # Handle attention and AdaIn control
                for param in outer.control_params:
                    if no_high_res_control:
                        continue

                    if param.guidance_stopped:
                        continue

                    if param.used_hint_cond_latent is None:
                        continue

                    if param.control_model_type not in [ControlModelType.AttentionInjection]:
                        continue

                    ref_xt = predict_q_sample(outer.sd_ldm, param.used_hint_cond_latent, torch.round(timesteps.float()).long())

                    # Inpaint Hijack
                    if x.shape[1] == 9:
                        ref_xt = torch.cat([
                            ref_xt,
                            torch.zeros_like(ref_xt)[:, 0:1, :, :],
                            param.used_hint_cond_latent
                        ], dim=1)

                    outer.current_style_fidelity = float(param.preprocessor['threshold_a'])
                    outer.current_style_fidelity = max(0.0, min(1.0, outer.current_style_fidelity))

                    if is_sdxl:
                        # sdxl's attention hacking is highly unstable.
                        # We have no other methods but to reduce the style_fidelity a bit.
                        # By default, 0.5 ** 3.0 = 0.125
                        outer.current_style_fidelity = outer.current_style_fidelity ** 3.0

                    if param.cfg_injection:
                        outer.current_style_fidelity = 1.0
                    elif param.soft_injection or is_in_high_res_fix:
                        outer.current_style_fidelity = 0.0

                    control_name = param.preprocessor['name']

                    if control_name in ['reference_only', 'reference_adain+attn']:
                        outer.attention_auto_machine = AutoMachine.Write
                        outer.attention_auto_machine_weight = param.weight

                    if control_name in ['reference_adain', 'reference_adain+attn']:
                        outer.gn_auto_machine = AutoMachine.Write
                        outer.gn_auto_machine_weight = param.weight

                    if is_sdxl:
                        outer.original_forward(
                            x=ref_xt.to(devices.dtype_unet),
                            timesteps=timesteps.to(devices.dtype_unet),
                            context=context.to(devices.dtype_unet),
                            y=y
                        )
                    else:
                        outer.original_forward(
                            x=ref_xt.to(devices.dtype_unet),
                            timesteps=timesteps.to(devices.dtype_unet),
                            context=context.to(devices.dtype_unet)
                        )

                    outer.attention_auto_machine = AutoMachine.Read
                    outer.gn_auto_machine = AutoMachine.Read

                # U-Net Encoder
                if not isinstance(timesteps, (int, float)):
                    real_timestep = timesteps[0].item()
                else:
                    real_timestep = timesteps
                hs = []
                with th.no_grad():
                    t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
                    emb = self.time_embed(t_emb)

                    if is_sdxl:
                        assert y.shape[0] == x.shape[0]
                        emb = emb + self.label_emb(y)
                    cached_h = get_cache(session.enumerated_timestep["value"], real_timestep)
                    

                    h = x
                    for i, module in enumerate(self.input_blocks):
                        self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                        t2i_injection = [3, 5, 8] if is_sdxl else [2, 5, 8, 11]
                        if cached_h is not None and use_cache_cond() and id > caching_level:
                            if i in t2i_injection:
                                total_t2i_adapter_embedding.pop(0)
                        else:
                            h = module(h, emb, context)
                            if i in t2i_injection:
                                h = aligned_adding(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)
                            hs.append(h)
                # U-Net Middle Block
                if not use_cache_cond():
                    self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                    h = self.middle_block(h, emb, context)
                    session.log_skip('run_before_cache_middle_block_cnet')
                    h = aligned_adding(h, total_controlnet_embedding.pop(), require_inpaint_hijack)
                    if len(total_t2i_adapter_embedding) > 0 and is_sdxl:
                        h = aligned_adding(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)
                relative_cache_level = len(self.output_blocks) - caching_level - 1
                # U-Net Decoder
                for i, module in enumerate(self.output_blocks):
                    if cached_h is not None and use_cache_cond() and idx == relative_cache_level:
                        # use cache
                        h = cached_h
                    elif cache_cond() and idx == relative_cache_level:
                        # put cache
                        put_cache(h, self.enumerated_timestep["value"], real_timestep)
                    else:
                        total_controlnet_embedding.pop()
                        continue
                    self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                    h = th.cat([h, aligned_adding(hs.pop(), total_controlnet_embedding.pop(), require_inpaint_hijack)], dim=1)
                    h = module(h, emb, context)

                # U-Net Output
                h = h.type(x.dtype)
                h = self.out(h)

                # Post-processing for color fix
                for param in outer.control_params:
                    if param.used_hint_cond_latent is None:
                        continue
                    if 'colorfix' not in param.preprocessor['name']:
                        continue

                    k = int(param.preprocessor['threshold_a'])
                    if is_in_high_res_fix and not no_high_res_control:
                        k *= 2

                    # Inpaint hijack
                    xt = x[:, :4, :, :]

                    x0_origin = param.used_hint_cond_latent
                    t = torch.round(timesteps.float()).long()
                    x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
                    x0 = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)

                    if '+sharp' in param.preprocessor['name']:
                        detail_weight = float(param.preprocessor['threshold_b']) * 0.01
                        neg = detail_weight * blur(x0, k) + (1 - detail_weight) * x0
                        x0 = cond_mark * x0 + (1 - cond_mark) * neg

                    eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

                    w = max(0.0, min(1.0, float(param.weight)))
                    h = eps_prd * w + h * (1 - w)

                # Post-processing for restore
                for param in outer.control_params:
                    if param.used_hint_cond_latent is None:
                        continue
                    if 'inpaint_only' not in param.preprocessor['name']:
                        continue
                    if param.used_hint_cond.shape[1] != 4:
                        continue

                    # Inpaint hijack
                    xt = x[:, :4, :, :]

                    mask = param.used_hint_cond[:, 3:4, :, :]
                    mask = torch.nn.functional.max_pool2d(mask, (10, 10), stride=(8, 8), padding=1)

                    x0_origin = param.used_hint_cond_latent
                    t = torch.round(timesteps.float()).long()
                    x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
                    x0 = x0_prd * mask + x0_origin * (1 - mask)
                    eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

                    w = max(0.0, min(1.0, float(param.weight)))
                    h = eps_prd * w + h * (1 - w)

                return h

            def move_all_control_model_to_cpu():
                for param in getattr(outer, 'control_params', []) or []:
                    if isinstance(param.control_model, torch.nn.Module):
                        param.control_model.to("cpu")

            def forward_webui(*args, **kwargs):
                # webui will handle other compoments 
                try:
                    if shared.cmd_opts.lowvram:
                        lowvram.send_everything_to_cpu()
                    return forward(*args, **kwargs)
                except Exception as e:
                    move_all_control_model_to_cpu()
                    raise e
                finally:
                    if outer.lowvram:
                        move_all_control_model_to_cpu()

            def hacked_basic_transformer_inner_forward(self, x, context=None):
                x_norm1 = self.norm1(x)
                self_attn1 = None
                if self.disable_self_attn:
                    # Do not use self-attention
                    self_attn1 = self.attn1(x_norm1, context=context)
                else:
                    # Use self-attention
                    self_attention_context = x_norm1
                    if outer.attention_auto_machine == AutoMachine.Write:
                        if outer.attention_auto_machine_weight > self.attn_weight:
                            self.bank.append(self_attention_context.detach().clone())
                            self.style_cfgs.append(outer.current_style_fidelity)
                    if outer.attention_auto_machine == AutoMachine.Read:
                        if len(self.bank) > 0:
                            style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                            self_attn1_uc = self.attn1(x_norm1, context=torch.cat([self_attention_context] + self.bank, dim=1))
                            self_attn1_c = self_attn1_uc.clone()
                            if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                                self_attn1_c[outer.current_uc_indices] = self.attn1(
                                    x_norm1[outer.current_uc_indices],
                                    context=self_attention_context[outer.current_uc_indices])
                            self_attn1 = style_cfg * self_attn1_c + (1.0 - style_cfg) * self_attn1_uc
                        self.bank = []
                        self.style_cfgs = []
                    if self_attn1 is None:
                        self_attn1 = self.attn1(x_norm1, context=self_attention_context)

                x = self_attn1.to(x.dtype) + x
                x = self.attn2(self.norm2(x), context=context) + x
                x = self.ff(self.norm3(x)) + x
                return x

            def hacked_group_norm_forward(self, *args, **kwargs):
                eps = 1e-6
                x = self.original_forward_cn_hijack(*args, **kwargs)
                y = None
                if outer.gn_auto_machine == AutoMachine.Write:
                    if outer.gn_auto_machine_weight > self.gn_weight:
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                        self.style_cfgs.append(outer.current_style_fidelity)
                if outer.gn_auto_machine == AutoMachine.Read:
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                        var_acc = sum(self.var_bank) / float(len(self.var_bank))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        y_uc = (((x - mean) / std) * std_acc) + mean_acc
                        y_c = y_uc.clone()
                        if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                            y_c[outer.current_uc_indices] = x.to(y_c.dtype)[outer.current_uc_indices]
                        y = style_cfg * y_c + (1.0 - style_cfg) * y_uc
                    self.mean_bank = []
                    self.var_bank = []
                    self.style_cfgs = []
                if y is None:
                    y = x
                return y.to(x.dtype)

            if getattr(process, 'sample_before_CN_hack', None) is None:
                process.sample_before_CN_hack = process.sample
            process.sample = process_sample

            model._original_forward = model.forward
            outer.original_forward = model.forward
            model.forward = forward_webui.__get__(model, UNetModel)

            if model_is_sdxl:
                register_schedule(sd_ldm)
                outer.revision_q_sampler = AbstractLowScaleModel()

            need_attention_hijack = False

            for param in outer.control_params:
                if param.control_model_type in [ControlModelType.AttentionInjection]:
                    need_attention_hijack = True

            all_modules = torch_dfs(model)

            if need_attention_hijack:
                attn_modules = [module for module in all_modules if isinstance(module, BasicTransformerBlock) or isinstance(module, BasicTransformerBlockSGM)]
                attn_modules = sorted(attn_modules, key=lambda x: - x.norm1.normalized_shape[0])

                for i, module in enumerate(attn_modules):
                    if getattr(module, '_original_inner_forward_cn_hijack', None) is None:
                        module._original_inner_forward_cn_hijack = module._forward
                    module._forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                    module.bank = []
                    module.style_cfgs = []
                    module.attn_weight = float(i) / float(len(attn_modules))

                gn_modules = [model.middle_block]
                model.middle_block.gn_weight = 0

                if model_is_sdxl:
                    input_block_indices = [4, 5, 7, 8]
                    output_block_indices = [0, 1, 2, 3, 4, 5]
                else:
                    input_block_indices = [4, 5, 7, 8, 10, 11]
                    output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7]

                for w, i in enumerate(input_block_indices):
                    module = model.input_blocks[i]
                    module.gn_weight = 1.0 - float(w) / float(len(input_block_indices))
                    gn_modules.append(module)

                for w, i in enumerate(output_block_indices):
                    module = model.output_blocks[i]
                    module.gn_weight = float(w) / float(len(output_block_indices))
                    gn_modules.append(module)

                for i, module in enumerate(gn_modules):
                    if getattr(module, 'original_forward_cn_hijack', None) is None:
                        module.original_forward_cn_hijack = module.forward
                    module.forward = hacked_group_norm_forward.__get__(module, torch.nn.Module)
                    module.mean_bank = []
                    module.var_bank = []
                    module.style_cfgs = []
                    module.gn_weight *= 2

                outer.attn_module_list = attn_modules
                outer.gn_module_list = gn_modules
            else:
                for module in enumerate(all_modules):
                    _original_inner_forward_cn_hijack = getattr(module, '_original_inner_forward_cn_hijack', None)
                    original_forward_cn_hijack = getattr(module, 'original_forward_cn_hijack', None)
                    if _original_inner_forward_cn_hijack is not None:
                        module._forward = _original_inner_forward_cn_hijack
                    if original_forward_cn_hijack is not None:
                        module.forward = original_forward_cn_hijack
                outer.attn_module_list = []
                outer.gn_module_list = []

            scripts.script_callbacks.on_cfg_denoiser(self.guidance_schedule_handler)
        return get_hook
