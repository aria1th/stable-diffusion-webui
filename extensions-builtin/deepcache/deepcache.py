from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import torch
from ldm.modules.diffusionmodules.openaimodel import timestep_embedding
from scripts.forward_timestep_embed_patch import forward_timestep_embed

from logging import getLogger
@dataclass
class DeepCacheParams:
    cache_in_level: int = 0
    cache_enable_step: int = 0
    full_run_step_rate: int = 5
    # cache_latents_cpu: bool = False
    # cache_latents_hires: bool = False

class DeepCacheStandAlone:
    """
    @source https://github.com/horseee/DeepCache
    Standalone version of DeepCache, which can be used without the DeepCacheScript.
    
    Code Snippet:
    ```python
        # U-Net Encoder

        for i, module in enumerate(self.input_blocks):
            if session.should_skip_unet_in(i):
                continue
            h = forward_timestep_embed(module, h, emb, context)
            hs.append(h)
        if not session.should_skip_unet_middle():
            h = forward_timestep_embed(self.middle_block, h, emb, context)
        # U-Net Decoder
        total_out_blocks = len(self.output_blocks)
        for idx, module in enumerate(self.output_blocks):
            if session.cond_skip_unet_out(idx, total_out_blocks):
                continue
            if session.should_load_unet_out(idx, total_out_blocks):
                h = session.get_cache()
            else:
                hsp = hs.pop()
                h = torch.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                h = forward_timestep_embed(module, h, emb, context, output_shape=output_shape)
                if session.should_cache_unet_out(idx, total_out_blocks):
                    session.put_cache(h)
    """
    def __init__(self, params:dict = None, enable:bool = True) -> None:
        if params is None:
            params = {}
        self.cache_in_level = params.get('cache_in_level', 0)
        self.cache_enable_step = params.get('cache_enable_step', 0)
        self.full_run_step_rate = params.get('full_run_step_rate', 5)
        self.cache = {
            "timestep": {0}
        }
        self.timestep_accumulator = 0
        self.enable = enable
        self.debug_mode = True
        self.skip_statistics = {} # will log in_true / in_false / out_true / out_false / middle_true / middle_false
    
    def report(self):
        """
        Reports the cache statistics.
        """
        print("DeepCache : report")
        # total, in, out, middle success rate and count
        total = sum(self.skip_statistics.values())
        in_sum = self.skip_statistics.get('in_true', 0) + self.skip_statistics.get('in_false', 0)
        out_sum = self.skip_statistics.get('out_true', 0) + self.skip_statistics.get('out_false', 0)
        middle_sum = self.skip_statistics.get('middle_true', 0) + self.skip_statistics.get('middle_false', 0)
        print(f"DeepCache : total {total} in {in_sum} out {out_sum} middle {middle_sum}")
        print("Success rates")
        print(f"IN : {self.skip_statistics.get('in_true', 0) / in_sum if in_sum > 0 else 0}, total {in_sum}")
        print(f"OUT : {self.skip_statistics.get('out_true', 0) / out_sum if out_sum > 0 else 0}, total {out_sum}")
        print(f"MIDDLE : {self.skip_statistics.get('middle_true', 0) / middle_sum if middle_sum > 0 else 0}, total {middle_sum}")
        
    def should_activate(self):
        """
        Returns if DeepCache should be activated.
        """
        if not self.enable:
            self.debug_print("DeepCache : Disabled")
            return False
        if self.full_run_step_rate < 1:
            self.debug_print(f"DeepCache : full_run_step_rate {self.full_run_step_rate} < 1")
            return False
        if not self.timestep_accumulator > self.cache_enable_step:
            self.debug_print(f"DeepCache : timestep {self.timestep_accumulator} <= cache_enable_step {self.cache_enable_step}")
            return False
        if self.timestep_accumulator % self.full_run_step_rate == 0:
            self.debug_print(f"DeepCache : timestep {self.timestep_accumulator} % full_run_step_rate {self.full_run_step_rate} == 0")
            return False
        return True
    def should_cache_unet_out(self, index, total_out_blocks:int) -> bool:
        """
        Returns if current in block should be cached.
        The function should be called exclusively with should_load_unet_out branch, to execute forward+cache / load+continue.
        Usage : if self.should_cache_unet_out(index): h = model.forward();self.put_cache(h)
        """
        if not self.should_activate():
            return False
        return index == total_out_blocks - self.cache_in_level - 1
    def should_load_unet_out(self, index, total_out_blocks:int) -> bool:
        """
        Returns if current out block should be loaded.
        The function should be called before the forward_timestep_embed call, to replace the input with the cached tensor.
        The function should be called before should_cache_unet_out call.
        Usage : if should_load_unet_out(index, len(unet.output_blocks)): h = self.get_cache()
        """
        return self.get_cache() is not None and index == total_out_blocks - self.cache_in_level - 1

    def cond_skip_unet_in(self, index) -> bool:
        """
        Returns if current in block should be skipped.
        The function should be called before the forward_timestep_embed call.
        Usage : if self.cond_skip_unet_in(index): continue
        """
        result = self.get_cache() is not None and index > self.cache_in_level
        if result:
            self.skip_statistics['in_true'] = self.skip_statistics.get('in_true', 0) + 1
        else:
            self.skip_statistics['in_false'] = self.skip_statistics.get('in_false', 0) + 1
    def cond_skip_unet_out(self, index, total_out_blocks) -> bool:
        """
        Returns if current out block should be skipped.
        The function should be called before should_load_net_out and should_cache_unet_out call.
        Usage : if self.cond_skip_unet_out(index, len(unet.output_blocks)): continue
        """
        result = self.get_cache() is not None and index < total_out_blocks - self.cache_in_level - 1
        if result:
            self.skip_statistics['out_true'] = self.skip_statistics.get('out_true', 0) + 1
        else:
            self.skip_statistics['out_false'] = self.skip_statistics.get('out_false', 0) + 1
    def cond_skip_unet_middle(self) -> bool:
        """
        Returns if middle block should be skipped.
        Usage : if self.cond_skip_unet_middle(): continue
        """
        timestep = self.timestep_accumulator
        result = timestep > self.cache_enable_step and self.get_cache() is not None and timestep % self.full_run_step_rate != 0
        if result:
            self.skip_statistics['middle_true'] = self.skip_statistics.get('middle_true', 0) + 1
        else:
            self.skip_statistics['middle_false'] = self.skip_statistics.get('middle_false', 0) + 1
    def put_cache(self, h):
        """
        Registers cache
        Usage : if self.should_cache_unet_out(index): self.put_cache(h)
        """
        timestep = self.timestep_accumulator
        self.cache["timestep"].add(timestep)
        assert h is not None, "Cannot cache None"
        # maybe move to cpu and load later for low vram?
        self.cache["last"] = h
        for _i in range(self.full_run_step_rate):
            # register for each step too
            self.cache[f"timestep_{timestep + _i}"] = h
        self.debug_print(f"DeepCache : put cache for timestep {timestep}")
    def debug_print(self, *args, **kwargs):
        if self.debug_mode:
            print(*args, **kwargs)
    def get_cache(self):
        """
        Returns the cached tensor for the given timestep and cache key.
        Usage : if self.should_load_unet_out(index, len(unet.output_blocks)): h = self.get_cache()
        """
        if not self.should_activate():
            self.debug_print("DeepCache : Disabled")
            return None
        current_timestep = self.timestep_accumulator
        if current_timestep < self.cache_enable_step:
            self.debug_print(f"DeepCache : timestep {current_timestep} < cache_enable_step {self.cache_enable_step}")
            return None
        elif self.full_run_step_rate < 1:
            self.debug_print(f"DeepCache : full_run_step_rate {self.full_run_step_rate} < 1")
            return None
        elif current_timestep % self.full_run_step_rate != 0:
            if f"timestep_{current_timestep}" in self.cache:
                self.cache["last"] = self.cache[f"timestep_{current_timestep}"] # update last
                self.debug_print(f"DeepCache : load cache for timestep {current_timestep}")
                return self.cache[f"timestep_{current_timestep}"]
            self.debug_print(f"DeepCache : cache for timestep {current_timestep} not found")
        self.debug_print(f"DeepCache : Step is divisible, running full forward")
        return None
    def increment_timestep(self):
        """
        Increments the timestep accumulator. Should be called after each forward pass.
        """
        self.timestep_accumulator += 1
    def reset_timestep(self):
        """
        Resets the timestep accumulator. Should be called after each generation / maybe before hires.fix
        """
        self.timestep_accumulator = 0
    def clear_cache(self):
        """
        Clears the cache. Should be called after each generation.
        """
        self.cache.clear()
        self.cache = {
            "timestep": {0}
        }
        self.timestep_accumulator = 0
        self.skip_statistics.clear()

class DeepCacheSession:
    """
    Session for DeepCache, which holds cache data and provides functions for hooking the model.
    """
    def __init__(self) -> None:
        self.CACHE_LAST = {"timestep": {0}}
        self.stored_forward = None
        self.unet_reference = None
        self.cache_success_count = 0
        self.cache_fail_count = 0
        self.fail_reasons = defaultdict(int)
        self.success_reasons = defaultdict(int)
        self.enumerated_timestep = {"value": -1}

    def log_skip(self, reason:str = 'disabled_by_default'):
        self.fail_reasons[reason] += 1
        self.cache_fail_count += 1
    
    def load_params(self, params:DeepCacheParams):
        """
        Loads the given params into the session.
        """
        self.params = params

    def report(self):
        # report cache success rate
        total = self.cache_success_count + self.cache_fail_count
        if total == 0:
            return
        logger = getLogger()
        logger.log("DeepCache Information :")
        for fail_reasons, count in self.fail_reasons.items():
            logger.log(f"  {fail_reasons}: {count}")
        for success_reasons, count in self.success_reasons.items():
            logger.log(f"  {success_reasons}: {count}")

    def deepcache_hook_model(self, unet, params:DeepCacheParams):
        """
        Hooks the given unet model to use DeepCache.
        """
        caching_level = params.cache_in_level
        # caching level 0 = no caching, idx for resnet layers
        cache_enable_step = params.cache_enable_step
        full_run_step_rate = params.full_run_step_rate # '5' means run full model every 5 steps
        if full_run_step_rate < 1:
            print(f"DeepCache disabled due to full_run_step_rate {full_run_step_rate} < 1 but enabled by user")
            return # disabled
        if getattr(unet, '_deepcache_hooked', False):
            return  # already hooked
        CACHE_LAST = self.CACHE_LAST
        self.stored_forward = unet.forward
        self.enumerated_timestep["value"] = -1
        valid_caching_in_level = min(caching_level, len(unet.input_blocks) - 1)
        valid_caching_out_level = min(valid_caching_in_level, len(unet.output_blocks) - 1)
        # set to max if invalid
        caching_level = valid_caching_out_level
        valid_cache_timestep_range = 50 # total 1000, 50
        def put_cache(h:torch.Tensor, timestep:int, real_timestep:float):
            """
            Registers cache
            """
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
                self.fail_reasons['disabled'] += 1
                self.cache_fail_count += 1
                return None
            elif full_run_step_rate < 1:
                self.fail_reasons['full_run_step_rate_disabled'] += 1
                self.cache_fail_count += 1
                return None
            elif current_timestep % full_run_step_rate == 0:
                if f"timestep_{current_timestep}" in CACHE_LAST:
                    self.cache_success_count += 1
                    self.success_reasons['cached_exact'] += 1
                    CACHE_LAST["last"] = CACHE_LAST[f"timestep_{current_timestep}"] # update last
                    return CACHE_LAST[f"timestep_{current_timestep}"]
                self.fail_reasons['full_run_step_rate_division'] += 1
                self.cache_fail_count += 1
                return None
            elif CACHE_LAST.get("real_timestep", 0) + valid_cache_timestep_range < real_timestep:
                self.fail_reasons['cache_outdated'] += 1
                self.cache_fail_count += 1
                return None
            # check if cache exists
            if "last" in CACHE_LAST:
                self.success_reasons['cached_last'] += 1
                self.cache_success_count += 1
                return CACHE_LAST["last"]
            self.fail_reasons['not_cached'] += 1
            self.cache_fail_count += 1
            return None
        def hijacked_unet_forward(x, timesteps=None, context=None, y=None, **kwargs):
            cache_cond = lambda : self.enumerated_timestep["value"] % full_run_step_rate == 0 or self.enumerated_timestep["value"] > cache_enable_step
            use_cache_cond = lambda : self.enumerated_timestep["value"] > cache_enable_step and self.enumerated_timestep["value"] % full_run_step_rate != 0
            nonlocal CACHE_LAST
            assert (y is not None) == (
                hasattr(unet, 'num_classes') and unet.num_classes is not None #v2 or xl
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(unet.dtype)
            emb = unet.time_embed(t_emb)
            if hasattr(unet, 'num_classes') and unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)
            real_timestep = timesteps[0].item()
            h = x.type(unet.dtype)
            cached_h = get_cache(self.enumerated_timestep["value"], real_timestep)
            for id, module in enumerate(unet.input_blocks):
                self.log_skip('run_before_cache_input_block')
                h = forward_timestep_embed(module, h, emb, context)
                hs.append(h)
                if cached_h is not None and use_cache_cond() and id == caching_level:
                    break
            if not use_cache_cond():
                self.log_skip('run_before_cache_middle_block')
                h = forward_timestep_embed(unet.middle_block, h, emb, context)
            relative_cache_level = len(unet.output_blocks) - caching_level - 1
            for idx, module in enumerate(unet.output_blocks):
                if cached_h is not None and use_cache_cond() and idx == relative_cache_level:
                    # use cache
                    h = cached_h
                elif cache_cond() and idx == relative_cache_level:
                    # put cache
                    put_cache(h, self.enumerated_timestep["value"], real_timestep)
                elif cached_h is not None and use_cache_cond() and idx < relative_cache_level:
                    # skip, h is already cached
                    continue
                hsp = hs.pop()
                h = torch.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                h = forward_timestep_embed(module, h, emb, context, output_shape=output_shape)
            h = h.type(x.dtype)
            self.enumerated_timestep["value"] += 1
            if unet.predict_codebook_ids:
                return unet.id_predictor(h)
            else:
                return unet.out(h)
        unet.forward = hijacked_unet_forward
        unet._deepcache_hooked = True
        self.unet_reference = unet

    def detach(self):
        if self.unet_reference is None:
            return
        if not getattr(self.unet_reference, '_deepcache_hooked', False):
            return
        # detach
        self.unet_reference.forward = self.stored_forward
        self.unet_reference._deepcache_hooked = False
        self.unet_reference = None
        self.stored_forward = None
        self.CACHE_LAST.clear()
        self.cache_fail_count = self.cache_success_count = 0#
