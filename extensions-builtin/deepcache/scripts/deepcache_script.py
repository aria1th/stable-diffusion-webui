from modules import scripts, script_callbacks, shared, processing
from deepcache import DeepCacheSession, DeepCacheParams
from scripts.deepcache_xyz import add_axis_options
from scripts.deepcache_controlnet import UNetHookModified

class ScriptDeepCache(scripts.Script):
    priority = 100
    name = "DeepCache"
    session: DeepCacheSession = None
    
    hook_controlnet_func = None
    hook_cls_ref = None

    def title(self):
        return self.name

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_deepcache_params(self, steps: int) -> DeepCacheParams:
        return DeepCacheParams(
            cache_in_level=shared.opts.deepcache_cache_resnet_level,
            cache_enable_step=int(shared.opts.deepcache_cache_enable_step_percentage * steps),
            full_run_step_rate=shared.opts.deepcache_full_run_step_rate,
        )

    def process_batch_tmp(self, p:processing.StableDiffusionProcessing, *args, **kwargs):
        print("DeepCache process")
        self.detach_deepcache()
        if shared.opts.deepcache_enable:
            self.configure_deepcache(self.get_deepcache_params(p.steps), p)

    def before_hr(self, p:processing.StableDiffusionProcessing, *args):
        print("DeepCache before_hr")
        if self.session is not None:
            self.session.enumerated_timestep["value"] = -1 # reset enumerated timestep
        if not shared.opts.deepcache_hr_reuse:
            self.detach_deepcache()
        if shared.opts.deepcache_enable:
            if self.session is None:
                self.configure_deepcache(self.get_deepcache_params(getattr(p, 'hr_second_pass_steps', 0) or p.steps), p) # use second pass steps if available

    def postprocess_batch_tmp(self, p:processing.StableDiffusionProcessing, *args, **kwargs):
        print("DeepCache postprocess")
        if UNetHookModified.session is not None:
            UNetHookModified.session.report()
            UNetHookModified.session = None
        self.detach_deepcache()
    
    def before_process_batch(self, p, *args, **kwargs):
        print("DeepCache before_process_batch")
        if self.session is not None:
            self.session.enumerated_timestep["value"] = -1 # reset enumerated timestep
    
    def before_process(self, p, *args, **kwargs):
        print("DeepCache before_process")
        self.detach_deepcache()
        if shared.opts.deepcache_enable:
            self.configure_deepcache(self.get_deepcache_params(p.steps), p)

    def configure_deepcache(self, params:DeepCacheParams, p:processing.StableDiffusionProcessing):
        controlnet_hooked = UNetHookModified.patch_controlnet(params=params, p=p)
        if controlnet_hooked:
            print("Controlnet hooked")
            hook_cls, hook_func, session = controlnet_hooked
            self.hook_controlnet_func = hook_func
            self.hook_cls_ref = hook_cls
            self.session = session
            return
        if self.session is None:
            self.session = DeepCacheSession()
        print("DeepCache for vanilla unet hooked")
        self.session.deepcache_hook_model(
            shared.sd_model.model.diffusion_model, #unet_model
            params
        )

    def detach_deepcache(self):
        print("Detaching DeepCache")
        if self.hook_controlnet_func is not None:
            print("DeepCache Controlnet unhooked")
            self.hook_cls_ref.hook = self.hook_controlnet_func
            self.hook_controlnet_func = None
            self.hook_cls_ref = None
        if self.session is None:
            return
        print("DeepCache unhooked")
        self.session.report()
        self.session.detach()
        self.session = None

def on_ui_settings():
    import gradio as gr
    options = {
        "deepcache_explanation": shared.OptionHTML("""
    <a href='https://github.com/horseee/DeepCache'>DeepCache</a> optimizes by caching the results of mid-blocks, which is known for high level features, and reusing them in the next forward pass.
    """),
        "deepcache_enable": shared.OptionInfo(False, "Enable DeepCache").info("noticeable change in details of the generated picture"),
        "deepcache_cache_resnet_level": shared.OptionInfo(0, "Cache Resnet level", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}).info("Deeper = fewer layers cached"),
        "deepcache_cache_enable_step_percentage": shared.OptionInfo(0.4, "Deepcaches is enabled after the step percentage", gr.Slider, {"minimum": 0, "maximum": 1}).info("Percentage of initial steps to disable deepcache"),
        "deepcache_full_run_step_rate": shared.OptionInfo(5, "Refreshes caches when step is divisible by number", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("5 = refresh caches every 5 steps"),
        "deepcache_hr_reuse" : shared.OptionInfo(False, "Reuse for HR").info("Reuses cache information for HR generation"),
    }
    for name, opt in options.items():
        opt.section = ('deepcache', "DeepCache")
        shared.opts.add_option(name, opt)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_ui(add_axis_options)
