from ast import literal_eval

import torch
import torch.nn.functional as F
from PIL import Image

# lets have fixed color codes, and just call it 'red', 'orange', etc.
# Then we can do this : TextBox : //('red', "some long string to be parsed"), ("blue", "cat maybe")// or // 'red', "some red cat".
# case 'red', 'some red cat' -> 1. cover with [], 2. literal eval, 3. assert all components is string, 4. accept
# case others -> 1. cover with [], 2. literal eval, 3. check if everything matches [(str, str),...]. 4. accept
# accept -> define color code dictionary. At default, it will use fixed values from json. so {'red' : (255, 0, 0), ...}. If key is in dict, use. If key is not in dict, reject.

defined_colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

defined_importance = {(255, 0, 0) : 1, (0, 255, 0) : 0.5, (0, 0, 255) : 0.5}

def totuple(colorcode: int):
    RG, B = divmod(colorcode, 256)
    R, G = divmod(colorcode, 256)
    return R, G, B


# [0x12415, "something"] [(0, 255, 255), "something"], [(), (),...], ['Red', "something"], ...
def parse_code(text: str):
    text = "[" + text + "]"
    result_dict = {}
    try:
        result = literal_eval(text)
    except:
        raise RuntimeError(f"Cannot parse {text}!")
    if len(result) == 2:
        assert isinstance(result[0], (str, int, tuple)) and isinstance(result[1], str), "Cannot parse text : format should be 'color', 'prompt'!"
        if isinstance(result[0], str):
            assert isinstance(result[0], str) and result[0].lower() in defined_colors, "Color is not defined!"
            result_dict[defined_colors[result[0].lower()]] = result[1]
        elif isinstance(result[0], int):
            assert 0 <= result[0] <= 16777215, "Colorcode exceends valid range!"
            result_dict[result[0]] = totuple(result[1])
        else:
            assert len(result[0]) == 3 and all(0 <= x <= 255 for x in result[0]), "Color code is invalid!"
            result_dict[result[0]] = result[1]
    else:
        assert all(type(x) is tuple and len(x) == 2 for x in result)
        for key, value in result:
            if isinstance(key, str):
                assert isinstance(key, str) and key.lower() in defined_colors, "Color is not defined!"
                result_dict[defined_colors[key.lower()]] = value
            elif isinstance(key, int):
                assert 0 <= result[0] <= 16777215, "Colorcode exceends valid range!"
                result_dict[result[0]] = totuple(value)
            else:
                assert len(key) == 3 and all(0 <= x <= 255 for x in key), "Color code is invalid!"
                result_dict[key] = value
    return result_dict


def resize_to(image:torch.Tensor, width:int, height:int):
    """
        Resizes image tensor or array-like object, to wanted width and height.
    """
    return F.interpolate(image, (width, height), mode="linear")


def process_condition(self, prompt, reference_image_info, reference_image_path, optional_decay=1):
    if not reference_image_info or reference_image_path:
        return {}
    image_dict = parse_code(reference_image_info)
    image_tensor: torch.Tensor = torch.tensor(Image.open(reference_image_path).convert('RGB'))
    original_token, max_token_count, target_count = self.sd_model.cond_stage_model.hijack.tokenize(prompt)[0]
    color_dict = {}
    for colors, prompts in image_dict.items():
        mask: torch.Tensor = torch.tensor((image_tensor == colors), dtype=torch.float32) * defined_importance.get(colors, 0) # do we need importance? or just global?
        remade_batch_tokens, token_count, target_token_count = self.sd_model.cond_stage_model.hijack.tokenize(prompts)
        if token_count < max_token_count:
            raise RuntimeError(f"Local token count exceeds Global token count! : {prompts}")
        print(remade_batch_tokens)  # check
        color_dict[colors] = (mask, remade_batch_tokens)
    w, h = image_tensor.shape
    weights = {i: process_patch_token(color_dict, w, h, i, max_token_count, optional_decay) for i in (64, 256, 1024, 4096)}
    return weights


def process_patch_token(color_dict, w, h, r, max_count, optional_decay = 1):
    init_tensor = torch.zeros((w // r * h // r, max_count), dtype = torch.float32) # or sparse?
    for mask, token_list in color_dict.values():
        for tokens in token_list:
            init_tensor[:, tokens] += resize_to(mask, w//r, h//r).reshape(-1) * optional_decay #flatten add.
            #conditional? only work when global token contains it?
    return init_tensor



