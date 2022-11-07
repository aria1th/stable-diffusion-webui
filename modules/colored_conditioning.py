from ast import literal_eval

import torch
import torch.nn.functional as F

# lets have fixed color codes, and just call it 'red', 'orange', etc.
# Then we can do this : TextBox : //('red', "some long string to be parsed"), ("blue", "cat maybe")// or // 'red', "some red cat".
# case 'red', 'some red cat' -> 1. cover with [], 2. literal eval, 3. assert all components is string, 4. accept
# case others -> 1. cover with [], 2. literal eval, 3. check if everything matches [(str, str),...]. 4. accept
# accept -> define color code dictionary. At default, it will use fixed values from json. so {'red' : (255, 0, 0), ...}. If key is in dict, use. If key is not in dict, reject.

defined_colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}


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
            result_dict[result[0].lower()] = result[1]
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
                result_dict[key.lower()] = value
            elif isinstance(key, int):
                assert 0 <= result[0] <= 16777215, "Colorcode exceends valid range!"
                result_dict[result[0]] = totuple(value)
            else:
                assert len(key) == 3 and all(0 <= x <= 255 for x in key), "Color code is invalid!"
                result_dict[key] = value
    return result_dict


def resize_to(image:torch.Tensor, width:int, height:int):
    '''
        Resizes image tensor or array-like object, to wanted width and height.
    '''
    return F.interpolate(image, (width, height))

