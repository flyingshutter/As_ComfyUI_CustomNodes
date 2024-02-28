import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys, os


MAX_RESOLUTION = 8192


class MaskToImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",),}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, mask):
        d2, d3 = mask.size()
        print("MASK SIZE:", mask.size())

        new_image = torch.zeros(
            (1, d2, d3, 3),
            dtype=torch.float32,
        )
        new_image[0, :, :, 0] = mask
        new_image[0, :, :, 1] = mask
        new_image[0, :, :, 2] = mask

        print("MaskSize", mask.size())
        print("Tyep New img", type(new_image))

        return (new_image,)


class ImageToMask_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, image):
        return (image.squeeze().mean(2),)


class LatentMix_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "blend": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, samples_to, samples_from, blend):
        samples_out = samples_to.copy()
        s_to = samples_to["samples"].clone()
        s_from = samples_from["samples"].clone()
        samples_out["samples"] = s_to * blend / 100 + s_from * (100 - blend) / 100
        return (samples_out,)


class LatentAdd_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, samples_to, samples_from):
        samples_out = samples_to.copy()
        s_to = samples_to["samples"].clone()
        s_from = samples_from["samples"].clone()
        samples_out["samples"] = (s_to + s_from)
        return (samples_out,)


class SaveLatent_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_in": ("LATENT",), }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, latent_in):
        torch.save(latent_in, 'latent.pt')
        return (latent_in, )
    

# a = torch.load("e:/portables/ComfyUI_windows_portable/latent.pt")   
# for idx in range(a['samples'].shape[1]):
#     plt.figure()
#     plt.imshow(a['samples'][0,idx,:,:])
# plt.show()

class LoadLatent_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self,):
        latent_out = torch.load('latent.pt')
        return (latent_out, )
    

class LatentToImages_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_in": ("LATENT",), }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, latent_in):
        s = latent_in['samples']
        s = (s - s.min()) / (s.max() - s.min())
        d1, d2, d3, d4 = s.shape
        images_out = torch.zeros(d2, d3, d4, 3)
        
        for idx in range(s.shape[1]):
            for chan in range(3):
                images_out[idx,:,:,chan] = s[0,idx,:,:]
        return (images_out, )


class LatentMixMasked_As:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "mask": ("MASK",),
                              }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, samples_to, samples_from, mask):
        print(samples_to["samples"].size())
        samples_out = samples_to.copy()
        s_to = samples_to["samples"].clone()
        s_from = samples_from["samples"].clone()
        samples_out["samples"] = s_to * mask + s_from * (1 - mask)
        return (samples_out,)


class ImageMixMasked_As:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image_to": ("IMAGE",),
                              "image_from": ("IMAGE",),
                              "mask": ("MASK",),
                              }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "ASNodes"

    def composite(self, image_to, image_from, mask):

        image_out = image_to.clone()    
        image_out[0,:,:,0] = image_to[0,:,:,0] * mask + image_from[0,:,:,0] * (1 - mask)
        image_out[0,:,:,1] = image_to[0,:,:,1] * mask + image_from[0,:,:,1] * (1 - mask)
        image_out[0,:,:,2] = image_to[0,:,:,2] * mask + image_from[0,:,:,2] * (1 - mask)
        return (image_out,)


class TextToImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        fonts = os.listdir('C:\Windows\Fonts')

        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "font": (fonts, ),
                "size": ("INT", {"default": 20, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, text, font, size, width, height):
        
        PIL_image = Image.new("RGB", (width, height), (0, 0, 0))
        
        draw = ImageDraw.Draw(PIL_image)

        # Set the font and size
        font = ImageFont.truetype(font, size)

        # Get the size of the text
        text_size = draw.textsize(text, font)

        # Calculate the position of the text
        x = (PIL_image.width - text_size[0]) / 2
        y = (PIL_image.height - text_size[1]) / 2

        # Draw the text on the image
        draw.text((x, y), text, font=font, fill=(255,255,255))

        np_image = np.array(PIL_image)

        new_image = torch.zeros(
            (1, height, width, 3),
            dtype=torch.float32,
        )
        new_image[0,:,:,:] = torch.from_numpy(np_image) / 256
        
        return (new_image,)


class BatchIndex_AS:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, batch_index):
        return (batch_index,)
    

class MapRange_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "in_0": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "in_1": ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "out_0": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                "out_1": ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
                
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "mapRange"
    CATEGORY = "ASNodes"

    def mapRange(self, value, in_0, in_1, out_0, out_1):
        if (in_0 == in_1):
            raise ValueError("MapRange_AS: in_0 and in_1 are equal")
        
        run_param = (value - in_0) / (in_1 - in_0)
        result = out_0 + run_param * (out_1 - out_0)
        return (result, round(result))


class Number2Float_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("number", { }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (value, )


class Int2Any_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, }),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (value, )


class Number2Int_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("number", { }),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ASNodes"

    def convert(self, value):
        return (round(value), )


class Eval_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "i1": ("INT", { "default": 0, "step": 1 }),
                "i2": ("INT", { "default": 0, "step": 1 }),
                "f1": ("FLOAT", { "default": 0.0, "step": 0.1 }),
                "f2": ("FLOAT", { "default": 0.0, "step": 0.1 }),
                "s1": ("STRING", { "multiline": True, "default": "" }),
                "s2": ("STRING", { "multiline": True, "default": "" }),
            },
            "required": {
                "int_prc": ("STRING", { "multiline": False, "default": "0" }),
                "float_prc": ("STRING", { "multiline": False, "default": "0" }),
                "str_prc": ("STRING", { "multiline": False, "default": "0" }),
            },

        }

    RETURN_TYPES = ("INT","FLOAT","STRING",)
    FUNCTION = "do_stuff"
    CATEGORY = "ASNodes"

    def do_stuff(self, int_prc, float_prc, str_prc, i1, i2, f1, f2, s1, s2):
        int_out = eval(int_prc)
        float_out = eval(float_prc)
        str_out = eval(str_prc)
        
        return (int_out, float_out, str_out )


class Number_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, }),
            },
        }

    RETURN_TYPES = ("number",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, value):
        return (value, )
    

class Math_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "do": (["add", "subtract", "multiply", "divide", "power"], ),
                "in_0": ("FLOAT", {"default": 0,}),
                "in_1": ("FLOAT", {"default": 0,}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "calculate"
    CATEGORY = "ASNodes"

    def calculate(self, do, in_0, in_1):
        if do=="add":
            result = in_0 + in_1
        if do=="subtract":
            result = in_0 - in_1
        if do=="multiply":
            result = in_0 * in_1
        if do=="divide":
            result = in_0 / in_1
        if do=="power":
            result = in_0 ** in_1

        return (result, round(result) )
    

class Increment_AS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 10, "min": 0, "max": 20, "step":3}),
            },
        }

    RETURN_TYPES = ("number",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, value):
        return (value, )


class CropImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                             "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                             "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                             "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                             "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                         }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "ASNodes"

    def process(self, image, width, height, x, y):

        image_out = image.clone()   
        print(image.shape) 
        image_out = image_out[:, y:y+height, x:x+width]
        return (image_out,)
    

class TextWildcardList_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": "$list", "multiline": True}), 
                             "strings": ("STRING", {"multiline": True}), 
                             "idx": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, text, strings, idx):
        string_list = strings.split(",")
        wildcard = string_list[idx % len(string_list)].strip()
        return (text.replace("$list", wildcard), )
    

class NoiseImage_AS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "idx": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doStuff"
    CATEGORY = "ASNodes"

    def doStuff(self, width, height, idx):
        new_image = torch.rand(
            (1, height, width, 3),
            dtype=torch.float32,
        )
        
        return (new_image,)

    


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MaskToImage_AS": MaskToImage_AS,
    "ImageToMask_AS": ImageToMask_AS,
    "LatentMix_AS": LatentMix_AS,
    "LatentAdd_AS": LatentAdd_AS,
    "SaveLatent_AS": SaveLatent_AS,
    "LoadLatent_AS": LoadLatent_AS,
    "LatentToImages_AS": LatentToImages_AS,
    "LatentMixMasked_As": LatentMixMasked_As,
    "ImageMixMasked_As": ImageMixMasked_As,
    "TextToImage_AS": TextToImage_AS,
    # "BatchIndex_AS": BatchIndex_AS,
    "MapRange_AS": MapRange_AS,
    "Number_AS": Number_AS,
    "Int2Any_AS": Int2Any_AS,
    "Number2Int_AS": Number2Int_AS,
    "Number2Float_AS": Number2Float_AS,
    "Math_AS": Math_AS,
    # "Increment_AS": Increment_AS,
    "CropImage_AS": CropImage_AS,
    "TextWildcardList_AS": TextWildcardList_AS,
    "NoiseImage_AS": NoiseImage_AS,
    "Eval_AS": Eval_AS,
}
