from .janus_pro_caption import JanusProModelLoader, JanusProDescribeImage, JanusProCaptionImageUnderDirectory
from .florence2_caption import Florence2ModelLoader, Florence2DescribeImage, Florence2CaptionImageUnderDirectory
from .utils import add_suffix, add_emoji

NODE_CLASS_MAPPINGS = {
    add_suffix("JanusProModelLoader"): JanusProModelLoader,
    add_suffix("JanusProDescribeImage"): JanusProDescribeImage,
    add_suffix("JanusProCaptionImageUnderDirectory"): JanusProCaptionImageUnderDirectory,
    add_suffix("Florence2ModelLoader"): Florence2ModelLoader,
    add_suffix("Florence2DescribeImage"): Florence2DescribeImage,
    add_suffix("Florence2CaptionImageUnderDirectory"): Florence2CaptionImageUnderDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    add_suffix("JanusProModelLoader"): add_emoji("Janus Pro Model Loader"),
    add_suffix("JanusProDescribeImage"): add_emoji("Janus Pro Describe Image"),
    add_suffix("JanusProCaptionImageUnderDirectory"): add_emoji("Janus Pro Caption Images Under Directory"),
    add_suffix("Florence2ModelLoader"): add_emoji("Florence2 Model Loader"),
    add_suffix("Florence2DescribeImage"): add_emoji("Florence2 Describe Image"),
    add_suffix("Florence2CaptionImageUnderDirectory"): add_emoji("Florence2 Caption Images Under Directory"),
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
