try:
    import disagmoe.ops.fp8_quantizer._C
except ImportError:
    pass

from .fp8_quant import sglang_per_token_group_quant_fp8

