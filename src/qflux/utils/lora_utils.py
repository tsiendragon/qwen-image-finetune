import re
import time
from collections import deque
from collections.abc import Callable

import safetensors.torch
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def classify_lora_weight(lora_weight):
    sd = safetensors.torch.load_file(lora_weight)
    keys = list(sd.keys())
    peft = any(re.search(r"\.lora_[AB](\.|$)", k) for k in keys)
    diff = any(".lora.down.weight" in k or ".lora.up.weight" in k for k in keys)
    proc = any(".processor" in k for k in keys)
    if peft and not diff:
        return "PEFT"
    if diff:
        return "DIFFUSERS(attn-processor)" if proc else "DIFFUSERS"
    return "UNKNOWN"


def get_lora_layers(model):
    """Traverse the model to find all LoRA-related modules"""
    lora_layers = {}

    def fn_recursive_find_lora_layer(name: str, module: torch.nn.Module, processors):
        if "lora" in name:
            lora_layers[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_find_lora_layer(f"{name}.{sub_name}", child, lora_layers)
        return lora_layers

    for name, module in model.named_children():
        fn_recursive_find_lora_layer(name, module, lora_layers)
    return lora_layers


def collect_lora_linears(root: nn.Module) -> list[nn.Linear]:
    loras = []
    for m in root.modules():
        if isinstance(m, nn.Linear):
            # 常见 PEFT LoRA 标记
            if (
                hasattr(m, "lora_A")
                or hasattr(m, "lora_B")
                or hasattr(m, "lora_embedding_A")
                or hasattr(m, "lora_embedding_B")
                or hasattr(m, "lora_edit")
                or hasattr(m, "lora_down")
                or hasattr(m, "lora_up")
            ):  # 你自定义的标记
                loras.append(m)
                continue
            # 兜底：该 Linear 下有名字包含 "lora" 的子参数，或除了 weight/bias 外仍有可训练参数
            names_params = dict[str, Parameter](m.named_parameters(recurse=False))
            if any(("lora" in n) for n in names_params.keys()):
                loras.append(m)
                continue
            other_trainables = [p for n, p in names_params.items() if n not in ("weight", "bias") and p.requires_grad]
            if other_trainables:
                loras.append(m)
    return loras


class FpsLogger:
    def __init__(
        self,
        warmup_steps: int = 5,
        window_size: int = 50,  # 滑动窗口步数，0 表示不用
        ema_alpha: float = 0.2,  # 指数滑动平均，None 关闭
        cuda_synchronize: Callable | None = None,  # 传入 torch.cuda.synchronize
    ):
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.cuda_sync = cuda_synchronize

        self._started = False
        self._paused = False
        self._pause_t = 0.0
        self._pause_accum = 0.0

        self.global_samples = 0
        self.global_tokens = 0
        self.steps = 0

        self._t0 = -1.0
        self._last_step_t: float = -1.0
        self._fps_ema: float = -1.0
        self._step_window: deque[tuple[int, float]] = deque(maxlen=window_size)
        self._last_fps = 0.0

    @staticmethod
    def _now() -> float:
        # 高精度&单调时钟
        return time.perf_counter()

    def start(self) -> None:
        if self.cuda_sync:
            self.cuda_sync()
        t = self._now()
        self._t0 = t
        self._last_step_t = t
        self._started = True
        self._paused = False
        self._pause_accum = 0.0

    def pause(self) -> None:
        if not self._started or self._paused:
            return
        if self.cuda_sync:
            self.cuda_sync()
        self._pause_t = self._now()
        self._paused = True

    def resume(self) -> None:
        if not self._started or not self._paused:
            return
        if self.cuda_sync:
            self.cuda_sync()
        t = self._now()
        self._pause_accum += t - self._pause_t
        self._last_step_t = t
        self._paused = False

    def update(self, batch_size: int, num_tokens: int | None = None) -> float:
        """
        batch_samples: 本步真实样本数（本进程/本机）
        batch_tokens:  可选，本步真实 token 数
        """
        if not self._started:
            self.start()

        if self.cuda_sync:
            self.cuda_sync()
        t = self._now()

        self.steps += 1
        self.global_samples += int(batch_size)
        if num_tokens is not None:
            self.global_tokens += int(num_tokens)

        # 逐步 FPS（含数据加载等端到端）
        dt = max(t - self._last_step_t, 1e-9)
        inst_fps = batch_size / dt

        # 滑动窗口
        if self.window_size > 0:
            self._step_window.append((batch_size, dt))
            tot_s, tot_t = 0, 0.0
            for s, d in self._step_window:
                tot_s += s
                tot_t += d
            window_fps = tot_s / max(tot_t, 1e-9)
        else:
            window_fps = inst_fps

        # EMA
        if self.ema_alpha is not None:
            self._fps_ema = (
                window_fps
                if self._fps_ema is -1
                else (self.ema_alpha * window_fps + (1 - self.ema_alpha) * self._fps_ema)
            )
            smoothed = self._fps_ema
        else:
            smoothed = window_fps

        self._last_step_t = t
        self._last_fps = smoothed
        return smoothed

    def total_fps(self) -> float:
        """
        端到端 FPS（丢弃 warmup_steps；排除了 pause 的时间）
        """
        if self.cuda_sync:
            self.cuda_sync()
        if not self._started or self._t0 == -1:
            return 0.0
        t = self._now()
        elapsed = t - self._t0 - self._pause_accum
        # 简单地按步均分 warmup 样本
        eff_samples = float(self.global_samples)
        if self.steps > self.warmup_steps:
            eff_samples *= (self.steps - self.warmup_steps) / self.steps
        return eff_samples / max(elapsed, 1e-9)

    def tokens_per_sec(self) -> float:
        if self.global_tokens == 0:
            return 0.0
        if self.cuda_sync:
            self.cuda_sync()
        t = self._now()
        elapsed = t - self._t0 - self._pause_accum
        return self.global_tokens / max(elapsed, 1e-9)

    def last_fps(self):
        return float(self._last_fps)


def get_lora_state_dict_oom_safe(model, adapter_name: str = "default"):
    """
    OOM-safe：仅读取 PEFT 风格的 LoRA 子模块 (lora_A/lora_B/...),
    导出到 CPU，并去掉 ".{adapter_name}" 后缀，便于后续 convert_state_dict_to_diffusers。
    """
    import torch
    import torch.nn as nn

    model = getattr(model, "_orig_mod", model)

    # 安全检查：LoRA 不应被 FSDP 切分
    for n, p in model.named_parameters():
        if ("lora_" in n or "lora_recover" in n) and getattr(p, "_is_sharded", False):
            raise RuntimeError(f"LoRA param is sharded by FSDP: {n}")

    sd = {}

    def _put(k: str, t: torch.Tensor):
        if adapter_name:
            k = k.replace(f".{adapter_name}", "")
        # 清理 FSDP/包装前缀，避免保存时出现不一致
        k = (
            k.replace("._fsdp_wrapped_module.", ".")
            .replace("._checkpoint_wrapped_module.", ".")
            .replace("._offload_wrapped_module.", ".")
        )
        if k.startswith("module."):
            k = k[len("module.") :]
        sd[k] = t.detach().to("cpu", non_blocking=True)

    with torch.inference_mode():
        # 只抓 PEFT 的子模块：lora_A / lora_B / lora_embedding_A / lora_embedding_B / (可选) adapter_name
        for mod_name, m in model.named_modules():
            for attr in (
                "lora_A",
                "lora_B",
                "lora_embedding_A",
                "lora_embedding_B",
                adapter_name,  # 一些实现把 adapter_name 当子模块名
            ):
                if hasattr(m, attr):
                    sub = getattr(m, attr)
                    if isinstance(sub, nn.Module):
                        for pn, p in sub.named_parameters(recurse=False):
                            _put(f"{mod_name}.{attr}.{pn}", p)

        # 兜底：极少实现把 lora_* 直接挂在本体上
        for n, p in model.named_parameters():
            if ("lora_A." in n or "lora_B." in n or "lora_embedding_" in n) and n not in sd:
                _put(n, p)

    torch.cuda.empty_cache()
    return sd
