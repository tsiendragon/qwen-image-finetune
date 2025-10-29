from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer as PeftLoraLayer
from tabulate import tabulate


NAME_PATTERNS = ("lora_a", "lora_b", "lora_up", "lora_down", "lora")


def _dtype_tag(dt: torch.dtype, owner: nn.Module | None = None, pname: str = "", p: torch.Tensor | None = None) -> str:
    # 常见浮点
    if dt is torch.float32 or dt is torch.float:
        return "fp32"
    if dt is torch.float16 or dt is torch.half:
        return "fp16"
    if hasattr(torch, "bfloat16") and dt is torch.bfloat16:
        return "bf16"
    # FP8（按版本名匹配）
    for attr, tag in [
        ("float8_e4m3fn", "fp8_e4m3"),
        ("float8_e5m2", "fp8_e5m2"),
        ("float8_e4m3fnuz", "fp8_e4m3"),
        ("float8_e5m2fnuz", "fp8_e5m2"),
    ]:
        t = getattr(torch, attr, None)
        if t is not None and dt is t:
            return tag

    # 整数/量化
    if dt is torch.int8:
        base = "int8"
    elif dt is torch.uint8:
        base = "uint8"
    elif dt is torch.int32:
        base = "int32"
    else:
        return str(dt).replace("torch.", "")

    # ← 仅当是这些存储类型时，尝试判定 4bit
    four = _detect_4bit_tag(owner, pname, p if p is not None else torch.tensor([], dtype=dt))
    return four if four is not None else base


def _detect_4bit_tag(owner: nn.Module | None, pname: str, p: torch.Tensor) -> str | None:
    """
    尝试判定当前 param 是否 4bit，并返回 "fp4"/"nf4"/"int4" 或 None。
    仅用只读启发式（不改模型/名字）：
      - bitsandbytes: owner.weight.quant_state.quant_type in {'fp4','nf4','int4'}
      - GPTQ/AWQ: owner.bits == 4 或 参数名含 'qweight'
      - 类名/模块名包含 '4bit'
    """
    if owner is None:
        return None

    # bitsandbytes: Linear4bit 等模块在 weight 上挂 quant_state
    try:
        w = getattr(owner, "weight", None)
        qs = getattr(w, "quant_state", None)
        if qs is not None:
            qt = None
            for k in ("quant_type", "quant_type_4bit", "qtype"):
                qt = getattr(qs, k, qt)
            if isinstance(qt, str) and qt.lower() in {"fp4", "nf4", "int4"}:
                return qt.lower()
            if getattr(qs, "nbits", None) == 4:
                return "int4"
    except Exception:
        pass

    # GPTQ/AWQ 风格
    try:
        if getattr(owner, "bits", None) == 4:
            return "int4"
    except Exception:
        pass
    if "qweight" in pname.lower():
        return "int4"

    # 类名/模块名提示
    oname = type(owner).__name__.lower()
    omod = getattr(owner, "__module__", "").lower()
    if "4bit" in oname or "4bit" in omod:
        return "int4"

    return None


def _collect_lora_info(model: nn.Module) -> tuple[int, Counter, int]:
    """
    只读统计：
      - LoRA 参数总数（去重）
      - LoRA rank 直方图（r -> count）
      - LoRA “块”数量（按 module 路径 + 适配器名/对名 粗粒度聚类）

    不修改模型，不依赖重命名；仅基于常见属性/命名习惯做推断。
    """
    # 可扩展的名字线索（不会改名，只匹配）

    m = model.module if hasattr(model, "module") else model

    counted_params = set()  # 去重

    def _accumulate_params(params: Iterable[torch.Tensor]) -> int:
        s = 0
        for p in params:
            if isinstance(p, torch.Tensor) and id(p) not in counted_params:
                counted_params.add(id(p))
                s += p.numel()
        return s

    lora_params = 0
    ranks = []
    blocks = set()  # (module_path, tag)

    for mod_path, mod in m.named_modules():
        # ---------- 1) PEFT.LoraLayer ----------
        if PeftLoraLayer is not None and isinstance(mod, PeftLoraLayer):
            # a) 适配器容器 lora_A / lora_B 里读 rank（更准确）
            for cont_name, axis in (("lora_A", 0), ("lora_B", 1)):
                cont = getattr(mod, cont_name, None)
                if isinstance(cont, (nn.ModuleDict, dict)):
                    for adapter, layer in cont.items():
                        w = getattr(layer, "weight", None)
                        if isinstance(w, torch.Tensor):
                            lora_params += _accumulate_params([w])
                            if w.ndim >= 2:
                                r = int(w.shape[axis])
                                if r > 0:
                                    ranks.append(r)
                                    blocks.add((mod_path, f"adapter:{adapter}"))
            # b) 直接挂在该层（如 bias/alpha 等）但名称含 lora
            local = dict(mod.named_parameters(recurse=False))
            lora_params += _accumulate_params(p for n, p in local.items() if any(k in n.lower() for k in NAME_PATTERNS))

        # ---------- 2) 直接属性：lora_A / lora_B 或 lora_down / lora_up ----------
        for a_name, b_name, axis_a, axis_b, tag in (
            ("lora_A", "lora_B", 0, 1, "A/B"),
            ("lora_down", "lora_up", 0, 1, "down/up"),
        ):
            a = getattr(mod, a_name, None)
            b = getattr(mod, b_name, None)
            if a is not None or b is not None:
                ra = rb = None
                for layer, axis in ((a, axis_a), (b, axis_b)):
                    if layer is None:
                        continue
                    w = getattr(layer, "weight", None)
                    if isinstance(w, torch.Tensor):
                        lora_params += _accumulate_params([w])
                        if w.ndim >= 2:
                            r = int(w.shape[axis])
                            if r > 0:
                                if layer is a:
                                    ra = r
                                else:
                                    rb = r
                # 任取一个有效 r（通常两者一致）
                r_eff = ra or rb
                if isinstance(r_eff, int) and r_eff > 0:
                    ranks.append(r_eff)
                blocks.add((mod_path, tag))

        # ---------- 3) 名字兜底：本模块本地参数名含 "lora" ----------
        local_b: list[tuple[str, nn.Parameter]] = list(mod.named_parameters(recurse=False))
        cand_shapes: list[Any] = []
        lora_locals: list[nn.Parameter] = []
        for n, p in local_b:
            if any(k in n.lower() for k in NAME_PATTERNS):
                lora_locals.append(p)
                if p.ndim >= 2:
                    cand_shapes.append(p.shape[-2:])
        if lora_locals:
            lora_params += _accumulate_params(lora_locals)
            if cand_shapes:
                r_guess = min(min(s) for s in cand_shapes)
                if r_guess > 0:
                    ranks.append(int(r_guess))
            blocks.add((mod_path, "name-match"))

    return lora_params, Counter(ranks), len(blocks)


# ---------- formatting helpers ----------
def _human_int(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def _human_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"


def _hist_to_str(d: dict[int, int]) -> str:
    if not d:
        return ""
    return ", ".join(f"{k}×{v}" for k, v in sorted(d.items()))


# ---------- transformer-ish stats ----------
def _is_attention_module(mod: nn.Module) -> bool:
    t = type(mod).__name__
    return (
        "Attention" in t
        or isinstance(mod, nn.MultiheadAttention)
        or any(x in t.lower() for x in ["attn", "selfatt", "mha", "multihead"])
    )


def _is_cross_attention(mod: nn.Module) -> bool:
    if hasattr(mod, "is_cross_attention"):
        try:
            return bool(mod.is_cross_attention)
        except Exception:
            return False
    return "CrossAttention" in type(mod).__name__


def _get_heads_dim(mod: nn.Module) -> tuple[int | None, int | None]:
    if isinstance(mod, nn.MultiheadAttention):
        return int(mod.num_heads), int(mod.embed_dim)
    heads = None
    for h_attr in ("num_heads", "num_attention_heads", "n_heads", "num_head", "heads"):
        if hasattr(mod, h_attr):
            try:
                heads = int(getattr(mod, h_attr))
                break
            except Exception:
                # 尝试从模块名称推断头数
                pass

    # 如果仍然没有找到头数，尝试从模块名称推断
    if heads is None:
        mod_name = type(mod).__name__.lower()
        # 查找形如 "...12heads..." 或 "...12head..." 的模式
        import re

        head_match = re.search(r"(\d+)[\s_-]?heads?", mod_name)
        if head_match:
            try:
                heads = int(head_match.group(1))
            except Exception:
                pass

    dim = None
    for d_attr in ("embed_dim", "hidden_size", "d_model", "inner_dim", "kdim", "dim"):
        if hasattr(mod, d_attr):
            try:
                dim = int(getattr(mod, d_attr))
                break
            except Exception:
                pass
    return heads, dim


def _is_norm(mod: nn.Module) -> bool:
    t = type(mod).__name__
    return t.endswith("Norm") or t in {
        "LayerNorm",
        "RMSNorm",
        "LlamaRMSNorm",
        "T5LayerNorm",
    }


def _is_mlp_block(mod: nn.Module) -> bool:
    t = type(mod).__name__
    if any(k in t for k in ("MLP", "Mlp", "FeedForward", "SwiGLU", "GatedMlp", "GLU")):
        return True
    # heuristic fallback: ≥2 linears, but not an attention module itself
    if not _is_attention_module(mod):
        linears = sum(1 for _ in mod.modules() if isinstance(_, nn.Linear))
        return linears >= 2
    return False


def _collect_transformer_stats(model: nn.Module) -> dict[str, Any]:
    heads, dims = [], []
    attn_total = attn_self = attn_cross = 0
    norm_counts: Counter[str] = Counter()
    mlp_blocks = 0

    for _, mod in model.named_modules():
        if _is_attention_module(mod):
            attn_total += 1
            if _is_cross_attention(mod):
                attn_cross += 1
            else:
                attn_self += 1
            h, d = _get_heads_dim(mod)
            if h is not None:
                heads.append(h)
            if d is not None:
                dims.append(d)
        if _is_norm(mod):
            norm_counts[type(mod).__name__] += 1
        if _is_mlp_block(mod):
            mlp_blocks += 1

    return {
        "attention": {
            "total": attn_total,
            "self": attn_self,
            "cross": attn_cross,
            "num_heads_hist": dict(Counter(heads)),
            "hidden_dim_hist": dict(Counter(dims)),
        },
        "norms": dict(norm_counts),
        "mlp_blocks": mlp_blocks,
    }


# ---------------- gather (新增 dtype 统计) ----------------
def gather_model_stats(model: nn.Module) -> dict[str, Any]:
    m = model.module if hasattr(model, "module") else model

    total_params = trainable_params = total_bytes = trainable_bytes = 0

    # dtype 聚合容器
    dtype_stats: dict[str, dict[str, float]] = {}  # {tag: {total, trainable, total_bytes, trainable_bytes}}
    mods = dict(model.named_modules())
    for pname, p in model.named_parameters():
        n = p.numel()
        try:
            es = p.element_size()  # bytes per element
        except Exception:
            es = 0
        b = n * es

        total_params += n
        total_bytes += b
        if p.requires_grad:
            trainable_params += n
            trainable_bytes += b

        # 聚合到 dtype
        # tag = _dtype_tag(p.dtype)
        owner = mods.get(pname.rsplit(".", 1)[0]) if "." in pname else None
        tag = _dtype_tag(p.dtype, owner=owner, pname=pname, p=p)

        ds = dtype_stats.setdefault(tag, {"total": 0, "trainable": 0, "total_bytes": 0, "trainable_bytes": 0})
        ds["total"] += n
        ds["total_bytes"] += b
        if p.requires_grad:
            ds["trainable"] += n
            ds["trainable_bytes"] += b

    # 其余统计（保持你原实现）
    total_modules = sum(1 for _ in m.modules()) - 1
    leaf_modules = sum(1 for mod in m.modules() if len(list(mod.children())) == 0)
    param_modules = sum(1 for mod in m.modules() if any(p.numel() > 0 for p in mod.parameters(recurse=False)))

    lora_params, lora_rank_counts, lora_blocks = _collect_lora_info(m)
    tf = _collect_transformer_stats(m)

    pct_trainable = (trainable_params / total_params * 100.0) if total_params else 0.0
    pct_lora = (lora_params / total_params * 100.0) if total_params else 0.0

    # 计算 dtype 百分比，转为可序列化
    dtype_list: list[dict[str, Any]] = []
    for tag, ds in dtype_stats.items():
        pct = (ds["total"] / total_params * 100.0) if total_params else 0.0
        dtype_list.append(
            {
                "dtype": tag,
                "total": int(ds["total"]),
                "trainable": int(ds["trainable"]),
                "pct_of_total_params": pct,
                "total_bytes": int(ds["total_bytes"]),
                "trainable_bytes": int(ds["trainable_bytes"]),
            }
        )
    # 按占比或数量降序
    dtype_list.sort(key=lambda x: (x["total"], x["total_bytes"]), reverse=True)

    return {
        "modules": {"total": total_modules, "leaf": leaf_modules, "parameterized": param_modules},
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "pct_trainable": pct_trainable,
            "total_bytes": total_bytes,
            "trainable_bytes": trainable_bytes,
        },
        "transformer": tf,
        "lora": {
            "blocks": lora_blocks,
            "num_params": lora_params,
            "pct_of_total_params": pct_lora,
            "ranks": dict(lora_rank_counts),
        },
        "dtypes": dtype_list,  # ← 新增
    }


# --------------- pretty table (新增 DTypes 区块) ---------------
def print_model_summary_table(model: nn.Module, name: str = "model") -> dict[str, Any]:
    stats = gather_model_stats(model)
    rows: list[list[str]] = []

    # Parameters
    p = stats["parameters"]
    rows += [
        ["Parameters", "Total", _human_int(p["total"])],
        ["Parameters", "Trainable", _human_int(p["trainable"])],
        ["Parameters", "Trainable %", f"{p['pct_trainable']:.2f}%"],
        ["Parameters", "Memory (total)", _human_bytes(p["total_bytes"])],
        ["Parameters", "Memory (trainable)", _human_bytes(p["trainable_bytes"])],
    ]

    # Modules
    m = stats["modules"]
    rows += [
        ["Modules", "Total", f"{m['total']}"],
        ["Modules", "Leaf", f"{m['leaf']}"],
        ["Modules", "Parameterized", f"{m['parameterized']}"],
    ]

    # Transformer-ish
    t = stats["transformer"]["attention"]
    rows += [
        ["Transformer", "Attention blocks (total/self/cross)", f"{t['total']}/{t['self']}/{t['cross']}"],
        ["Transformer", "Heads (hist)", _hist_to_str(t["num_heads_hist"]) or "未检测到"],
        ["Transformer", "Attn hidden dim (hist)", _hist_to_str(t["hidden_dim_hist"]) or "未检测到"],
    ]
    norms = stats["transformer"]["norms"]
    rows += [["Transformer", "Norms", ", ".join(f"{k}×{v}" for k, v in norms.items()) or ""]]
    rows += [["Transformer", "FeedForward blocks", f"{stats['transformer']['mlp_blocks']}"]]

    # LoRA
    l_state = stats["lora"]
    rank_hist = ", ".join(f"r={r}×{c}" for r, c in sorted(l_state["ranks"].items()))
    rank_vals = sorted(int(r) for r in l_state["ranks"].keys()) if l_state["ranks"] else []
    rows += [
        ["LoRA", "Blocks", f"{l_state['blocks']}"],
        ["LoRA", "Parameters", _human_int(l_state["num_params"])],
        ["LoRA", "Params % of total", f"{l_state['pct_of_total_params']:.2f}%"],
        ["LoRA", "Ranks (hist)", rank_hist or ""],
        ["LoRA", "Rank min/max", f"{rank_vals[0]} / {rank_vals[-1]}" if rank_vals else ""],
    ]

    # NEW: DTypes breakdown
    if stats["dtypes"]:
        rows += [["DTypes", "dtype", "params(total/train) | % | mem(total)"]]
        for ds in stats["dtypes"]:
            val = (
                f"{_human_int(ds['total'])}/{_human_int(ds['trainable'])} | "
                f"{ds['pct_of_total_params']:.2f}% | {_human_bytes(ds['total_bytes'])}"
            )
            rows += [["DTypes", ds["dtype"], val]]

    # None -> ""
    def _clean(x: Any) -> str:
        return "" if x is None else str(x)

    rows = [[_clean(a), _clean(b), _clean(c)] for a, b, c in rows]

    title = f"Model Summary: {name}"

    # rich / tabulate / fallback
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        table.add_column("Section", style="bold")
        table.add_column("Metric")
        table.add_column("Value")
        for r in rows:
            table.add_row(*r)
        Console().print(table)
    except Exception:
        print(f"\n{title}")
        print(tabulate(rows, headers=["Section", "Metric", "Value"], tablefmt="github"))
    return {
        "rows": rows,
        "columns": ["Section", "Metric", "Value"],
        "stats": stats,
    }


if __name__ == "__main__":
    from qflux.models.flux_kontext_loader import load_flux_kontext_transformer

    dit = load_flux_kontext_transformer(repo="eramth/flux-kontext-4bit-fp4")
    print_model_summary_table(dit)
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    dit.add_adapter(lora_config, adapter_name="lora")
    dit.set_adapter("lora")
    a = print_model_summary_table(dit)
    print(a)
