from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
from ..training.checkpoints import apply_delta_checkpoint, resolve_optional_checkpoint
import torch
from torchao.quantization import int8_weight_only, quantize_
from typing import Dict, Any, Optional, Union
import os
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import json
from contextlib import nullcontext
import gc
import numpy as np


def _resolve_paths(pretrained_path: str, version: str):

    heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
    heartcodec_path = os.path.join(pretrained_path, "HeartCodec-oss")
    tokenizer_path = os.path.join(pretrained_path, "tokenizer.json")
    gen_config_path = os.path.join(pretrained_path, "gen_config.json")

    if not os.path.exists(heartmula_path):
        raise FileNotFoundError(
            f"Expected to find checkpoint for HeartMuLa at {heartmula_path} but not found. Please check your folder {pretrained_path}."
        )
    if not os.path.exists(heartcodec_path):
        raise FileNotFoundError(
            f"Expected to find checkpoint for HeartCodec at {heartcodec_path} but not found. Please check your folder {pretrained_path}."
        )
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"Expected to find tokenizer.json for HeartMuLa at {tokenizer_path} but not found. Please check your folder {pretrained_path}."
        )
    if not os.path.isfile(gen_config_path):
        raise FileNotFoundError(
            f"Expected to find gen_config.json for HeartMuLa at {gen_config_path} but not found. Please check your folder {pretrained_path}."
        )

    return heartmula_path, heartcodec_path, tokenizer_path, gen_config_path


def _resolve_devices(
    device: Union[torch.device, Dict[str, torch.device]], lazy_load: bool
):
    if isinstance(device, torch.device):
        print(f"All model components will be loaded to device: {device}.")
        mula_device = device
        codec_device = device
    elif isinstance(device, dict):
        print("Model components will be loaded to devices as specified:")
        for k, v in device.items():
            print(f"  {k}: {v}")
        mula_device = device["mula"]
        codec_device = device["codec"]
    else:
        raise ValueError(
            "device must be either torch.device or Dict[str, torch.device]"
        )

    single_device = mula_device == codec_device
    if not single_device:
        print("HeartMuLa and HeartCodec will be loaded to different devices.")

    return mula_device, codec_device, lazy_load


def _device_memory_allocated(device: torch.device) -> Optional[int]:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "current_allocated_memory"):
            try:
                return mps.current_allocated_memory()
            except RuntimeError:
                return None
    return None


def _empty_device_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            try:
                mps.empty_cache()
            except RuntimeError:
                pass


def _log_device_memory(prefix: str, device: torch.device):
    allocated = _device_memory_allocated(device)
    if allocated is None:
        print(f"{prefix} on {device}: unavailable")
        return
    print(f"{prefix} on {device}: {allocated / 1024**3:.2f} GB")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        return nullcontext()
    if device.type == "cpu" and dtype != torch.bfloat16:
        return nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)


class HeartMuLaGenPipeline:
    def __init__(
        self,
        heartmula_path: str,
        heartcodec_path: str,
        heartmula_device: torch.device,
        heartcodec_device: torch.device,
        heartmula_dtype: torch.dtype,
        heartcodec_dtype: torch.dtype,
        lazy_load: bool,
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        heartmula_delta_path: Optional[str] = None,
    ):

        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config

        # Remain fixed here for simplicity.
        self._parallel_number = 8 + 1
        self._muq_dim = 512

        self.mula_dtype = heartmula_dtype
        self.mula_path = heartmula_path
        self.mula_delta_path = resolve_optional_checkpoint(heartmula_delta_path)
        self.mula_device = heartmula_device
        self.codec_dtype = heartcodec_dtype
        self.codec_path = heartcodec_path
        self.codec_device = heartcodec_device

        self._mula: Optional[HeartMuLa] = None
        self._codec: Optional[HeartCodec] = None
        self._mula_compiled = False
        if not lazy_load:
            print(
                f"You have set lazy_load = False. Loading HeartMuLa and HeartCodec onto device..."
            )
            self._mula = apply_delta_checkpoint(
                HeartMuLa.from_pretrained(
                    self.mula_path,
                    device_map=self.mula_device,
                    dtype=self.mula_dtype,
                ),
                self.mula_delta_path,
            )
            self._codec = HeartCodec.from_pretrained(
                self.codec_path,
                device_map=self.codec_device,
                dtype=self.codec_dtype,
            )
        self.lazy_load = lazy_load

    @property
    def mula(self) -> HeartMuLa:
        if isinstance(self._mula, HeartMuLa):
            return self._mula
        self._mula = apply_delta_checkpoint(
            HeartMuLa.from_pretrained(
                self.mula_path,
                device_map=self.mula_device,
                dtype=self.mula_dtype,
            ),
            self.mula_delta_path,
        )
        return self._mula

    @property
    def codec(self) -> HeartCodec:
        if isinstance(self._codec, HeartCodec):
            return self._codec
        self._codec = HeartCodec.from_pretrained(
            self.codec_path,
            device_map=self.codec_device,
            dtype=self.codec_dtype,
        )
        return self._codec

    def _unload(
        self,
        *,
        force: bool = False,
        unload_mula: bool = True,
        unload_codec: bool = True,
    ):
        if not self.lazy_load and not force:
            return
        unload_reason = "lazy_load=True" if self.lazy_load and not force else "manual release"
        if unload_mula and isinstance(self._mula, HeartMuLa):
            print(f"Unloading HeartMuLa from device ({unload_reason}).")
            _log_device_memory("Memory before unloading", self.mula_device)
            del self._mula
            gc.collect()
            _empty_device_cache(self.mula_device)
            _log_device_memory("Memory after unloading", self.mula_device)
            self._mula = None
            self._mula_compiled = False
        if unload_codec and isinstance(self._codec, HeartCodec):
            print(f"Unloading HeartCodec from device ({unload_reason}).")
            _log_device_memory("Memory before unloading", self.codec_device)
            del self._codec
            gc.collect()
            _empty_device_cache(self.codec_device)
            _log_device_memory("Memory after unloading", self.codec_device)
            self._codec = None
        return

    def _current_cache_batch_size(self) -> Optional[int]:
        if not isinstance(self._mula, HeartMuLa):
            return None
        try:
            first_layer = self._mula.backbone.layers[0]
            kv_cache = first_layer.attn.kv_cache
        except (AttributeError, IndexError):
            return None
        if kv_cache is None:
            return None
        return getattr(kv_cache, "batch_size", None)

    def _reload_mula(self):
        if isinstance(self._mula, HeartMuLa):
            del self._mula
            gc.collect()
            _empty_device_cache(self.mula_device)
        self._mula_compiled = False
        self._mula = apply_delta_checkpoint(
            HeartMuLa.from_pretrained(
                self.mula_path,
                device_map=self.mula_device,
                dtype=self.mula_dtype,
            ),
            self.mula_delta_path,
        )

    def _prepare_mula_caches(self, batch_size: int):
        cache_batch_size = self._current_cache_batch_size()
        if cache_batch_size is None:
            self.mula.setup_caches(batch_size)
            self._try_compile_mula()
            return
        if cache_batch_size == batch_size:
            self.mula.reset_caches()
            return
        # Rebuild the model only when the requested cache batch size changes.
        self._reload_mula()
        self.mula.setup_caches(batch_size)
        self._try_compile_mula()

    def _try_compile_mula(self):
        """Apply INT8 quantization and torch.compile for faster autoregressive generation."""
        if self._mula_compiled or not isinstance(self._mula, HeartMuLa):
            return
        # INT8 weight-only quantization via torchao (reduces memory ~2x, speeds up matmuls)
        try:
            quantize_(self._mula.backbone, int8_weight_only())
            quantize_(self._mula.decoder, int8_weight_only())
            print("⚡ INT8 weight-only quantization applied to HeartMuLa backbone + decoder.")
        except Exception as e:
            print(f"INT8 quantization skipped ({e}).")
        # torch.compile for fused kernels
        try:
            self._mula.backbone = torch.compile(self._mula.backbone)
            self._mula.decoder = torch.compile(self._mula.decoder)
            print("⚡ torch.compile applied to HeartMuLa backbone + decoder.")
        except Exception as e:
            print(f"torch.compile skipped ({e}), using eager mode.")
        self._mula_compiled = True

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {"cfg_scale": kwargs.get("cfg_scale", 1.5)}
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "show_progress": kwargs.get("show_progress", True),
        }
        postprocess_kwargs = {
            "save_path": kwargs.get("save_path", "output.mp3"),
            "codec_chunk_duration_s": kwargs.get("codec_chunk_duration_s", None),
            "codec_num_steps": kwargs.get("codec_num_steps", 10),
            "codec_guidance_scale": kwargs.get("codec_guidance_scale", 1.25),
            "save_audio_tokens": kwargs.get("save_audio_tokens", False),
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):

        # process tags
        tags = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        # encapsulate with special <tag> and </tag> tokens
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        # process reference audio
        ref_audio = inputs.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = torch.zeros([self._muq_dim], dtype=self.mula_dtype)
        muq_idx = len(tags_ids)

        # process lyrics
        lyrics = inputs["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp:
                lyrics = fp.read()
        assert isinstance(
            lyrics, str
        ), f"lyrics must be a string, but got {type(lyrics)}"
        lyrics = lyrics.lower()

        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        # cat them together. tags, ref_audio, lyrics
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
        }

    def prepare_inputs(self, inputs: Dict[str, Any], cfg_scale: float = 1.5):
        return self.preprocess(inputs, cfg_scale=cfg_scale)

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        show_progress: bool,
    ):
        prompt_tokens = model_inputs["tokens"].to(self.mula_device)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(self.mula_device)
        continuous_segment = model_inputs["muq_embed"].to(self.mula_device)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(self.mula_device)
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self._prepare_mula_caches(bs_size)
        with torch.inference_mode():
            with _autocast_context(self.mula_device, self.mula_dtype):
                curr_token = self.mula.generate_frame(
                    tokens=prompt_tokens,
                    tokens_mask=prompt_tokens_mask,
                    input_pos=prompt_pos,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=continuous_segment,
                    starts=starts,
                )
        frames.append(curr_token[0:1,])

        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], self._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * self.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        max_audio_frames = max_audio_length_ms // 80

        with torch.inference_mode():
            for i in tqdm(range(max_audio_frames), disable=not show_progress):
                curr_token, curr_token_mask = _pad_audio_token(curr_token)
                with _autocast_context(self.mula_device, self.mula_dtype):
                    curr_token = self.mula.generate_frame(
                        tokens=curr_token,
                        tokens_mask=curr_token_mask,
                        input_pos=prompt_pos[..., -1:] + i + 1,
                        temperature=temperature,
                        topk=topk,
                        cfg_scale=cfg_scale,
                        continuous_segments=None,
                        starts=None,
                    )
                if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                    break
                frames.append(curr_token[0:1,])
        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        self._unload()
        return {"frames": frames}

    def postprocess(
        self,
        model_outputs: Dict[str, Any],
        save_path: str,
        codec_chunk_duration_s: Optional[float],
        codec_num_steps: int,
        codec_guidance_scale: float,
        save_audio_tokens: bool,
    ):
        raw_frames = model_outputs["frames"]
        audio_tokens_path = None
        if save_audio_tokens:
            audio_tokens_path = Path(save_path).with_suffix(".audio_tokens.npy")
            audio_tokens_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                audio_tokens_path,
                raw_frames.detach().to(torch.int16).cpu().numpy(),
            )

        frames = raw_frames.to(self.codec_device)
        with torch.inference_mode():
            wav = self.codec.detokenize(
                frames,
                duration=codec_chunk_duration_s,
                num_steps=codec_num_steps,
                guidance_scale=codec_guidance_scale,
            )
        self._unload()
        sf.write(
            save_path,
            wav.to(torch.float32).transpose(0, 1).cpu().numpy(),
            48000,
        )
        return {
            "save_path": save_path,
            "audio_tokens_path": str(audio_tokens_path) if audio_tokens_path else "",
        }

    def generate_from_prepared(self, model_inputs: Dict[str, Any], **kwargs):
        _, forward_kwargs, postprocess_kwargs = self._sanitize_parameters(**kwargs)
        model_outputs = self._forward(model_inputs, **forward_kwargs)
        return self.postprocess(model_outputs, **postprocess_kwargs)

    def generate_frames_from_prepared(self, model_inputs: Dict[str, Any], **kwargs):
        _, forward_kwargs, _ = self._sanitize_parameters(**kwargs)
        return self._forward(model_inputs, **forward_kwargs)

    def decode_frames(self, model_outputs: Dict[str, Any], **kwargs):
        _, _, postprocess_kwargs = self._sanitize_parameters(**kwargs)
        return self.postprocess(model_outputs, **postprocess_kwargs)

    def release_generation_model(self):
        self._unload(force=True, unload_mula=True, unload_codec=False)

    def __call__(self, inputs: Dict[str, Any], **kwargs):
        preprocess_kwargs, _, _ = self._sanitize_parameters(**kwargs)
        model_inputs = self.prepare_inputs(inputs, **preprocess_kwargs)
        return self.generate_from_prepared(model_inputs, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: Union[torch.device, Dict[str, torch.device]],
        dtype: Union[torch.dtype, Dict[str, torch.dtype]],
        version: str,
        lazy_load: bool = False,
        heartmula_delta_path: Optional[str] = None,
    ):

        mula_path, codec_path, tokenizer_path, gen_config_path = _resolve_paths(
            pretrained_path, version
        )
        mula_device, codec_device, lazy_load = _resolve_devices(device, lazy_load)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        mula_dtype = dtype["mula"] if isinstance(dtype, dict) else dtype
        codec_dtype = dtype["codec"] if isinstance(dtype, dict) else dtype

        return cls(
            heartmula_path=mula_path,
            heartcodec_path=codec_path,
            heartmula_device=mula_device,
            heartcodec_device=codec_device,
            lazy_load=lazy_load,
            muq_mulan=None,
            text_tokenizer=tokenizer,
            config=gen_config,
            heartmula_dtype=mula_dtype,
            heartcodec_dtype=codec_dtype,
            heartmula_delta_path=heartmula_delta_path,
        )
