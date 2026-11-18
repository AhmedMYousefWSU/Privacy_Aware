"""Frozen LLaVA-1.5 wrapper that can (a) expose visual tokens, (b) score them, (c) answer with a subset.

Pruning is done by building `inputs_embeds` ourselves:
    [ embeds(text before <image>) | projector(z_i) for i in keep | embeds(text after <image>) ]
and calling `generate(inputs_embeds=...)` with no pixel_values, so the LLM never sees dropped tokens.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from . import geometry as G

PROMPT = "USER: <image>\n{question} ASSISTANT:"


class PrunableLlava:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda", dtype=torch.float16,
                 attn_implementation="eager"):
        # eager attention is required for output_attentions (attention-based relevance R).
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation=attn_implementation).to(device).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.feature_layer = cfg.vision_feature_layer  # -2 for LLaVA-1.5

    # ------------------------------------------------------------------ vision side
    def pixel_values(self, img336):
        """img336: output of geometry.llava_square_image (already padded + 336x336)."""
        pv = self.processor.image_processor(img336, return_tensors="pt")["pixel_values"]
        return pv.to(self.device, self.dtype)

    @torch.no_grad()
    def encode(self, pixel_values):
        """Vision tokens z_i before the projector: (B, 576, 1024). Input to the privacy head."""
        out = self.model.vision_tower(pixel_values, output_hidden_states=True)
        return out.hidden_states[self.feature_layer][:, 1:]

    @torch.no_grad()
    def project(self, z):
        """z (…, 576, 1024) -> LLM-space visual tokens (…, 576, 4096)."""
        return self.model.multi_modal_projector(z.to(self.dtype))

    # ------------------------------------------------------------------ prompt assembly
    def _prompt_ids(self, question):
        ids = self.tok(PROMPT.format(question=question), return_tensors="pt").input_ids[0]
        pos = (ids == self.image_token_id).nonzero()
        assert len(pos) == 1, "expected exactly one <image> placeholder"
        p = int(pos[0])
        return ids[:p], ids[p + 1:]

    @torch.no_grad()
    def build_embeds(self, v_tokens, question, keep_idx=None, suffix_ids=None):
        """v_tokens: (576, 4096) projected tokens. Returns (embeds (1,L,D), spans dict).

        Kept tokens stay in their original raster order. `suffix_ids` (answer tokens) are appended
        for teacher-forced scoring.
        """
        pre, post = self._prompt_ids(question)
        if suffix_ids is not None:
            post = torch.cat([post, suffix_ids.cpu()])
        emb = self.model.get_input_embeddings()
        if keep_idx is not None:
            keep_idx = torch.as_tensor(sorted(int(i) for i in keep_idx), device=v_tokens.device)
            v_tokens = v_tokens.index_select(0, keep_idx)
        e = torch.cat([emb(pre.to(self.device)), v_tokens.to(self.dtype), emb(post.to(self.device))], 0)
        n_pre, n_img = len(pre), v_tokens.shape[0]
        spans = {"img": (n_pre, n_pre + n_img), "text": (n_pre + n_img, e.shape[0])}
        return e.unsqueeze(0), spans

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, v_tokens, question, keep_sets, max_new_tokens=20, max_batch_tokens=8192):
        """Answer `question` once per keep set. Equal-length keep sets are batched together, in chunks of at
        most `max_batch_tokens` visual tokens (eager attention memory grows with batch x length^2)."""
        answers = [None] * len(keep_sets)
        groups = {}
        for j, ks in enumerate(keep_sets):
            groups.setdefault(len(ks), []).append(j)
        chunks = []
        for n, js in groups.items():
            bs = max(1, max_batch_tokens // max(n, 1))
            chunks += [js[b:b + bs] for b in range(0, len(js), bs)]
        for js in chunks:
            embs = torch.cat([self.build_embeds(v_tokens, question, keep_sets[j])[0] for j in js], 0)
            out = self.model.generate(inputs_embeds=embs, attention_mask=torch.ones(embs.shape[:2], device=self.device, dtype=torch.long),
                                      max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            for j, seq in zip(js, out):
                answers[j] = self.tok.decode(seq, skip_special_tokens=True).strip()
        return answers

    # ------------------------------------------------------------------ relevance: attention
    @contextmanager
    def _truncated_llm(self, n_layers):
        """Run only the first n_layers decoder layers (FastV-style cheap scoring pass)."""
        lm = self.model.model.language_model
        full = lm.layers
        lm.layers = full[:n_layers]
        try:
            yield lm
        finally:
            lm.layers = full

    @torch.no_grad()
    def attention_relevance(self, v_tokens, question, layers=range(2, 16), query="text"):
        """Raw R~_i = mean over layers l, heads h, query tokens t of A^{l,h}_{t,i}. Returns (576,) float32.

        query="text": all text tokens after the image (question + 'ASSISTANT:'),
        query="last": only the last prompt token (the one that emits the first answer token).
        """
        layers = list(layers)
        e, sp = self.build_embeds(v_tokens, question)
        with self._truncated_llm(max(layers) + 1) as lm:
            out = lm(inputs_embeds=e, output_attentions=True, use_cache=False)
        (i0, i1), (t0, t1) = sp["img"], sp["text"]
        rows = slice(t0, t1) if query == "text" else slice(t1 - 1, t1)
        acc = torch.zeros(i1 - i0, device=self.device, dtype=torch.float32)
        for l in layers:
            a = out.attentions[l][0]  # (H, L, L)
            acc += a[:, rows, i0:i1].float().mean(dim=(0, 1))
        return (acc / len(layers)).cpu()

    # ------------------------------------------------------------------ relevance: leave-block-out (oracle)
    @torch.no_grad()
    def answer_logprob(self, v_tokens, question, answer_ids, keep_sets, batch_size=16):
        """Teacher-forced log p(answer | kept tokens, question) for each keep set (equal lengths)."""
        answer_ids = answer_ids.to(self.device)
        n_ans = len(answer_ids)
        res = []
        for b in range(0, len(keep_sets), batch_size):
            embs = torch.cat([self.build_embeds(v_tokens, question, ks, suffix_ids=answer_ids)[0]
                              for ks in keep_sets[b:b + batch_size]], 0)
            logits = self.model.language_model(inputs_embeds=embs, use_cache=False).last_hidden_state
            logits = self.model.lm_head(logits[:, -n_ans - 1:-1]).float()
            lp = F.log_softmax(logits, -1).gather(-1, answer_ids.view(1, -1, 1).expand(len(embs), -1, 1))
            res.append(lp.squeeze(-1).sum(-1).cpu())
        return torch.cat(res)

    @torch.no_grad()
    def loo_relevance(self, v_tokens, question, block=2, max_new_tokens=10):
        """Oracle relevance: drop in log p(y_full) when a block x block group of tokens is removed.

        block=1 is true leave-one-token-out (576 passes); block=2 -> 144 passes; block=4 -> 36.
        """
        full = list(range(G.NUM_TOKENS))
        answer = self.generate(v_tokens, question, [full], max_new_tokens=max_new_tokens)[0]
        ans_ids = self.tok(answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ans_ids) == 0:
            return torch.zeros(G.NUM_TOKENS), answer
        base = self.answer_logprob(v_tokens, question, ans_ids, [full])[0]
        blocks = []
        for br in range(0, G.GRID, block):
            for bc in range(0, G.GRID, block):
                blocks.append({r * G.GRID + c for r in range(br, br + block) for c in range(bc, bc + block)})
        keep_sets = [[i for i in full if i not in blk] for blk in blocks]
        lps = self.answer_logprob(v_tokens, question, ans_ids, keep_sets)
        R = torch.zeros(G.NUM_TOKENS)
        for blk, lp in zip(blocks, lps):
            R[list(blk)] = torch.clamp(base - lp, min=0)
        return R, answer


class ClipRelevance:
    """R2: question/patch similarity computed with CLIP ViT-L/14-336 (the same tower LLaVA uses).

    Needs no LLM pass, so it can run on-device before any token is sent to the LLM.
    mode="maskclip" uses the value-embedding trick (Zhou et al., ECCV'22), which localises far
    better than raw patch tokens; mode="raw" uses final patch tokens.
    """

    def __init__(self, model_id="openai/clip-vit-large-patch14-336", device="cuda", dtype=torch.float16, mode="maskclip",
                 revision="refs/pr/20"):
        # The main branch of openai/clip-vit-large-patch14-336 only has pytorch_model.bin, which transformers>=4.50
        # refuses to torch.load on torch<2.6. refs/pr/20 is the official SFconvertbot safetensors conversion
        # (its 391 vision tensors are identical to LLaVA-1.5's vision tower).
        from transformers import CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(model_id, revision=revision, torch_dtype=dtype, use_safetensors=True).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(model_id, revision=revision)
        self.device, self.mode = device, mode

    @torch.no_grad()
    def __call__(self, pixel_values, question):
        vm = self.model.vision_model
        out = vm(pixel_values.to(self.model.dtype), output_hidden_states=True)
        if self.mode == "maskclip":
            last = vm.encoder.layers[-1]
            x = last.layer_norm1(out.hidden_states[-2])
            x = last.self_attn.out_proj(last.self_attn.v_proj(x))
        else:
            x = out.last_hidden_state
        patches = self.model.visual_projection(vm.post_layernorm(x[:, 1:]))[0]
        t = self.tok([question], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        text = self.model.get_text_features(**t)[0]
        return F.cosine_similarity(patches.float(), text.float().unsqueeze(0), dim=-1).cpu()
