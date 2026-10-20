"""Experimental complete-text medical classifier; NEVER display authorization."""
import hashlib
import json
from pathlib import Path
import time

MAX_WINDOW = 256
OVERLAP = 64
MAX_INPUT = 4096


def token_spans(length, width=MAX_WINDOW-2, overlap=OVERLAP):
    if length < 1 or width < 2 or not 0 <= overlap < width:
        raise ValueError('Invalid coverage budget')
    start = 0
    result = []
    while start < length:
        end = min(start+width, length)
        result.append((start, end))
        if end == length:
            break
        start = end-overlap
    return result


def encode_windows(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    if not ids or len(ids) > MAX_INPUT:
        return None, len(ids)
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError('DeBERTa special-token contract missing')
    windows = []
    for start, end in token_spans(len(ids)):
        window = [tokenizer.cls_token_id]+ids[start:end]+[tokenizer.sep_token_id]
        windows.append({'input_ids': window, 'attention_mask': [1]*len(window)})
    return windows, len(ids)


def probabilities(model, tokenizer, text, torch):
    windows, count = encode_windows(tokenizer, text)
    if windows is None:
        return None, count, 0
    outputs = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for window in windows:
            encoded = {k: torch.tensor([v], device='cuda') for k, v in window.items()}
            outputs.append(model(**encoded).logits.float().sigmoid()[0])
    return torch.stack(outputs).amax(0).cpu().tolist(), count, len(windows)


class ParagraphOutputModel:
    """Versioned calibration paired with exact weights and coverage settings."""
    def __init__(self, root):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .output_training_data import LABELS
        root = Path(root)
        manifest = json.loads((root/'training_manifest.json').read_text(encoding='utf-8'))
        calibration = json.loads((root/'calibration.json').read_text(encoding='utf-8'))
        with (root/'model.safetensors').open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if (manifest['model_sha256'] != checksum or manifest['calibration'] != calibration
                or calibration['labels'] != list(LABELS)
                or calibration['window_tokens'] != MAX_WINDOW or calibration['overlap_tokens'] != OVERLAP
                or calibration['max_input_tokens'] != MAX_INPUT):
            raise ValueError('Unpaired paragraph checkpoint or inference contract')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no silent CPU fallback')
        self.torch, self.labels, self.thresholds = torch, LABELS, calibration['thresholds']
        self.model_sha256 = checksum
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(root, local_files_only=True).float().cuda().eval()

    def assess(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            return {'status': 'OUT_OF_SCOPE_INPUT', 'codes': [], 'no_display_authorization': True}
        self.torch.cuda.synchronize()
        tick = time.perf_counter()
        scores, count, windows = probabilities(self.model, self.tokenizer, text, self.torch)
        self.torch.cuda.synchronize()
        return {'status': 'ASSESSED' if scores is not None else 'OUT_OF_SCOPE_LENGTH',
            'codes': [c for c, p, t in zip(self.labels, scores or [], self.thresholds) if p >= t],
            'scores': scores, 'tokens': count, 'windows': windows,
            'latency_ms': (time.perf_counter()-tick)*1000,
            'model_sha256': self.model_sha256, 'no_display_authorization': True}
