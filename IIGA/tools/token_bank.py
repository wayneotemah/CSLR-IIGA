from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class TokenEntry:
    token_id: int
    token_text: str
    sample_id: str
    span_start: int
    span_end: int
    span_length: int
    score: float
    embedding: list[float]
    metadata: dict[str, Any]


class TokenBank:
    def __init__(self) -> None:
        self._entries_by_token: dict[int, list[TokenEntry]] = defaultdict(list)

    def register(self, entry: TokenEntry) -> None:
        self._entries_by_token[int(entry.token_id)].append(entry)

    def query_by_token(self, token_id: int) -> list[TokenEntry]:
        return list(self._entries_by_token.get(int(token_id), []))

    def token_ids(self) -> list[int]:
        return sorted(self._entries_by_token.keys())

    def num_entries(self) -> int:
        return sum(len(entries) for entries in self._entries_by_token.values())

    def summary(self) -> dict[str, Any]:
        return {
            'token_count': len(self._entries_by_token),
            'entry_count': self.num_entries(),
            'entries_per_token': {
                str(token_id): len(entries)
                for token_id, entries in sorted(self._entries_by_token.items())
            },
        }

    def top_k_similar(self, query_embedding: torch.Tensor, token_id: int | None = None, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        if query_embedding.ndim != 1:
            raise ValueError(f'Expected 1D query embedding, got shape {tuple(query_embedding.shape)}')

        candidates = []
        entries = []
        token_iter = [int(token_id)] if token_id is not None else self.token_ids()
        for current_token_id in token_iter:
            for entry in self._entries_by_token.get(current_token_id, []):
                entries.append(entry)
                candidates.append(torch.tensor(entry.embedding, dtype=query_embedding.dtype))

        if not candidates:
            return []

        stacked = torch.stack(candidates)
        normalized_bank = F.normalize(stacked, dim=-1)
        normalized_query = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        similarities = torch.mm(normalized_query, normalized_bank.t()).squeeze(0)
        top_k = min(k, similarities.numel())
        values, indices = torch.topk(similarities, k=top_k)
        results = []
        for value, index in zip(values.tolist(), indices.tolist()):
            entry = entries[index]
            results.append({
                'similarity': float(value),
                'token_id': int(entry.token_id),
                'token_text': entry.token_text,
                'sample_id': entry.sample_id,
                'span_start': int(entry.span_start),
                'span_end': int(entry.span_end),
                'span_length': int(entry.span_length),
                'score': float(entry.score),
                'metadata': entry.metadata,
            })
        return results

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'summary': self.summary(),
            'entries': {
                str(token_id): [asdict(entry) for entry in entries]
                for token_id, entries in sorted(self._entries_by_token.items())
            },
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        return target

    @classmethod
    def load(cls, path: str | Path) -> 'TokenBank':
        source = Path(path)
        payload = json.loads(source.read_text(encoding='utf-8'))
        bank = cls()
        for token_id, entries in payload.get('entries', {}).items():
            for raw_entry in entries:
                bank.register(TokenEntry(
                    token_id=int(raw_entry['token_id']),
                    token_text=str(raw_entry['token_text']),
                    sample_id=str(raw_entry['sample_id']),
                    span_start=int(raw_entry['span_start']),
                    span_end=int(raw_entry['span_end']),
                    span_length=int(raw_entry['span_length']),
                    score=float(raw_entry['score']),
                    embedding=[float(value) for value in raw_entry['embedding']],
                    metadata=dict(raw_entry.get('metadata', {})),
                ))
        return bank
