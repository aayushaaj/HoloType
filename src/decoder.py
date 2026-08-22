# -*- coding: utf-8 -*-
"""Decoder - Noisy-channel decoder with spatial confidence integration."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import os

try:
    from spellchecker import SpellChecker
    HAS_SPELLCHECKER = True
except ImportError:
    HAS_SPELLCHECKER = False


@dataclass
class KeyObservation:
    key: str
    finger: str
    xy_dist: float
    z_dist: float
    tap_confidence: float
    timestamp: float


class SpatialConfidenceModel:
    def __init__(self):
        self._fitted = False

    def fit(self, calibration_samples: List[KeyObservation]):
        self._fitted = True

    def emission_logprob(self, observed: KeyObservation, intended_key: str) -> float:
        if observed.key == intended_key:
            xy_penalty = observed.xy_dist * 50
            z_penalty = observed.z_dist * 20
            conf_bonus = observed.tap_confidence * 2
            return -(xy_penalty + z_penalty) + conf_bonus
        return -10.0


class NGramLanguageModel:
    def __init__(self, n: int = 3):
        self.n = n
        self.counts = defaultdict(Counter)
        self.total = defaultdict(int)
        self.vocab = set()

    def train(self, corpus: List[str]):
        for word in corpus:
            word = word.lower()
            self.vocab.add(word)
            padded = " " * (self.n - 1) + word + " "
            for i in range(len(padded) - self.n + 1):
                context = padded[i:i+self.n-1]
                char = padded[i+self.n-1]
                self.counts[context][char] += 1
                self.total[context] += 1

    def logprob(self, word: str) -> float:
        word = word.lower()
        padded = " " * (self.n - 1) + word + " "
        logp = 0.0
        for i in range(len(padded) - self.n + 1):
            context = padded[i:i+self.n-1]
            char = padded[i+self.n-1]
            count = self.counts[context].get(char, 0)
            total = self.total[context]
            if total == 0:
                logp += np.log(1e-6)
            else:
                logp += np.log((count + 0.1) / (total + 0.1 * len(self.counts[context])))
        return logp


class NoisyChannelDecoder:
    def __init__(self, vocabulary_path: str = None):
        self.buffer: List[KeyObservation] = []
        self.decoded_text = ""
        self.spatial_model = SpatialConfidenceModel()
        self.lm = NGramLanguageModel(n=3)
        self.word_freq = Counter()

        if vocabulary_path and os.path.exists(vocabulary_path):
            with open(vocabulary_path) as f:
                words = [line.strip().lower() for line in f if line.strip()]
        else:
            words = self._default_vocabulary()

        self.lm.train(words)
        for w in words:
            self.word_freq[w] += 1

        if HAS_SPELLCHECKER:
            self.spell = SpellChecker()
            self.spell.word_frequency.load_words(words)
        else:
            self.spell = None

    def _default_vocabulary(self) -> List[str]:
        return [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "is", "are", "was", "were", "been", "has", "had", "were", "said", "each",
            "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "packing", "boxes",
            "with", "zest", "very", "zombies", "jump", "packing", "my", "box", "five",
            "dozen", "liquor", "jugs", "how", "vexingly", "daft", "zebras",
            "hello", "world", "test", "typing", "air", "keyboard", "camera", "hand",
            "finger", "tap", "glass", "morphism", "visual", "interface", "system",
        ]

    def feed_key(self, key: str, finger: str, xy_dist: float, z_dist: float,
                 tap_confidence: float, timestamp: float):
        obs = KeyObservation(key, finger, xy_dist, z_dist, tap_confidence, timestamp)
        if key == " ":
            self._flush_word()
        elif key.isalpha():
            self.buffer.append(obs)

    def _flush_word(self):
        if not self.buffer:
            self.decoded_text += " "
            return
        raw_word = "".join(obs.key for obs in self.buffer)
        corrected = self._correct_word(raw_word, self.buffer)
        self.decoded_text += corrected + " "
        self.buffer = []

    def _correct_word(self, raw_word: str, observations: List[KeyObservation]) -> str:
        if not raw_word:
            return ""
        if raw_word in self.word_freq and all(o.tap_confidence > 0.7 for o in observations):
            return raw_word

        candidates = set()
        if self.spell:
            candidates.update(self.spell.candidates(raw_word) or [])
        for word in self.word_freq:
            if self._edit_distance(raw_word, word) <= 2:
                candidates.add(word)
        if not candidates:
            return raw_word

        best_word, best_score = raw_word, float("-inf")
        for candidate in candidates:
            score = self._score_candidate(candidate, observations)
            if score > best_score:
                best_score, best_word = score, candidate
        return best_word

    def _score_candidate(self, word: str, observations: List[KeyObservation]) -> float:
        if len(word) != len(observations):
            length_penalty = -abs(len(word) - len(observations)) * 5
        else:
            length_penalty = 0

        emission_logp = 0.0
        for i, obs in enumerate(observations):
            if i < len(word):
                emission_logp += self.spatial_model.emission_logprob(obs, word[i])
            else:
                emission_logp += -10.0

        lm_logp = self.lm.logprob(word)
        freq_logp = np.log(self.word_freq.get(word, 1) + 1)

        return 1.0 * emission_logp + 0.5 * lm_logp + 0.3 * freq_logp + length_penalty

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def finalize(self) -> str:
        self._flush_word()
        return self.decoded_text.strip()

    def get_raw_buffer(self) -> str:
        return "".join(obs.key for obs in self.buffer)


class RawSequenceDecoder:
    def __init__(self):
        self.buffer = []
        self.decoded_text = ""
        if HAS_SPELLCHECKER:
            self.spell = SpellChecker()
        else:
            self.spell = None

    def feed_key(self, key: str, confidence: float = 1.0):
        if key == " ":
            self._flush_word()
        elif key.isalpha():
            self.buffer.append(key)

    def _flush_word(self):
        if not self.buffer:
            return
        raw_word = "".join(self.buffer)
        corrected = self._correct(raw_word)
        self.decoded_text += corrected + " "
        self.buffer = []

    def _correct(self, raw_word: str) -> str:
        if self.spell is None:
            return raw_word
        correction = self.spell.correction(raw_word)
        return correction if correction else raw_word

    def finalize(self) -> str:
        self._flush_word()
        return self.decoded_text.strip()

    def get_raw_buffer(self) -> str:
        return "".join(self.buffer)