#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script trích xuất toàn bộ Lexicon và Metadata từ TextPreprocessor V40
xuất ra file reference_lexicon_v40.json.
"""

import json

class TextPreprocessorLexicon:
    EMOJI_MAP = {
        '😊': ' vui ', '😄': ' vui ', '😍': ' yeu ', '👍': ' tot ',
        '😞': ' buon ', '😡': ' gian ', '👎': ' te ', '💯': ' hoan_hao ',
        '✅': ' ok ', '❤️': ' yeu ', '😢': ' buon ', '😭': ' rat_buon ',
        '🥰': ' yeu ', '🤬': ' cuc_gian '
    }
    TEENCODE_MAP = {
        'ko': 'không', 'khong': 'không', 'k': 'không', 'kg': 'không',
        'dc': 'được', 'đc': 'được', 'j': 'gì', 'vs': 'với', 'ok': 'được',
        'vcl': 'quá', 'vl': 'quá', 'sp': 'sản phẩm', 'ship': 'giao hàng',
        'r': 'rồi', 'đỉnh': 'rất tốt', 'xịn': 'chất lượng', 'ib': 'nhắn tin',
        'hok': 'không', 'hem': 'không', 'bt': 'bình thường',
        'tệ': 'kém', 'tồi': 'kém', 'chán': 'không hài lòng'
    }
    POSITIVE_WORDS = {
        'tốt': 2, 'tuyệt': 3, 'hay': 2, 'đẹp': 2, 'thích': 2,
        'hài lòng': 2, 'ngon': 2, 'tuyệt vời': 3, 'chất lượng': 2,
        'nhanh': 2, 'đỉnh': 3, 'xịn': 2, 'ok': 1, 'bền': 2,
        'xuất sắc': 3, 'hoàn hảo': 3, 'rẻ': 1, 'đáng mua': 2,
        'tuyệt hảo': 3, 'yêu': 2, 'thân thiện': 2, 'nhiệt tình': 2
    }
    NEGATIVE_WORDS = {
        'tệ': 3, 'xấu': 2, 'tồi': 3, 'chậm': 2, 'hỏng': 3,
        'thất vọng': 3, 'buồn': 2, 'chán': 2, 'kém': 2,
        'fake': 3, 'rởm': 3, 'lừa': 3, 'rác': 3, 'đắt': 2,
        'lỗi': 2, 'treo': 3, 'tệ hại': 3, 'bực mình': 3,
        'khó chịu': 3, 'không hài lòng': 2
    }
    NEGATION_WORDS = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'k', 'ko', 'chớ', 'hok', 'hem'}
    INTENSIFIERS = {'rất': 1.5, 'quá': 1.5, 'cực': 2.0, 'siêu': 2.0, 'hơi': 0.7, 'khá': 1.2, 'cực kỳ': 2.0}
    CLAUSE_BOUNDARY = {'.', ',', ';', '!', '?', '...'}
    CONTRAST_MARKERS = {'nhưng', 'mà', 'tuy nhiên', 'ai ngờ', 'vậy mà', 'thế mà'}
    SENTENCE_FINAL_PARTICLES = {'vậy', 'ạ', 'nhỉ', 'nhé', 'à', 'hả', 'thế', 'đấy', 'chứ', 'ha', 'hử', 'cơ'}
    LOCATIVE_DAU_BEFORE = {'ở', 'tại', 'đi', 'đến', 'tới', 'về', 'từ', 'sang', 'qua'}
    LOCATIVE_DAU_AFTER = {'rồi', 'nào', 'nữa'}
    NEUTRAL_THRESHOLD_DEFAULT = 2.0

def export_lexicon():
    data = {
        "metadata": {
            "version": "v40",
            "description": "Reference Lexicon & Rule Assets extracted from V40 TextPreprocessor",
            "neutral_threshold_default": TextPreprocessorLexicon.NEUTRAL_THRESHOLD_DEFAULT,
            "total_positive_words": len(TextPreprocessorLexicon.POSITIVE_WORDS),
            "total_negative_words": len(TextPreprocessorLexicon.NEGATIVE_WORDS),
            "total_teencode_mappings": len(TextPreprocessorLexicon.TEENCODE_MAP),
            "total_emoji_mappings": len(TextPreprocessorLexicon.EMOJI_MAP)
        },
        "emoji_map": TextPreprocessorLexicon.EMOJI_MAP,
        "teencode_map": TextPreprocessorLexicon.TEENCODE_MAP,
        "positive_words": TextPreprocessorLexicon.POSITIVE_WORDS,
        "negative_words": TextPreprocessorLexicon.NEGATIVE_WORDS,
        "negation_words": sorted(list(TextPreprocessorLexicon.NEGATION_WORDS)),
        "intensifiers": TextPreprocessorLexicon.INTENSIFIERS,
        "clause_boundary": sorted(list(TextPreprocessorLexicon.CLAUSE_BOUNDARY)),
        "contrast_markers": sorted(list(TextPreprocessorLexicon.CONTRAST_MARKERS)),
        "sentence_final_particles": sorted(list(TextPreprocessorLexicon.SENTENCE_FINAL_PARTICLES)),
        "locative_dau_rules": {
            "before_words": sorted(list(TextPreprocessorLexicon.LOCATIVE_DAU_BEFORE)),
            "after_words": sorted(list(TextPreprocessorLexicon.LOCATIVE_DAU_AFTER))
        }
    }
    
    output_filename = "reference_lexicon_v40.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã xuất lexicon và metadata ra file: {output_filename}")

if __name__ == "__main__":
    export_lexicon()
