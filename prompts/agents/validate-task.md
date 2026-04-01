请对以下粤语歌词草稿进行全面校验。

【草稿歌词】
{draft_lyrics}

【草稿粤拼】
{draft_jyutping}

【校验参数】
  MIDI 要求音节数：{syllable_count}
  预期 lean 0243 序列：{tone_seq_str}
  强拍位置：{strong_str}
  押韵位置：{rhyme_str}
  参考文本：{reference_text}

【MIDI 内嵌歌词线索】
  来源：{embedded_lyrics_source}
  单位数：{embedded_lyric_unit_count}
  本次音节数依据：{effective_syllable_count_source}
  预览：{embedded_lyrics_str}

【操作步骤】
1. 若草稿粤拼为空，先调用 chinese_to_jyutping 获取草稿歌词的粤拼。
2. 结合预期 lean 0243 序列，评估歌词是否贴合旋律走向与强拍/句末落点。
3. 调用 score_lyrics 工具，传入以下参数：
   - lyrics: 草稿歌词原文
   - draft_jyutping: 草稿粤拼
   - expected_tone_sequence: {tone_json}
   - strong_beat_positions: {strong_json}
   - rhyme_positions: {rhyme_json}
   - expected_syllable_count: {syllable_count}
4. 调用 suggest_corrections 工具，将 score_lyrics 的 JSON 输出传入，获取优先级修改建议。
5. 在 score_lyrics 的客观评分基础上，由你主观评判艺术质量（意象、语言自然度、情感共鸣），给出 0–10 的艺术分，并入综合评分（艺术质量权重 15%）。
6. 若草稿明显贴近现实中已知的著名粤语版本，必须在 feedback 与 corrections 中明确指出“过度贴近已知版本”，并建议重写相关句段。
7. 以 JSON 返回最终校验结果：
{{"score": 0.0, "passed": false, "feedback": "说明", "corrections": ["修改项1"], "melody_0243_score": 0.0, "dimension_scores": {{"syllable_count": 0.0, "tonal_accuracy": 0.0, "rhyme_consistency": 0.0, "artistic_quality": 0.0}}}}
