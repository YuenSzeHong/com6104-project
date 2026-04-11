请创作恰好 {syllable_count} 个音节的粤语歌词。

参考文本类型：{reference_text_kind_label}

参考文本 / 创作意境：
{reference_text}

说明：如果参考文本类型是 原歌词，请将其视为已有原歌词或原文，尽量保留其叙事与意境；如果是 主题灵感，请将其视为创作灵感，不要逐句复述。

MIDI 内嵌歌词线索：
  来源：{embedded_lyrics_source}
  单位数：{embedded_lyric_unit_count}
  本次音节数依据：{effective_syllable_count_source}
  预览：{embedded_lyrics_str}

旋律约束：
  速度：{bpm} BPM
  调性：{key}
  强拍位置（音节索引）：{strong_str}
  押韵位置（音节索引）：{rhyme_str}
    音符时值（拍+音符值，quarter=1 beat）：{note_values_str}

0243 旋律目标：
  lean 0243 序列：{melody_tone_seq_str}

粤拼参考：
  参考文本最佳读音：{selected_jp}
  参考 1-6 声调序列：{reference_tone_seq_str}

强拍位置预查词库（可直接选用，无需调用工具）：
{sbc_summary}

主题相关候选词（已按声调码分类，可直接选用）：
{theme_summary}

创作要求：
1. 音节总数必须恰好等于 MIDI 要求的数量。
2. 以 0243.hk 默认 lean 0243 规则为主，优先贴合旋律目标码。
3. 强拍位置优先使用适合 sustained beat 的字，避免明显生硬的入声落点。
4. 押韵位置保持一致的韵尾。
5. 保留参考文本的艺术意境和情感基调。
6. **优先使用上方预查词库中的候选词**；若候选不足，可补充调用工具，但必须使用批量查询（单次多码），禁止逐条循环查询。
  批量示例：`find_words_by_tone_code(["0","43","14"])`、`find_tone_continuation(["青山","我要"], ["1","43"])`。
7. 若来源歌曲在现实中已有著名粤语填词版本，禁止直接复述、拼贴或只改少数字词；必须重新组织句子与意象。
8. 完成后用 score_lyrics 工具自检，优先检查 0243 旋律贴合度、音节数和押韵。

以 JSON 返回：
{{"lyrics": "<歌词，行间用\n分隔>", "jyutping": "<粤拼，音节间空格>", "lines": [{{"text": "...", "jyutping": "...", "syllable_count": 0}}], "rhyme_scheme": "AABB", "total_syllable_count": 0, "notes": "..."}}
