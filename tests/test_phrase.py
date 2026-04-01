import sys
import json

sys.path.insert(0, 'mcp-servers/midi-analyzer')
from server import analyze_midi

result = json.loads(analyze_midi('test/midi/ドラえもんのうた.mid'))

print(f"音节数: {result['syllable_count']}")
print(f"乐句数: {len(result.get('phrase_boundaries', []))}")
print("\n乐句分割:")
for i, seg in enumerate(result.get('phrase_boundaries', [])):
    print(f"  乐句{i+1}: 音节[{seg['start_idx']}-{seg['end_idx']}] "
          f"({seg['note_count']}个音), "
          f"时长{seg['duration_sec']}s, "
          f"间隔{seg['gap_before_sec']}s")
