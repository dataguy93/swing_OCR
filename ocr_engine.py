import os
import base64
import json
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image, ImageOps

client = genai.Client(
api_key=os.environ.get("GEMINI_API_KEY")
)

MODEL_PRIMARY = "gemini-2.5-flash"
MODEL_FAST = "gemini-2.5-flash"

SCORECARD_PROMPT = """You are an expert golf scorecard OCR system. Extract every number and name from this golf scorecard photo with extreme precision.

OUTPUT CONTRACT (MUST FOLLOW EXACTLY):
- Return ONLY valid JSON.
- Do not wrap JSON in markdown.
- Do not use ```json fences.
- Do not include explanation before or after JSON.

SCORECARD LAYOUT (columns left to right):
NAME | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | OUT | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | IN | TOT | HCP | NET

KEY DEFINITIONS:
- OUT = front 9 subtotal (holes 1-9), IN = back 9 subtotal (holes 10-18), TOT = OUT + IN
- HCP = handicap, NET = TOT minus HCP

STEP 1 — PAR ROW:
The PAR row is PRINTED text (not handwritten), labeled "Par" or "PAR". Read each of the 18 par values (each MUST be 3, 4, or 5). Also read printed OUT and IN subtotals.
VERIFY: front 9 pars MUST sum to printed OUT. Back 9 pars MUST sum to printed IN. If not, re-read.

STEP 1B — HOLE HANDICAP ROW (STROKE INDEX):
Look for a PRINTED row that indicates the MEN'S hole-by-hole difficulty ranking. Common labels include "Ventaja Caballeros", "Men's Handicap", "Handicap", "HDCP", "HCP", or "Ventaja". If the scorecard has BOTH a men's row ("Ventaja Caballeros" / "Men's Handicap") and a women's row ("Ventaja Damas" / "Ladies Handicap"), use the MEN'S row. This row ranks each hole's difficulty from 1 (hardest) to 18 (easiest). Read all 18 values into the "hole_handicaps" array. Each value MUST be an integer 1-18 with no duplicates. If this row is not present on the scorecard, set "hole_handicaps" to null.

STEP 2 — PLAYER ROWS:
For each player (handwritten row):
- Read NAME from leftmost column.
- Read EXACTLY 18 hole scores (raw strokes, typically 2-9), one per column. Output exactly 18 integers in the holes array.
- STOP after 18 scores. Do NOT include OUT/IN/TOT/HCP/NET in the holes array — they go in separate fields.
- Read OUT, IN, TOT from their columns into front_9_total, back_9_total, gross_total fields.
- HCP column: second-to-last column (after TOT, before NET), header may say "HCP"/"HDCP"/"H/C". Read for EVERY player. If blank, set handicap to null.
- Read scores EXACTLY as written. NEVER change a score to match written totals — hole scores are authoritative.
- Ignore circles, squares, triangles around numbers — read only the digit inside.
- If "+1"/"-1"/"E"/"0" notation is used instead of raw strokes, convert to actual strokes using par.
- Card may be UPSIDE DOWN or ROTATED — use PAR label and hole numbers to orient.

STEP 3 — HANDWRITING TIPS:
- "4" vs "9": 4 has open angular top, 9 has closed round loop
- "5" vs "6": 5 has flat/angular top stroke, 6 has rounded top
- "3" vs "8": 3 has open curves, 8 has two closed loops. Shapes around a "3" can make it look like "8" — focus on the digit itself.
- "1" vs "7": 1 is simple vertical stroke, 7 has horizontal top bar
- "4" vs "1": These can look similar — check carefully.
- Score of 0 is IMPOSSIBLE in golf. Every hole must have at least 1 stroke. If you think you see a 0, look again — it is almost certainly a different digit (6, 8, 9, or 3). If after careful re-examination it truly appears to be 0, report it as 0 and flag in issues.
- Score of 1 (ace) is valid on ANY hole.
- HCP column: a single vertical stroke or mark is 1, not 0. HCP of 0 (scratch golfer) is rare — if you see any mark at all in the HCP cell, it is most likely a number (1, 2, etc.), not zero.

Return ONLY compact JSON (no extra whitespace):
{"course_name":"name or null","par":[18 ints],"par_front_9_total":N,"par_back_9_total":N,"hole_handicaps":[18 ints] or null,"players":[{"name":"str","holes":[18 ints],"handicap":N_or_null,"front_9_total":N,"back_9_total":N,"gross_total":N,"notes":"str"}],"confidence":"HIGH"|"MEDIUM"|"LOW","issues":["str"],"card_type":"STANDARD_18"|"FRONT_9"|"BACK_9"|"OTHER","low_confidence_holes":[{"player":"str","hole":N,"extracted":N,"reason":"str"}]}"""


def strip_subtotal_columns(values):
    if not values or not isinstance(values, list):
        return values
    if len(values) == 18:
        return values

    def is_subtotal(val, expected_sum):
        if val is None or not isinstance(val, (int, float)):
            return False
        return val > 9 or abs(val - expected_sum) <= 3

    if len(values) == 21:
        return values[:9] + values[10:19]

    if len(values) == 20:
        front_9 = values[:9]
        front_sum = sum(v for v in front_9 if isinstance(v, (int, float)) and v is not None)
        if is_subtotal(values[9], front_sum):
            return front_9 + values[10:19]
        back_check = values[9:18]
        back_sum = sum(v for v in back_check if isinstance(v, (int, float)) and v is not None)
        if is_subtotal(values[18], back_sum):
            return values[:9] + values[9:18]
        return values[:9] + values[10:19]

    if len(values) == 19:
        front_9 = values[:9]
        front_sum = sum(v for v in front_9 if isinstance(v, (int, float)) and v is not None)
        if is_subtotal(values[9], front_sum):
            return front_9 + values[10:19]
        back_9 = values[9:18]
        back_sum = sum(v for v in back_9 if isinstance(v, (int, float)) and v is not None)
        if is_subtotal(values[18], back_sum):
            return values[:9] + back_9
        return values[:9] + values[10:19]

    if len(values) > 18:
        return values[:9] + values[len(values)-9:]
    return values


def _strip_handicap_subtotals(values):
    """Remove OUT/IN/TOT positions from a hole handicap array.

    Unlike scores, handicap values (1-18) can't be distinguished from
    subtotals by magnitude.  Instead, remove values at the known subtotal
    positions (index 9 = OUT, index 19 = IN) and try every combination
    until we get exactly 18 unique values in range 1-18.
    """
    if not values or not isinstance(values, list):
        return values
    if len(values) == 18:
        return values

    # 21 elements: OUT at 9, IN at 19, TOT at 20
    if len(values) == 21:
        candidate = values[:9] + values[10:19]
        if len(set(v for v in candidate if v is not None and 1 <= v <= 18)) == 18:
            return candidate

    # 20 elements: likely OUT at 9 and IN at 19 (no TOT), or missing one
    if len(values) == 20:
        for drop1, drop2 in [(9, 19), (9, 18), (9, 17)]:
            candidate = [v for i, v in enumerate(values) if i not in (drop1, drop2)]
            valid = [v for v in candidate if v is not None and 1 <= v <= 18]
            if len(valid) == 18 and len(set(valid)) == 18:
                return candidate

    # 19 elements: one subtotal column (OUT or IN)
    if len(values) == 19:
        for drop in [9, 18]:
            candidate = values[:drop] + values[drop + 1:]
            valid = [v for v in candidate if v is not None and 1 <= v <= 18]
            if len(valid) == 18 and len(set(valid)) == 18:
                return candidate

    # Fallback: filter to only valid 1-18 values and take first 18 unique
    seen = set()
    filtered = []
    for v in values:
        if v is not None and 1 <= v <= 18 and v not in seen:
            seen.add(v)
            filtered.append(v)
    if len(filtered) == 18:
        return filtered

    return values


def _normalize_image_orientation(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if h > w * 1.5:
            img = img.rotate(90, expand=True)
            w, h = img.size
        MAX_DIM = 3000
        if w > MAX_DIM or h > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=90)
        return buf.getvalue(), 'image/jpeg'
    except Exception:
        return image_bytes, None


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    return media_types.get(ext, "image/jpeg")


def _parse_json_response(raw_text):
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()
    json_start = raw_text.find('{')
    json_end = raw_text.rfind('}')
    if json_start >= 0 and json_end > json_start:
        raw_text = raw_text[json_start:json_end + 1]
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            repaired = _repair_truncated_json(raw_text)
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            raise json.JSONDecodeError("Could not parse or repair JSON", raw_text[:200], 0)


def _repair_truncated_json(text):
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False
    last_good = 0
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            open_braces += 1
        elif c == '}':
            open_braces -= 1
        elif c == '[':
            open_brackets += 1
        elif c == ']':
            open_brackets -= 1
        if open_braces >= 0 and open_brackets >= 0:
            last_good = i
    result = text[:last_good + 1]
    if in_string:
        result += '"'
    while open_brackets > 0:
        result += ']'
        open_brackets -= 1
    while open_braces > 0:
        result += '}'
        open_braces -= 1
    return result


def _call_gemini(prompt_text, image_bytes, media_type, model=None, max_tokens=4096):
    import time as _time
    if model is None:
        model = MODEL_PRIMARY
    last_err = None
    for attempt in range(3):
        try:
            if attempt > 0:
                _time.sleep(3 + attempt * 2)
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt_text),
                            types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            text = ""
            if hasattr(response, 'text') and response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text += part.text
            if not text:
                raise ValueError("Empty response from Gemini")
            return text
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "ApiKeyNotApproved" in err_str or "401" in err_str:
                continue
            if "Empty response" in err_str:
                continue
            raise
    raise last_err


def _normalize_name(n):
    return (n or "").strip().lower().replace(".", "").replace("-", " ")


def _match_player(sp_name, result_players):
    sp_norm = _normalize_name(sp_name)
    for p in result_players:
        if _normalize_name(p.get("name")) == sp_norm:
            return p
    for p in result_players:
        rn = _normalize_name(p.get("name"))
        if sp_norm and rn and (sp_norm in rn or rn in sp_norm):
            return p
    return None


def ocr_scorecard(image_path):
    if not os.path.exists(image_path):
        return {
            "error": f"Image file not found: {image_path}",
            "players": [],
            "confidence": "LOW",
            "issues": ["File not found"]
        }

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_bytes, normalized_media_type = _normalize_image_orientation(image_bytes)
        media_type = normalized_media_type or get_image_media_type(image_path)

        raw_text = _call_gemini(SCORECARD_PROMPT, image_bytes, media_type, model=MODEL_PRIMARY)
        try:
            result = _parse_json_response(raw_text)
        except json.JSONDecodeError:
            import time as _t0
            _t0.sleep(3)
            raw_text = _call_gemini(
                SCORECARD_PROMPT + "\n\nIMPORTANT: STRICT JSON ONLY. No markdown, no code fences, no prose.",
                image_bytes, media_type, model=MODEL_PRIMARY
            )
            result = _parse_json_response(raw_text)

        par = result.get("par", [])
        if len(par) != 18:
            par = strip_subtotal_columns(par)
            result["par"] = par

        hole_handicaps = result.get("hole_handicaps")
        if hole_handicaps is not None:
            if isinstance(hole_handicaps, list):
                hole_handicaps = [int(v) if isinstance(v, (int, float)) else None for v in hole_handicaps]
                if len(hole_handicaps) != 18:
                    hole_handicaps = _strip_handicap_subtotals(hole_handicaps)
                valid_vals = [v for v in hole_handicaps if v is not None and 1 <= v <= 18]
                if len(valid_vals) == 18 and len(set(valid_vals)) == 18:
                    result["hole_handicaps"] = valid_vals
                else:
                    if "issues" not in result:
                        result["issues"] = []
                    result["issues"].append(
                        f"Warning: hole_handicaps extracted but invalid (got {hole_handicaps}). Discarding."
                    )
                    result["hole_handicaps"] = None
            else:
                result["hole_handicaps"] = None

        for player in result.get("players", []):
            if player.get("name"):
                player["name"] = player["name"].upper()
            holes = player.get("holes", [])
            cleaned_holes = []
            for h in holes:
                if h is None:
                    cleaned_holes.append(None)
                elif isinstance(h, (int, float)):
                    cleaned_holes.append(int(h))
                else:
                    cleaned_holes.append(None)
            if len(cleaned_holes) != 18:
                cleaned_holes = strip_subtotal_columns(cleaned_holes)
            if len(cleaned_holes) < 18:
                if "issues" not in result:
                    result["issues"] = []
                result["issues"].append(
                    f"Warning: {player.get('name','?')} only has {len(cleaned_holes)} holes extracted (expected 18)"
                )
                while len(cleaned_holes) < 18:
                    cleaned_holes.append(None)
            player["holes"] = cleaned_holes

            if player.get("handicap") is not None:
                try:
                    player["handicap"] = int(player["handicap"])
                except (ValueError, TypeError):
                    player["handicap"] = None

            if player.get("gross_total") is not None:
                try:
                    player["gross_total"] = int(player["gross_total"])
                except (ValueError, TypeError):
                    player["gross_total"] = None

            if player.get("front_9_total") is not None:
                try:
                    player["front_9_total"] = int(player["front_9_total"])
                except (ValueError, TypeError):
                    player["front_9_total"] = None
            if player.get("back_9_total") is not None:
                try:
                    player["back_9_total"] = int(player["back_9_total"])
                except (ValueError, TypeError):
                    player["back_9_total"] = None

        result = _fix_swapped_subtotals(result)

        for player in result.get("players", []):
            holes = player.get("holes", [])
            if len(holes) != 18:
                holes = strip_subtotal_columns(holes)
                player["holes"] = holes
            valid = [h for h in holes if h is not None]
            if valid:
                computed_total = sum(valid)
                written_total = player.get("gross_total")
                player["written_gross_total"] = written_total
                player["gross_total"] = computed_total
                if player.get("front_9_total"):
                    player["front_9_total"] = int(player["front_9_total"])
                if player.get("back_9_total"):
                    player["back_9_total"] = int(player["back_9_total"])

        par = result.get("par", [])
        if len(par) == 18:
            par_front_printed = result.get("par_front_9_total")
            par_back_printed = result.get("par_back_9_total")
            if par_front_printed and par_back_printed:
                pf = int(par_front_printed)
                pb = int(par_back_printed)
                comp_pf = sum(par[:9])
                comp_pb = sum(par[9:18])
                cur_diff = abs(comp_pf - pf) + abs(comp_pb - pb)
                swap_diff = abs(comp_pf - pb) + abs(comp_pb - pf)
                if swap_diff < cur_diff:
                    result["par_front_9_total"] = par_back_printed
                    result["par_back_9_total"] = par_front_printed
                    par_front_printed = par_back_printed
                    par_back_printed = result["par_back_9_total"]
            if par_front_printed:
                par_front_printed = int(par_front_printed)
                par_front_computed = sum(par[:9])
                if par_front_computed != par_front_printed:
                    diff = par_front_computed - par_front_printed
                    if abs(diff) <= 2:
                        for i in range(9):
                            alt = par[i] - diff
                            if alt in (3, 4, 5) and alt != par[i]:
                                par[i] = alt
                                break
                            if sum(par[:9]) == par_front_printed:
                                break
            if par_back_printed:
                par_back_printed = int(par_back_printed)
                par_back_computed = sum(par[9:18])
                if par_back_computed != par_back_printed:
                    diff = par_back_computed - par_back_printed
                    if abs(diff) <= 2:
                        for i in range(9, 18):
                            alt = par[i] - diff
                            if alt in (3, 4, 5) and alt != par[i]:
                                par[i] = alt
                                break
                            if sum(par[9:18]) == par_back_printed:
                                break
            result["par"] = par

        result = _programmatic_crosscheck(result)

        for p in result.get("players", []):
            holes = p.get("holes", [])
            computed = sum(h for h in holes if h is not None)
            p["gross_total"] = computed
            wt = p.get("written_gross_total")
            if wt and computed != wt:
                if "issues" not in result:
                    result["issues"] = []
                result["issues"].append(
                    f"Note: {p.get('name', '?')} hole sum={computed}, card total={wt} "
                    f"(diff={computed - wt}). Hole scores are authoritative."
                )

        result = _enrich_holes_with_confidence(result)

        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse OCR response: {str(e)}",
            "raw_response": raw_text if 'raw_text' in dir() else "No response",
            "players": [],
            "confidence": "LOW",
            "issues": ["JSON parse error from AI response"]
        }
    except Exception as e:
        return {
            "error": f"OCR processing failed: {str(e)}",
            "players": [],
            "confidence": "LOW",
            "issues": [str(e)]
        }


def _fix_swapped_subtotals(result):
    if "issues" not in result:
        result["issues"] = []

    has_clear_swap = False
    player_data = []

    for player in result.get("players", []):
        written_front = player.get("front_9_total")
        written_back = player.get("back_9_total")
        written_total = player.get("gross_total")
        holes = player.get("holes", [])

        if written_front is None or written_back is None or len(holes) < 18:
            player_data.append(None)
            continue

        if written_total and written_front + written_back != written_total:
            player_data.append(None)
            continue

        comp_front = sum(h for h in holes[:9] if h is not None)
        comp_back = sum(h for h in holes[9:18] if h is not None)

        current_diff = abs(comp_front - written_front) + abs(comp_back - written_back)
        swapped_diff = abs(comp_front - written_back) + abs(comp_back - written_front)

        if swapped_diff < current_diff:
            has_clear_swap = True

        player_data.append({
            "player": player,
            "written_front": written_front,
            "written_back": written_back,
            "comp_front": comp_front,
            "comp_back": comp_back,
            "current_diff": current_diff,
            "swapped_diff": swapped_diff,
        })

    swap_direction = None
    for pd in player_data:
        if pd is None:
            continue
        if pd["swapped_diff"] < pd["current_diff"]:
            wf = pd["written_front"]
            wb = pd["written_back"]
            if wf > wb:
                swap_direction = "big_first"
            elif wb > wf:
                swap_direction = "small_first"
            break

    for pd in player_data:
        if pd is None:
            continue

        player = pd["player"]
        name = player.get("name", "?")
        wf = pd["written_front"]
        wb = pd["written_back"]

        do_swap = False
        if pd["swapped_diff"] < pd["current_diff"]:
            do_swap = True
        elif has_clear_swap and wf != wb and swap_direction:
            if swap_direction == "big_first" and wf < wb:
                do_swap = True
            elif swap_direction == "small_first" and wb < wf:
                do_swap = True

        if do_swap:
            player["front_9_total"], player["back_9_total"] = wb, wf
            result["issues"].append(
                f"Subtotal swap fix: {name} front/back subtotals were swapped "
                f"(OUT={wf}→{wb}, IN={wb}→{wf}). "
                f"Computed F9={pd['comp_front']}, B9={pd['comp_back']}."
            )

    return result


def _programmatic_crosscheck(result):
    par = result.get("par", [])
    if "issues" not in result:
        result["issues"] = []

    flagged_holes = []

    for player in result.get("players", []):
        holes = player.get("holes", [])
        name = player.get("name", "?")

        for i in range(min(len(holes), len(par))):
            if holes[i] is None:
                continue
            hole_par = par[i] if i < len(par) else 4
            score = holes[i]
            if score <= 0:
                result["issues"].append(
                    f"FLAG: {name} hole {i+1} score is {score} — score of 0 or less is impossible in golf. Please verify and correct."
                )
                flagged_holes.append({"player": name, "hole": i+1, "score": score, "reason": f"Score {score} is impossible in golf"})
            else:
                diff = score - hole_par
                if diff >= 5:
                    result["issues"].append(
                        f"Flag: {name} hole {i+1} score {score} is +{diff} over par {hole_par} — verify"
                    )
                    flagged_holes.append({"player": name, "hole": i+1, "score": score, "reason": f"+{diff} over par {hole_par}"})

        player["holes"] = holes
        player["gross_total"] = sum(h for h in holes if h is not None)

    if flagged_holes:
        result["flagged_holes"] = flagged_holes

    return result


def _enrich_holes_with_confidence(result):
    par = result.get("par", [])
    low_conf_list = result.get("low_confidence_holes", [])
    flagged_list = result.get("flagged_holes", [])

    low_conf_map = {}
    for lc in low_conf_list:
        key = (_normalize_name(lc.get("player", "")), lc.get("hole"))
        low_conf_map[key] = lc

    flagged_map = {}
    for fh in flagged_list:
        key = (_normalize_name(fh.get("player", "")), fh.get("hole"))
        flagged_map[key] = fh

    for player in result.get("players", []):
        name = player.get("name", "?")
        name_norm = _normalize_name(name)
        holes = player.get("holes", [])
        enriched = []
        for i, score in enumerate(holes):
            hole_num = i + 1
            hole_par = par[i] if i < len(par) else None
            lc = low_conf_map.get((name_norm, hole_num))
            fg = flagged_map.get((name_norm, hole_num))

            if lc:
                confidence_level = "low"
                confidence_val = 0.3
            elif fg:
                confidence_level = "low"
                confidence_val = 0.4
            elif score is None:
                confidence_level = "low"
                confidence_val = 0.0
            else:
                confidence_level = "high"
                confidence_val = 0.95

            enriched.append({
                "score": score,
                "confidence_level": confidence_level,
                "confidence": confidence_val,
            })
        player["holes"] = enriched

    return result
