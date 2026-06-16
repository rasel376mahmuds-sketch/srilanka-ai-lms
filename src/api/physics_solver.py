import re

def solve_kinematics_s(text: str, lang: str = "en") -> str:
    """
    Extracts u, a, and t from text (both strict syntax and natural language)
    and calculates s = ut + 0.5*a*t^2.
    """
    # Initialize variables
    u, a, t = None, None, None

    # 1. Natural Language Heuristics
    text_lower = text.lower()
    
    # Heuristic for u
    if "from rest" in text_lower:
        u = 0.0
    else:
        match_u = re.search(r"initial velocity of (\d+(?:\.\d+)?)", text_lower)
        if match_u:
            u = float(match_u.group(1))

    # Heuristic for a
    match_a = re.search(r"accelerates.*?(\d+(?:\.\d+)?)\s*(?:m/s2|m/s\^2|m/s²)", text_lower)
    if not match_a:
        match_a = re.search(r"acceleration of (\d+(?:\.\d+)?)", text_lower)
    if not match_a:
        match_a = re.search(r"rate of (\d+(?:\.\d+)?)", text_lower)
    if match_a:
        a = float(match_a.group(1))

    # Heuristic for t
    match_t = re.search(r"for (\d+(?:\.\d+)?)\s*sec", text_lower)
    if not match_t:
        match_t = re.search(r"time of (\d+(?:\.\d+)?)", text_lower)
    if match_t:
        t = float(match_t.group(1))

    # 2. Strict Syntax Fallback (u=0, a=5, t=10)
    def extract_strict(var_name, string):
        pattern = rf"\b{var_name}\s*[=:]\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    if u is None: u = extract_strict('u', text)
    if a is None: a = extract_strict('a', text)
    if t is None: t = extract_strict('t', text)
    
    if u is None or a is None or t is None:
        missing = []
        if u is None: missing.append("u (initial velocity)")
        if a is None: missing.append("a (acceleration)")
        if t is None: missing.append("t (time)")
        missing_str = ", ".join(missing)
        
        if lang == "si":
            return f"මට ගැටළුව තේරුම් ගත නොහැක. කරුණාකර පහත අගයන් පැහැදිලිව දක්වන්න: {missing_str}."
        elif lang == "ta":
            return f"என்னை மன்னிக்கவும், எனக்கு புரியவில்லை. இந்த மதிப்புகளை தெளிவாகக் கொடுங்கள்: {missing_str}."
        else:
            return f"I couldn't fully understand the problem. I am missing values for: {missing_str}. Please provide them clearly (e.g., u=0 a=3 t=4)."
            
    # Calculate s = ut + 1/2 at^2
    s = (u * t) + (0.5 * a * (t ** 2))
    
    # Format response
    if lang == "si":
        return f"<b>ගැටළුව හඳුනා ගන්නා ලදී:</b><br>ආරම්භක ප්‍රවේගය (u) = {u} m/s<br>ත්වරණය (a) = {a} m/s²<br>කාලය (t) = {t} s<br><br>ගණනය කිරීම: s = ({u} * {t}) + 0.5 * {a} * ({t}²)<br><br><b>පිළිතුර: විස්ථාපනය (s) = {round(s, 2)} m</b>"
    elif lang == "ta":
        return f"<b>கணக்கு கண்டறியப்பட்டது:</b><br>ஆரம்ப திசைவேகம் (u) = {u} m/s<br>முடுக்கம் (a) = {a} m/s²<br>நேரம் (t) = {t} s<br><br>கணக்கீடு: s = ({u} * {t}) + 0.5 * {a} * ({t}²)<br><br><b>விடை: இடப்பெயர்ச்சி (s) = {round(s, 2)} m</b>"
    else:
        return f"<b>Problem Understood:</b><br>Initial Velocity (u) = {u} m/s<br>Acceleration (a) = {a} m/s²<br>Time (t) = {t} s<br><br>Calculation: s = ({u} * {t}) + 0.5 * {a} * ({t}²)<br><br><b>Answer: Total Distance Traveled (s) = {round(s, 2)} m</b>"
