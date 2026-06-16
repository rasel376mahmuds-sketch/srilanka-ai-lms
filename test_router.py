text = "5m trave in 2 seconds what is velocity"
text_lower = text.lower()
math_keywords = ["solve", "calculate", "find", "value", "velocity", "acceleration", "distance", "mass", "force", "energy", "equation"]
has_math_intent = any(kw in text_lower for kw in math_keywords) and any(char.isdigit() for char in text_lower)

print("Text:", text_lower)
print("Has Math Intent:", has_math_intent)
