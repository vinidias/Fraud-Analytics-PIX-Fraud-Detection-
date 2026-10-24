import io
path = r"D:/Dev/fraud/scripts/ensemble_gpu.py"
with io.open(path, "r", encoding="utf-8") as f:
    src = f.read()
# Remove wrapper se existir
if src.startswith("<path>"):
    start = src.find("<content>")
    end = src.find("</content>")
    if start != -1 and end != -1:
        src = src[start+len("<content>"):end].strip("\n")
# Aplicar substituições ASCII
src = src.replace("✓ ", "[OK] ").replace(" · ", " | ").replace("→", "->")
with io.open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("OK, primeiros 80 chars:", repr(src[:80]))
