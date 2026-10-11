"""Check syntax, source provenance and renamed model ASTs without PyTorch."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    "BPMPNet": "Net", "MCAB": "block1", "SKAFE": "PanExt",
    "SCWAB": "SCAttention", "CrossModalAttention": "CrossSwinTransformer_1",
    "ShuffleSelectiveKernelAttention": "MSKAattention", "ResidualBlock": "ResnetBlock",
    "ParallelMultiscaleConv": "four_conv", "SerialMultiscaleConv": "five_conv",
    "FeatureProjection": "convert_conv",
}
LOCAL_NAMES = {
    "MCAB": {"attention_features": "f1", "parallel1": "f11", "projected1": "f2",
             "serial1": "f21", "projected2": "f3", "parallel2": "f12",
             "projected3": "f4", "serial2": "f22", "fused_features": "f5", "pan_details": "hp_pan"},
    "BPMPNet": {"lrms": "l_ms", "pan": "x_pan", "ms_half": "ms_b1",
                "fusion_half": "f1", "guided_half": "f2", "fusion_full_seed": "f3",
                "fusion_full": "f4", "guided_full": "f5", "full_concat": "f6",
                "full_refined": "f7", "auxiliary_hrms": "hrms1", "final_hrms": "hrms2"},
    "SKAFE": {"recovery": "x1", "recovery_up": "x11", "branch3": "x21",
              "branch5": "x22", "branch7": "x23", "branch9": "x24", "branch13": "x25",
              "selected3": "x31", "selected5": "x32", "selected7": "x33",
              "selected9": "x34", "selected13": "x35", "projected3": "x41",
              "projected5": "x42", "projected7": "x43", "projected9": "x44", "projected13": "x45"},
    "SCWAB": {"spatial1": "x1", "channel1": "x2", "spatial2": "x3", "channel2": "x4"},
}


class Normalize(ast.NodeTransformer):
    """Undo only the documented identifier changes and discard docstrings."""

    def __init__(self):
        self.names = CLASS_NAMES

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.names.get(node.arg, node.arg)
        return node

    def visit_ClassDef(self, node):
        original_name = node.name
        node.name = CLASS_NAMES.get(node.name, node.name)
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and method.name == "forward":
                self.names = {**CLASS_NAMES, **LOCAL_NAMES.get(original_name, {})}
                self.visit(method)
                self.names = CLASS_NAMES
            else:
                self.visit(method)
        self.strip_docstring(node)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        self.strip_docstring(node)
        # An unused SciPy import was removed from the cleaned initializer.
        node.body = [statement for statement in node.body
                     if not (isinstance(statement, ast.ImportFrom) and statement.module == "scipy.stats")]
        return node

    @staticmethod
    def strip_docstring(node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)


def definitions(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            normalized = Normalize().visit(node)
            result[normalized.name] = ast.dump(normalized, include_attributes=False)
    return result


def main():
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        compile(path.read_bytes(), str(path), "exec")
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((ROOT / "legacy/prj3113" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Historical source changed: {name}")
    mapping = {
        "network.py": "Net2.py", "window_attention.py": "CAB.py",
        "scwab.py": "MSA.py", "skafe.py": "SCExt.py", "common.py": "base_net.py",
    }
    checked = 0
    for clean, legacy in mapping.items():
        old = definitions(ROOT / "legacy/prj3113" / legacy)
        new = definitions(ROOT / "bpmp/models" / clean)
        for name, tree in new.items():
            if old.get(name) != tree:
                raise AssertionError(f"Unexpected model computation change: {clean}:{name}")
            checked += 1
    for script in ("train.py", "predict.py", "scripts/convert_checkpoint.py"):
        subprocess.run([sys.executable, str(ROOT / script), "--help"], check=True, stdout=subprocess.DEVNULL)
    print(f"PASS: {len(files)} Python files compile; 9 pristine sources match SHA256; {checked} model definitions retain their AST; CLI help works")


if __name__ == "__main__":
    main()
