import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from app.router.LLMClient import LLMClient
from app.util.logger import Logger


SYSTEM_PROMPT = """You are optimizing train.py to minimize val_bpb.

Return ONLY JSON with this schema:
{
  "description": "short experiment description",
  "edits": [
    {"find": "exact text in train.py", "replace": "new text"}
  ]
}

Rules:
- Edit ONLY hyperparameter constants under the '# Hyperparameters' section.
- Use 1 to 4 edits.
- Each `find` must be exact text from the file and should appear once.
- No markdown, no explanations outside JSON.
"""


class ResearchAgent:
    def __init__(self, repo_root: Path, llm: LLMClient, logger: Logger):
        self.repo_root = repo_root
        self.llm = llm
        self.logger = logger
        self.train_path = repo_root / "train.py"
        self.results_path = repo_root / "results.tsv"
        self.run_log_path = repo_root / "run.log"

    def ensure_results_file(self) -> None:
        if self.results_path.exists():
            return
        self.results_path.write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
            encoding="utf-8",
        )

    def read_best_kept_bpb(self) -> float | None:
        if not self.results_path.exists():
            return None
        best = None
        with self.results_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") != "keep":
                    continue
                try:
                    value = float(row["val_bpb"])
                except (TypeError, ValueError):
                    continue
                if best is None or value < best:
                    best = value
        return best

    def extract_hparams_block(self, text: str) -> str:
        pattern = r"# Hyperparameters.*?# ---------------------------------------------------------------------------\n# Setup:"
        m = re.search(pattern, text, flags=re.S)
        if not m:
            raise RuntimeError("Could not locate Hyperparameters block in train.py")
        block = m.group(0)
        return block.removesuffix("# Setup:")

    def _extract_assistant_text(self, response: dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {response}") from e

    def _parse_json_payload(self, text: str) -> dict[str, Any]:
        candidate = text.strip()
        candidate = candidate.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", candidate, flags=re.S)
            if not m:
                raise
            return json.loads(m.group(0))

    def propose_experiment(self, train_text: str, best_bpb: float | None) -> dict[str, Any]:
        hparams_block = self.extract_hparams_block(train_text)
        user_prompt = (
            f"Current best val_bpb: {best_bpb if best_bpb is not None else 'N/A (baseline not run yet)'}\n\n"
            "Current train.py hyperparameters block:\n"
            f"{hparams_block}\n"
        )
        response = self.llm.discuss(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = self._extract_assistant_text(response)
        payload = self._parse_json_payload(text)

        if "description" not in payload or "edits" not in payload:
            raise RuntimeError(f"Invalid LLM payload keys: {payload}")
        if not isinstance(payload["edits"], list) or not payload["edits"]:
            raise RuntimeError(f"LLM returned empty edits: {payload}")
        return payload

    def apply_edits(self, original: str, edits: list[dict[str, str]]) -> str:
        updated = original
        for i, edit in enumerate(edits, start=1):
            find = edit.get("find")
            replace = edit.get("replace")
            if not isinstance(find, str) or not isinstance(replace, str):
                raise RuntimeError(f"Edit #{i} is invalid: {edit}")
            if find not in updated:
                raise RuntimeError(f"Edit #{i} find-string not found: {find!r}")
            updated = updated.replace(find, replace, 1)
        return updated

    def _git_short_hash(self) -> str:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.repo_root,
                text=True,
            ).strip()
            return out or "local"
        except Exception:
            return "local"

    def _append_result(self, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
        commit = self._git_short_hash()
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")

    def run_training(self, timeout_sec: int = 700) -> tuple[bool, float, float, str]:
        cmd = ["python3", "-u", "train.py"]
        # using system python3 (Orin CUDA compatible)




        with self.run_log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        if proc.returncode != 0:
            log_tail = self.run_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            reason = f"train exited with code {proc.returncode}"
            if log_tail:
                reason += " | tail: " + " | ".join(log_tail)
            return False, 0.0, 0.0, reason

        log_text = self.run_log_path.read_text(encoding="utf-8", errors="replace")
        val_match = re.search(r"^val_bpb:\s*([0-9.]+)", log_text, flags=re.M)
        mem_match = re.search(r"^peak_vram_mb:\s*([0-9.]+)", log_text, flags=re.M)
        if not val_match:
            log_tail = log_text.splitlines()[-20:]
            reason = "train finished but no val_bpb found"
            if log_tail:
                reason += " | tail: " + " | ".join(log_tail)
            return False, 0.0, 0.0, reason
        val_bpb = float(val_match.group(1))
        peak_vram_mb = float(mem_match.group(1)) if mem_match else 0.0
        memory_gb = peak_vram_mb / 1024.0
        return True, val_bpb, memory_gb, "ok"

    def run(self, iterations: int) -> None:
        self.ensure_results_file()
        best_bpb = self.read_best_kept_bpb()
        self.logger.info(f"Starting research loop, iterations={iterations}, best={best_bpb}")
        print(f"[ResearchAgent] start iterations={iterations} best={best_bpb}")

        for i in range(1, iterations + 1):
            self.logger.info(f"[iter={i}] proposing experiment")
            print(f"[ResearchAgent] iter={i} proposing experiment...")
            original_text = self.train_path.read_text(encoding="utf-8")
            plan = self.propose_experiment(original_text, best_bpb)
            description = str(plan["description"]).replace("\t", " ").strip()
            updated_text = self.apply_edits(original_text, plan["edits"])
            self.train_path.write_text(updated_text, encoding="utf-8")
            print(f"[ResearchAgent] iter={i} running train.py ...")

            try:
                ok, val_bpb, memory_gb, reason = self.run_training()
            except subprocess.TimeoutExpired:
                ok, val_bpb, memory_gb, reason = False, 0.0, 0.0, "train timed out"

            if not ok:
                self.train_path.write_text(original_text, encoding="utf-8")
                self._append_result(0.0, 0.0, "crash", description)
                self.logger.error(f"[iter={i}] crash: {description} | {reason}")
                print(f"[ResearchAgent] iter={i} crash: {description}")
                print(f"[ResearchAgent] iter={i} reason: {reason}")
                continue

            improved = best_bpb is None or val_bpb < best_bpb
            if improved:
                best_bpb = val_bpb
                self._append_result(val_bpb, memory_gb, "keep", description)
                self.logger.info(f"[iter={i}] keep val_bpb={val_bpb:.6f} desc={description}")
                print(f"[ResearchAgent] iter={i} KEEP val_bpb={val_bpb:.6f} mem={memory_gb:.1f}GB")
            else:
                self.train_path.write_text(original_text, encoding="utf-8")
                self._append_result(val_bpb, memory_gb, "discard", description)
                self.logger.info(f"[iter={i}] discard val_bpb={val_bpb:.6f} desc={description}")
                print(f"[ResearchAgent] iter={i} DISCARD val_bpb={val_bpb:.6f} mem={memory_gb:.1f}GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoresearch loop using app.router.LLMClient")
    parser.add_argument("--iterations", type=int, default=1, help="How many experiments to run")
    args = parser.parse_args()

    logger = Logger("ResearchAgent")
    llm = LLMClient(logger=logger)
    agent = ResearchAgent(repo_root=Path(__file__).resolve().parents[2], llm=llm, logger=logger)
    agent.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
