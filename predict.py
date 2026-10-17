"""
Single-Molecule Command-Line Predictor
=====================================
Usage:
  python predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
  python predict.py --name "caffeine"
"""

import argparse
import sys
from demo import simulate_neural_inference, evaluate_lipinski, PRESET_MOLECULES, BOLD, GREEN, CYAN, RESET, YELLOW, RED


def main():
    parser = argparse.ArgumentParser(description="Quick Molecular Property Predictor")
    parser.add_argument("--smiles", type=str, help="SMILES representation of the molecule")
    parser.add_argument("--name", type=str, help="Common name (e.g. aspirin, caffeine, paracetamol)")
    args = parser.parse_args()

    smiles = args.smiles
    name = "Target Molecule"

    if args.name:
        for k, v in PRESET_MOLECULES.items():
            if v[0].lower() == args.name.lower():
                name = v[0]
                smiles = v[1]
                break

    if not smiles:
        print("Usage: python predict.py --smiles <SMILES> OR python predict.py --name <drug_name>")
        print("Example: python predict.py --name aspirin")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}=== MolPredict Neural Inference Result ==={RESET}")
    print(f"Molecule : {YELLOW}{name}{RESET}")
    print(f"SMILES   : {smiles}")

    res = simulate_neural_inference(smiles)
    print(f"\n{BOLD}Predictions:{RESET}")
    print(f"  • LogP             : {GREEN}{res['pred_logp']}{RESET}")
    print(f"  • Molecular Weight : {res['mw']} g/mol")
    print(f"  • Polar Surf Area  : {res['tpsa']} Å²")
    print(f"  • Rotatable Bonds  : {res['rotatable_bonds']}")

    rules, violations = evaluate_lipinski(res["mw"], res["pred_logp"], res["hbd"], res["hba"])
    status = f"{GREEN}PASS (0 violations){RESET}" if violations == 0 else f"{YELLOW}{violations} violation(s){RESET}"
    print(f"  • Drug-likeness    : {status}\n")


if __name__ == "__main__":
    main()
