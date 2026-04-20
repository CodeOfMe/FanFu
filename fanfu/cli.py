"""Command-line interface for FanFu."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fanfu import __version__
from fanfu.api import convert_gguf_to_hf, convert_hf_to_gguf, compare_weights

console = Console()


def cmd_gguf_to_hf(args: argparse.Namespace) -> int:
    """Handle gguf-to-hf subcommand."""
    console.print(f"[bold]Converting GGUF to HuggingFace...[/bold]")
    console.print(f"  Input:  {args.gguf}")
    console.print(f"  Output: {args.output}")
    console.print(f"  Type:   {args.outtype}")

    result = convert_gguf_to_hf(
        args.gguf,
        args.output,
        outtype=args.outtype,
        extract_tokenizer=args.extract_tokenizer,
    )

    if result.success:
        console.print(f"\n[green]Conversion successful![/green]")
        console.print(f"  Tensors: {result.data['tensors']}")
        console.print(f"  Skipped: {result.data['skipped']}")
        console.print(f"  Output:  {result.data['output_dir']}")
        return 0
    else:
        console.print(f"\n[red]Conversion failed: {result.error}[/red]")
        return 1


def cmd_hf_to_gguf(args: argparse.Namespace) -> int:
    """Handle hf-to-gguf subcommand."""
    console.print(f"[bold]Converting HuggingFace to GGUF...[/bold]")
    console.print(f"  Input:  {args.hf}")
    console.print(f"  Output: {args.output}")
    console.print(f"  Type:   {args.outtype}")

    result = convert_hf_to_gguf(
        args.hf,
        args.output,
        outtype=args.outtype,
    )

    if result.success:
        console.print(f"\n[green]Conversion successful![/green]")
        console.print(f"  Tensors: {result.data['tensors']}")
        console.print(f"  Output:  {result.data['output_path']}")
        return 0
    else:
        console.print(f"\n[red]Conversion failed: {result.error}[/red]")
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Handle compare subcommand."""
    console.print(f"[bold]Comparing weights...[/bold]")
    console.print(f"  GGUF: {args.gguf}")
    console.print(f"  HF:   {args.hf}")
    console.print(f"  Tolerance: {args.tolerance}")

    result = compare_weights(args.gguf, args.hf, tolerance=args.tolerance)

    if not result.success:
        console.print(f"\n[red]Comparison failed: {result.error}[/red]")
        return 1

    data = result.data
    table = Table(title="Weight Comparison Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Matched", str(data["matched"]))
    table.add_row("Value Mismatched", str(data["mismatched"]))
    table.add_row("Shape Mismatched", str(data["shape_mismatch"]))
    table.add_row("GGUF-only", str(data["gguf_only"]))
    table.add_row("HF-only", str(data["hf_only"]))
    table.add_row("Accuracy", f"{data['accuracy']:.1f}%")

    if data["max_diff_name"]:
        table.add_row("Max Diff", f"{data['max_diff_name']} ({data['max_diff']:.6e})")

    console.print(table)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"\nDetailed results saved to {output_path}")

    return 0


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="fanfu",
        description="FanFu - Bidirectional GGUF/HuggingFace converter with weight verification",
    )
    parser.add_argument("--version", action="version", version=f"fanfu {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # gguf-to-hf
    p_gguf_to_hf = subparsers.add_parser("gguf-to-hf", help="Convert GGUF to HuggingFace format")
    p_gguf_to_hf.add_argument("gguf", help="Path to input GGUF file")
    p_gguf_to_hf.add_argument("-o", "--output", required=True, help="Output directory")
    p_gguf_to_hf.add_argument("-t", "--outtype", default="f32", choices=["f32", "f16", "bf16"], help="Output float type")
    p_gguf_to_hf.add_argument("--no-tokenizer", action="store_true", help="Skip tokenizer extraction")
    p_gguf_to_hf.set_defaults(extract_tokenizer=True, func=cmd_gguf_to_hf)

    # hf-to-gguf
    p_hf_to_gguf = subparsers.add_parser("hf-to-gguf", help="Convert HuggingFace to GGUF format")
    p_hf_to_gguf.add_argument("hf", help="Path to input HF model directory")
    p_hf_to_gguf.add_argument("-o", "--output", required=True, help="Output GGUF file path")
    p_hf_to_gguf.add_argument("-t", "--outtype", default="f32", choices=["f32", "f16", "bf16", "q8_0", "auto"], help="Output quantization type")
    p_hf_to_gguf.set_defaults(func=cmd_hf_to_gguf)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare weights between GGUF and HF models")
    p_compare.add_argument("gguf", help="Path to GGUF file")
    p_compare.add_argument("hf", help="Path to HF model directory")
    p_compare.add_argument("--tolerance", type=float, default=0.5, help="Comparison tolerance")
    p_compare.add_argument("-o", "--output", help="Save detailed results to JSON file")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "extract_tokenizer") and args.no_tokenizer:
        args.extract_tokenizer = False

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
