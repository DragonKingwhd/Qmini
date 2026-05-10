"""Create a lighter Qmini USD by referencing heavy link child subtrees.

The output keeps the robot physics and joint hierarchy local, while each link's
``visuals`` and ``collisions`` child prims become references back to the source
USD. The source file is not modified.
"""

from __future__ import annotations

import argparse
import os

from pxr import Sdf, Usd


ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "Qmini", "Qmini", "assets"))
DEFAULT_SOURCE_USD = os.path.join(ASSETS_DIR, "Qmini.usd")
DEFAULT_OUTPUT_USD = os.path.join(ASSETS_DIR, "Qmini_ref.usda")
TARGET_CHILD_NAMES = {"visuals", "collisions"}


def _relative_asset_path(source_path: str, output_path: str) -> str:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    return os.path.relpath(os.path.abspath(source_path), output_dir)


def _get_robot_root(stage: Usd.Stage, robot_root: str | None) -> str:
    if robot_root:
        return robot_root

    default_prim = stage.GetDefaultPrim()
    if default_prim.IsValid():
        return default_prim.GetPath().pathString

    for prim in stage.GetPseudoRoot().GetChildren():
        if any(_is_link_prim(child) for child in prim.GetChildren()):
            return prim.GetPath().pathString

    raise RuntimeError("Could not infer robot root. Pass --robot-root explicitly.")


def _is_link_prim(prim: Usd.Prim) -> bool:
    if not prim.IsValid():
        return False
    return prim.GetTypeName() == "Xform" and "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas()


def convert_to_references(source_usd: str, output_usd: str, robot_root: str | None = None) -> tuple[str, list[str]]:
    source_usd = os.path.abspath(source_usd)
    output_usd = os.path.abspath(output_usd)

    if source_usd == output_usd:
        raise RuntimeError("SOURCE_USD and OUTPUT_USD must be different files.")
    if not os.path.exists(source_usd):
        raise RuntimeError(f"Source file does not exist: {source_usd}")

    source_stage = Usd.Stage.Open(source_usd)
    if source_stage is None:
        raise RuntimeError(f"Failed to open source stage: {source_usd}")

    robot_root = _get_robot_root(source_stage, robot_root)
    if not source_stage.GetPrimAtPath(robot_root).IsValid():
        raise RuntimeError(f"Robot root not found in source: {robot_root}")

    os.makedirs(os.path.dirname(output_usd), exist_ok=True)
    if not source_stage.Export(output_usd):
        raise RuntimeError(f"Failed to export initial output stage: {output_usd}")

    stage = Usd.Stage.Open(output_usd)
    if stage is None:
        raise RuntimeError(f"Failed to open output stage: {output_usd}")

    robot_prim = stage.GetPrimAtPath(robot_root)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot root not found in output: {robot_root}")

    reference_asset = _relative_asset_path(source_usd, output_usd)
    replaced: list[str] = []

    link_prims = [child for child in robot_prim.GetChildren() if _is_link_prim(child)]
    with Sdf.ChangeBlock():
        for link_prim in link_prims:
            for child in list(link_prim.GetChildren()):
                if child.GetName() not in TARGET_CHILD_NAMES:
                    continue

                child_path = child.GetPath()
                child_type = child.GetTypeName() or "Xform"

                stage.RemovePrim(str(child_path))
                new_child = stage.DefinePrim(str(child_path), child_type)
                if not new_child.IsValid():
                    raise RuntimeError(f"DefinePrim failed: {child_path}")

                new_child.GetReferences().AddReference(reference_asset, str(child_path))
                new_child.SetInstanceable(True)
                replaced.append(str(child_path))

    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to save: {output_usd}")

    return reference_asset, replaced


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qmini link visuals/collisions to referenced subtrees.")
    parser.add_argument("--source", default=DEFAULT_SOURCE_USD, help="Source USD file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_USD, help="Output USD/USDA file.")
    parser.add_argument("--robot-root", default=None, help="Robot root prim path. Defaults to the source defaultPrim.")
    args = parser.parse_args()

    reference_asset, replaced = convert_to_references(args.source, args.output, args.robot_root)

    print("Source:", os.path.abspath(args.source))
    print("Output:", os.path.abspath(args.output))
    print("Reference asset path:", reference_asset)
    print("Replaced", len(replaced), "subtrees with references")
    for path in replaced:
        print("  ", path)
    print("Done")


if __name__ == "__main__":
    main()
