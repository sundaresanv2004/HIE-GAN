#!/usr/bin/env python3
"""
Temporary Dataset Structure Checker
Checks which object folders have 'images/' subdirectory and reports missing ones.
Creates a temporary report file which can be reviewed and deleted.
"""

import os
from pathlib import Path
import argparse


def check_dataset_structure(root_dir, create_report=True):
    """
    Check dataset structure and identify objects missing 'images' folder.
    
    Args:
        root_dir: Root directory of dataset (e.g., ShapeNetCore_V5/train)
        create_report: Whether to create a temporary report file
    
    Returns:
        dict: Statistics about the dataset
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ Directory not found: {root_dir}")
        return None
    
    # Class IDs we're interested in
    class_ids = [
        "02691156",  # Airplane
        "02958343",  # Car
        "03001627",  # Chair
        "03636649",  # Lamp
        "04379243",  # Table
    ]
    
    class_names = {
        "02691156": "Airplane",
        "02958343": "Car",
        "03001627": "Chair",
        "03636649": "Lamp",
        "04379243": "Table",
    }
    
    stats = {}
    all_issues = []
    
    print("=" * 70)
    print(f"📊 Checking Dataset Structure: {root_dir}")
    print("=" * 70)
    
    for class_id in class_ids:
        class_dir = root_path / class_id
        
        if not class_dir.exists():
            print(f"⚠️  Class directory not found: {class_id} ({class_names.get(class_id, 'Unknown')})")
            continue
        
        total_objects = 0
        missing_images = []
        missing_ply = []
        valid_objects = []
        
        # Check each object folder
        for obj_folder in class_dir.iterdir():
            if not obj_folder.is_dir():
                continue
            
            total_objects += 1
            obj_name = obj_folder.name
            
            has_images = (obj_folder / "images").exists()
            has_ply = (obj_folder / "model_normalized.ply").exists()
            
            if not has_images:
                missing_images.append(obj_name)
            
            if not has_ply:
                missing_ply.append(obj_name)
            
            if has_images and has_ply:
                valid_objects.append(obj_name)
        
        # Store stats
        stats[class_id] = {
            "name": class_names.get(class_id, "Unknown"),
            "total": total_objects,
            "valid": len(valid_objects),
            "missing_images": len(missing_images),
            "missing_ply": len(missing_ply),
        }
        
        # Print summary for this class
        print(f"\n📁 Class: {class_id} ({class_names.get(class_id, 'Unknown')})")
        print(f"   Total objects: {total_objects}")
        print(f"   ✅ Valid objects (has images + ply): {len(valid_objects)}")
        print(f"   ⚠️  Missing 'images' folder: {len(missing_images)}")
        print(f"   ⚠️  Missing 'model_normalized.ply': {len(missing_ply)}")
        
        if missing_images:
            all_issues.append({
                "class_id": class_id,
                "class_name": class_names.get(class_id, "Unknown"),
                "issue_type": "missing_images",
                "objects": missing_images[:10]  # First 10 for report
            })
        
        if missing_ply:
            all_issues.append({
                "class_id": class_id,
                "class_name": class_names.get(class_id, "Unknown"),
                "issue_type": "missing_ply",
                "objects": missing_ply[:10]
            })
    
    print("\n" + "=" * 70)
    print("📊 Overall Summary")
    print("=" * 70)
    
    total_valid = sum(s["valid"] for s in stats.values())
    total_objects = sum(s["total"] for s in stats.values())
    total_missing_images = sum(s["missing_images"] for s in stats.values())
    
    print(f"Total objects scanned: {total_objects}")
    print(f"✅ Valid objects: {total_valid}")
    print(f"⚠️  Objects missing 'images': {total_missing_images}")
    print(f"📊 Usable dataset: {total_valid}/{total_objects} ({100*total_valid/total_objects if total_objects > 0 else 0:.1f}%)")
    
    # Create temporary report file
    if create_report and all_issues:
        report_path = Path("dataset_issues_TEMP.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Dataset Structure Report: {root_dir}\n")
            f.write("=" * 70 + "\n\n")
            
            for issue in all_issues:
                f.write(f"\nClass: {issue['class_id']} ({issue['class_name']})\n")
                f.write(f"Issue: {issue['issue_type']}\n")
                f.write(f"Sample affected objects (first 10):\n")
                for obj in issue['objects']:
                    f.write(f"  - {obj}\n")
                f.write("\n" + "-" * 70 + "\n")
            
            f.write(f"\nSummary:\n")
            f.write(f"Total objects: {total_objects}\n")
            f.write(f"Valid objects: {total_valid}\n")
            f.write(f"Missing images: {total_missing_images}\n")
        
        print(f"\n📄 Temporary report created: {report_path}")
        print(f"💡 Review the file, then delete it when done:")
        print(f"   rm {report_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Check dataset structure for missing 'images' folders")
    parser.add_argument(
        "--data-root", 
        type=str, 
        required=True,
        help="Path to dataset root (e.g., data/ShapeNetCore_V5/train)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't create temporary report file"
    )
    parser.add_argument(
        "--delete-report",
        action="store_true",
        help="Delete the temporary report file after showing results"
    )
    
    args = parser.parse_args()
    
    stats = check_dataset_structure(args.data_root, create_report=not args.no_report)
    
    # Auto-delete report if requested
    if args.delete_report and not args.no_report:
        report_path = Path("dataset_issues_TEMP.txt")
        if report_path.exists():
            report_path.unlink()
            print(f"\n🗑️  Deleted temporary report: {report_path}")


if __name__ == "__main__":
    main()
