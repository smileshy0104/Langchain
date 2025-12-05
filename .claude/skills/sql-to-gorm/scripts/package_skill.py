#!/usr/bin/env python3
"""Packaging script for sql-to-gorm skill distribution"""

import os
import sys
import zipfile
import yaml
from pathlib import Path

def validate_skill(skill_path):
    """Validate the skill structure and content"""
    errors = []

    # Check required files
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        errors.append("Missing SKILL.md file")
        return errors

    # Parse and validate frontmatter
    with open(skill_md, 'r') as f:
        content = f.read()
        if not content.startswith('---'):
            errors.append("SKILL.md must start with YAML frontmatter")
            return errors

        try:
            # Extract frontmatter
            parts = content.split('---', 2)
            if len(parts) < 3:
                errors.append("Invalid YAML frontmatter format")
                return errors

            frontmatter = yaml.safe_load(parts[1])

            # Check required fields
            if not frontmatter.get('name'):
                errors.append("Missing 'name' in frontmatter")
            if not frontmatter.get('description'):
                errors.append("Missing 'description' in frontmatter")

            # Validate description quality
            desc = frontmatter.get('description', '')
            if len(desc) < 50:
                errors.append("Description should be at least 50 characters")
            if 'This skill should be used when' not in desc:
                errors.append("Description should explain when to use the skill")

        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML in frontmatter: {e}")

    # Check directory structure
    for dir_name in ['scripts', 'references', 'assets']:
        dir_path = skill_path / dir_name
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix == '.py':
                    # Check Python script syntax
                    try:
                        with open(item, 'r') as f:
                            compile(f.read(), item, 'exec')
                    except SyntaxError as e:
                        errors.append(f"Syntax error in {item}: {e}")

    return errors

def package_skill(skill_path, output_dir=None):
    """Package the skill into a distributable zip file"""
    skill_path = Path(skill_path).resolve()

    if output_dir is None:
        output_dir = skill_path.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Validate skill
    errors = validate_skill(skill_path)
    if errors:
        print("✗ Skill validation failed:")
        for error in errors:
            print(f"  - {error}")
        return None

    print("✓ Skill validation passed")

    # Get skill name from SKILL.md
    skill_md = skill_path / "SKILL.md"
    with open(skill_md, 'r') as f:
        content = f.read()
        parts = content.split('---', 2)
        frontmatter = yaml.safe_load(parts[1])
        skill_name = frontmatter.get('name', 'skill').replace(' ', '-').lower()

    # Create zip file
    zip_path = output_dir / f"{skill_name}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in skill_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path
                arcname = file_path.relative_to(skill_path)
                zf.write(file_path, arcname)

    print(f"✓ Skill packaged: {zip_path}")
    return zip_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python package_skill.py <skill-path> [output-dir]")
        sys.exit(1)

    skill_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    package_skill(skill_path, output_dir)