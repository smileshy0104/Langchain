#!/usr/bin/env python3
"""
SQL to GORM Conversion Skill
Interactive wrapper for sql_to_gorm.py with natural language support
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Import the core conversion logic
sys.path.append(str(Path(__file__).parent))
from sql_to_gorm import parse_sql_string, parse_multiple_tables

class SQLToGORMSkill:
    """Main skill class for SQL to GORM conversion"""

    def __init__(self):
        self.sql_content = ""
        self.output_dir = None
        self.package_name = "models"
        self.multiple_tables = False

    def extract_sql_from_text(self, text: str) -> List[str]:
        """Extract SQL CREATE TABLE statements from text"""
        # Find CREATE TABLE statements
        pattern = re.compile(
            r'CREATE\s+TABLE[^;]*?(?:ENGINE[^;]*)?;',
            re.IGNORECASE | re.DOTALL
        )

        matches = pattern.findall(text)
        return matches

    def find_sql_file(self, prompt: str) -> Optional[str]:
        """Try to find SQL file path from user prompt"""
        # Look for file paths in common patterns
        patterns = [
            r'(?:file|path|directory)?\s*[:\s]+[\'"]?([^\'"\s]+\.sql)[\'"]?',
            r'(?:in|at|from)\s+[\'"]?([^\'"\s]+\.sql)[\'"]?',
            r'([^\s]+\.sql)',
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                file_path = match.group(1)
                # Try common locations
                for base in ['', './', '../', '~/']:
                    test_path = os.path.expanduser(os.path.join(base, file_path))
                    if os.path.exists(test_path):
                        return test_path

        return None

    def parse_user_intent(self, prompt: str) -> Dict:
        """Parse user intent from natural language prompt"""
        intent = {
            'has_sql': False,
            'sql_file': None,
            'output_dir': None,
            'package': None,
            'multiple': False
        }

        # Check for SQL statements in prompt
        sql_matches = self.extract_sql_from_text(prompt)
        if sql_matches:
            intent['has_sql'] = True
            intent['sql_content'] = '\n'.join(sql_matches)
            intent['multiple'] = len(sql_matches) > 1

        # Look for file paths
        sql_file = self.find_sql_file(prompt)
        if sql_file:
            intent['sql_file'] = sql_file

        # Extract output directory
        output_patterns = [
            r'(?:save|output|generate|create)\s+(?:to|in)\s+[\'"]?([^\'"\s]+)[\'"]?',
            r'(?:directory|folder|path)\s*[=:]\s*[\'"]?([^\'"\s]+)[\'"]?',
        ]

        for pattern in output_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                intent['output_dir'] = match.group(1)
                break

        # Extract package name
        package_patterns = [
            r'package\s+[\'"]?(\w+)[\'"]?',
            r'with\s+package\s+[\'"]?(\w+)[\'"]?',
        ]

        for pattern in package_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                intent['package'] = match.group(1)
                break

        # Check for multiple table keywords
        multiple_keywords = ['all tables', 'multiple tables', 'each table', 'separate files']
        if any(keyword in prompt.lower() for keyword in multiple_keywords):
            intent['multiple'] = True

        return intent

    def load_sql_from_file(self, file_path: str) -> str:
        """Load SQL content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read SQL file: {e}")

    def generate_models(self, sql: str) -> Tuple[str, Dict]:
        """Generate Go models from SQL"""
        # Try parsing as multiple tables first
        models = parse_multiple_tables(sql, self.package_name)

        if len(models) == 0:
            # Try single table parsing
            go_code = parse_sql_string(sql, self.package_name)
            if go_code:
                return go_code, {}
            else:
                raise Exception("No valid CREATE TABLE statements found")
        elif len(models) == 1 and not self.multiple_tables:
            # Single table mode
            table_name = list(models.keys())[0]
            return models[table_name]['go_code'], models
        else:
            # Multiple tables
            return "", models

    def save_models(self, models: Dict, output_dir: str) -> List[str]:
        """Save multiple models to separate files"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for table_name, data in models.items():
            struct_name = data['struct_name']
            go_code = data['go_code']
            filename = f"{struct_name}.go"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(go_code)
                saved_files.append(filepath)
            except Exception as e:
                raise Exception(f"Failed to save {filename}: {e}")

        return saved_files

    def process(self, prompt: str) -> Dict:
        """Process user prompt and generate models"""
        result = {
            'success': False,
            'message': '',
            'go_code': '',
            'files': [],
            'count': 0
        }

        # Parse user intent
        intent = self.parse_user_intent(prompt)

        # Set configuration
        self.output_dir = intent.get('output_dir')
        self.package_name = intent.get('package', 'models')
        self.multiple_tables = intent.get('multiple', False)

        # Get SQL content
        if intent.get('sql_content'):
            self.sql_content = intent['sql_content']
        elif intent.get('sql_file'):
            try:
                self.sql_content = self.load_sql_from_file(intent['sql_file'])
                result['message'] = f"Loaded SQL from: {intent['sql_file']}"
            except Exception as e:
                result['message'] = f"Error loading file: {e}"
                return result
        else:
            # Try to find SQL in the original prompt
            sql_matches = self.extract_sql_from_text(prompt)
            if sql_matches:
                self.sql_content = '\n'.join(sql_matches)
            else:
                result['message'] = "Please provide SQL statements or file path"
                return result

        # Generate models
        try:
            go_code, models = self.generate_models(self.sql_content)

            if self.output_dir and models:
                # Save multiple files
                saved_files = self.save_models(models, self.output_dir)
                result['success'] = True
                result['files'] = saved_files
                result['count'] = len(saved_files)
                result['message'] = f"Successfully generated {len(saved_files)} model files in {self.output_dir}"
            elif go_code:
                # Single model output
                result['success'] = True
                result['go_code'] = go_code
                result['count'] = 1
                result['message'] = "Successfully generated Go model"

        except Exception as e:
            result['message'] = f"Error generating models: {e}"

        return result


# Command line interface for testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sql_to_gorm_skill.py \"your prompt here\"")
        print("\nExample prompts:")
        print('  "Convert this SQL: CREATE TABLE users (id int primary key)"')
        print('  "Convert schema.sql to Go models in ./models"')
        sys.exit(1)

    skill = SQLToGORMSkill()
    prompt = " ".join(sys.argv[1:])
    result = skill.process(prompt)

    if result['success']:
        print(f"✓ {result['message']}")
        if result['go_code']:
            print("\nGenerated Code:")
            print("-" * 40)
            print(result['go_code'])
        if result['files']:
            print("\nGenerated Files:")
            for file_path in result['files']:
                print(f"  - {file_path}")
    else:
        print(f"✗ {result['message']}")