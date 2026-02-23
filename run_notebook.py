#!/usr/bin/env python
"""Execute notebook cells and capture errors"""

import json
import sys
import traceback
import os

# Add parent directory to path to import autolyse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_notebook(notebook_path):
    """Execute all code cells in a notebook"""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print(f"{'='*60}\n")
    
    cell_count = 0
    error_count = 0
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            cell_count += 1
            source = ''.join(cell['source'])
            
            # Skip commented out cells
            if source.strip().startswith('#'):
                print(f"⏭️  Cell {cell_count}: Skipped (commented)")
                continue
            
            print(f"\n📝 Cell {cell_count}:")
            print(f"{'─'*50}")
            print(source[:200] + ('...' if len(source) > 200 else ''))
            print(f"{'─'*50}")
            
            try:
                exec(source, globals())
                print(f"✅ Cell {cell_count}: Success")
            except Exception as e:
                error_count += 1
                print(f"❌ Cell {cell_count}: ERROR")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {str(e)}")
                traceback.print_exc()
                print()
    
    print(f"\n{'='*60}")
    print(f"Summary: {cell_count} cells, {error_count} errors")
    print(f"{'='*60}\n")
    
    return error_count == 0

if __name__ == "__main__":
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "examples/tutorial.ipynb"
    success = run_notebook(notebook_path)
    sys.exit(0 if success else 1)
