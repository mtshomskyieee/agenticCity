#!/usr/bin/env python3
import json
import shutil
from datetime import datetime

def load_history(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {filename}: {e}")
        return []

def save_history(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def compare_data(entry1, entry2):
    """Compare the data portion of two history entries, ignoring timestamps."""
    if not entry1 or not entry2:
        return False
    return entry1.get('data') == entry2.get('data')

def deduplicate_history(history):
    """Remove consecutive duplicate entries while preserving timestamps of changes."""
    if not history:
        return []

    deduped = [history[0]]  # Keep the first entry
    
    for i in range(1, len(history)):
        current = history[i]
        previous = history[i-1]
        
        # If data is different from previous entry, keep this entry
        if not compare_data(current, previous):
            deduped.append(current)
    
    return deduped

def main():
    # Input and backup filenames
    input_file = 'history.json'
    backup_file = 'history.orig.json'
    
    # Load the original history
    print(f"Loading {input_file}...")
    history = load_history(input_file)
    
    if not history:
        print("No history data found or error loading file.")
        return
    
    # Create backup
    print(f"Creating backup as {backup_file}...")
    shutil.copy2(input_file, backup_file)
    
    # Get original entry count
    original_count = len(history)
    
    # Deduplicate
    print("Deduplicating history...")
    deduped_history = deduplicate_history(history)
    
    # Get deduplicated entry count
    deduped_count = len(deduped_history)
    
    # Save deduplicated history
    print("Saving deduplicated history...")
    save_history(deduped_history, input_file)
    
    # Print statistics
    print("\nDeduplication Statistics:")
    print(f"Original entries: {original_count}")
    print(f"Deduplicated entries: {deduped_count}")
    print(f"Removed entries: {original_count - deduped_count}")
    print(f"Reduction: {((original_count - deduped_count) / original_count * 100):.2f}%")
    
    if deduped_count > 0:
        first_timestamp = deduped_history[0]['timestamp']
        last_timestamp = deduped_history[-1]['timestamp']
        print(f"\nTimeline covered:")
        print(f"First entry: {first_timestamp}")
        print(f"Last entry: {last_timestamp}")

if __name__ == "__main__":
    main() 