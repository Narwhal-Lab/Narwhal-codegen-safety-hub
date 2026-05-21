#!/usr/bin/env python3
import os
import csv
import pandas as pd
import json
from collections import defaultdict

"""
This script analyzes CSV files in the SecurityEval/Result directory,
collecting CWE vulnerability types and details for each code file.
"""

def extract_cwe_info(result_dir, model_name="c_llama_sc"):
    """
    Extract CWE vulnerability information from CSV files in specified directory
    
    Args:
        result_dir: Directory path containing CSV files
        model_name: Model name used to construct output filename
        
    Returns:
        Dictionary of vulnerability info {code_file_path: [{vulnerability_details}]}
    """
    # Build full directory path
    csv_dir = os.path.join(result_dir, f"testcases_{model_name}")
    
    if not os.path.exists(csv_dir):
        print(f"Error: Directory does not exist - {csv_dir}")
        return {}
    
    # Dictionary to store results {code_file_path: [{vulnerability_details}]}
    vulnerabilities = defaultdict(list)
    
    # Count vulnerability types
    cwe_type_counts = defaultdict(int)
    
    # Process all CSV files in directory
    print(f"Analyzing directory: {csv_dir}")
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            # Extract CWE type from filename
            cwe_type = file_name.split('_')[2].split('.')[0]
            file_path = os.path.join(csv_dir, file_name)
            
            try:
                # Check if file is empty
                if os.path.getsize(file_path) == 0:
                    print(f"Warning: Empty file - {file_name}")
                    continue
                
                # Read CSV file
                print(f"Processing file: {file_name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:  # Skip empty rows
                            continue
                        
                        # Extract relevant info
                        vuln_type = row[0]
                        description = row[1]
                        severity = row[2]
                        detail = row[3]
                        code_file_path = row[4]  # e.g. "/CWE-022/codeql_1.py"
                        line_start = row[5] if len(row) > 5 else ""
                        col_start = row[6] if len(row) > 6 else ""
                        line_end = row[7] if len(row) > 7 else ""
                        col_end = row[8] if len(row) > 8 else ""
                        
                        # Build vulnerability info object
                        vuln_info = {
                            "cwe_type": cwe_type,
                            "vulnerability_type": vuln_type,
                            "description": description,
                            "severity": severity,
                            "detail": detail,
                            "line_start": line_start,
                            "col_start": col_start,
                            "line_end": line_end,
                            "col_end": col_end
                        }
                        
                        # Add to dictionary
                        vulnerabilities[code_file_path].append(vuln_info)
                        
                        # Update vulnerability type count
                        cwe_type_counts[cwe_type] += 1
                
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
    
    # Print vulnerability type statistics
    print("\nVulnerability type statistics:")
    for cwe_type, count in sorted(cwe_type_counts.items()):
        print(f"CWE-{cwe_type}: {count} vulnerabilities")
    
    return vulnerabilities

def save_results(vulnerabilities, output_file):
    """
    Save results to JSON file
    
    Args:
        vulnerabilities: Dictionary of vulnerability information
        output_file: Output file path
    """
    # Convert defaultdict to regular dict for serialization
    result = {k: v for k, v in vulnerabilities.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Found {len(vulnerabilities)} code files with vulnerabilities")
    
    # Calculate vulnerability count per code file
    vuln_counts = {code_file: len(vulns) for code_file, vulns in vulnerabilities.items()}
    total_vulns = sum(len(vulns) for vulns in vulnerabilities.values())
    
    print(f"Total vulnerabilities: {total_vulns}")
    
    # Generate summary CSV
    summary_file = output_file.replace('.json', '_summary.csv')
    with open(summary_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Code File', 'CWE Types', 'Vulnerability Count'])
        
        for code_file, vulns in vulnerabilities.items():
            # Extract all CWE types for this file
            cwe_types = set(vuln['cwe_type'] for vuln in vulns)
            writer.writerow([code_file, ', '.join(sorted(cwe_types)), len(vulns)])
    
    print(f"Summary report saved to: {summary_file}")

def main():
    # Parameter setup
    result_dir = "../SecurityEval/Result"
    model_name = "c_llama_rd_4_2_2"  # Default model name
    output_file = f"../SecurityEval/Result/analyze_res/{model_name}_vulnerabilities.json"
    
    # Read model name from command line if provided
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        output_file = f"../SecurityEval/Result/analyze_res/{model_name}_vulnerabilities.json"
    
    # Extract vulnerability information
    vulnerabilities = extract_cwe_info(result_dir, model_name)
    
    # Save results
    save_results(vulnerabilities, output_file)

if __name__ == "__main__":
    main()
