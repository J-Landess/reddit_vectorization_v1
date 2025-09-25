#!/usr/bin/env python3
"""
Organize outputs into structured subfolders for better organization.
"""
import os
import shutil
import glob
from datetime import datetime
from pathlib import Path


def organize_outputs(base_dir: str = '.', create_dated_folders: bool = True):
    """
    Organize outputs into structured subfolders.
    
    Args:
        base_dir: Base directory to organize (default: current directory)
        create_dated_folders: Whether to create dated subfolders
    """
    print("🗂️  Organizing outputs...")
    
    # Define source and target directories
    csv_exports_dir = os.path.join(base_dir, 'csv_exports')
    outputs_dir = os.path.join(base_dir, 'outputs')
    historical_runs_dir = os.path.join(base_dir, 'historical_runs')
    
    # Create organized structure
    if create_dated_folders:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        organized_dir = os.path.join(outputs_dir, f'organized_{timestamp}')
    else:
        organized_dir = os.path.join(outputs_dir, 'organized')
    
    # Create subdirectories
    subdirs = {
        'csv_exports': os.path.join(organized_dir, 'csv_exports'),
        'plots': os.path.join(organized_dir, 'plots'),
        'reports': os.path.join(organized_dir, 'reports'),
        'logs': os.path.join(organized_dir, 'logs'),
        'historical_runs': os.path.join(organized_dir, 'historical_runs'),
        'data': os.path.join(organized_dir, 'data')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Organize CSV exports
    if os.path.exists(csv_exports_dir):
        print("📊 Organizing CSV exports...")
        csv_files = glob.glob(os.path.join(csv_exports_dir, '*.csv'))
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            dest = os.path.join(subdirs['csv_exports'], filename)
            shutil.copy2(csv_file, dest)
            print(f"  📄 {filename}")
    
    # Organize plots
    if os.path.exists(outputs_dir):
        print("📈 Organizing plots...")
        plot_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']
        for ext in plot_extensions:
            plot_files = glob.glob(os.path.join(outputs_dir, ext))
            for plot_file in plot_files:
                filename = os.path.basename(plot_file)
                dest = os.path.join(subdirs['plots'], filename)
                shutil.copy2(plot_file, dest)
                print(f"  🖼️  {filename}")
    
    # Organize reports
    if os.path.exists(outputs_dir):
        print("📋 Organizing reports...")
        report_extensions = ['*.md', '*.txt', '*.html']
        for ext in report_extensions:
            report_files = glob.glob(os.path.join(outputs_dir, ext))
            for report_file in report_files:
                filename = os.path.basename(report_file)
                dest = os.path.join(subdirs['reports'], filename)
                shutil.copy2(report_file, dest)
                print(f"  📄 {filename}")
    
    # Organize logs
    logs_dir = os.path.join(base_dir, 'logs')
    if os.path.exists(logs_dir):
        print("📝 Organizing logs...")
        log_files = glob.glob(os.path.join(logs_dir, '*.log'))
        for log_file in log_files:
            filename = os.path.basename(log_file)
            dest = os.path.join(subdirs['logs'], filename)
            shutil.copy2(log_file, dest)
            print(f"  📄 {filename}")
    
    # Organize historical runs (copy latest few)
    if os.path.exists(historical_runs_dir):
        print("📚 Organizing historical runs...")
        run_dirs = sorted([d for d in os.listdir(historical_runs_dir) 
                          if os.path.isdir(os.path.join(historical_runs_dir, d))])
        
        # Copy latest 5 runs
        latest_runs = run_dirs[-5:] if len(run_dirs) > 5 else run_dirs
        for run_dir in latest_runs:
            src = os.path.join(historical_runs_dir, run_dir)
            dest = os.path.join(subdirs['historical_runs'], run_dir)
            shutil.copytree(src, dest)
            print(f"  📁 {run_dir}")
    
    # Organize data files
    data_dir = os.path.join(base_dir, 'data')
    if os.path.exists(data_dir):
        print("💾 Organizing data files...")
        data_files = glob.glob(os.path.join(data_dir, '*'))
        for data_file in data_files:
            if os.path.isfile(data_file):
                filename = os.path.basename(data_file)
                dest = os.path.join(subdirs['data'], filename)
                shutil.copy2(data_file, dest)
                print(f"  💾 {filename}")
    
    # Create summary file
    summary_file = os.path.join(organized_dir, 'organization_summary.md')
    with open(summary_file, 'w') as f:
        f.write(f"# Output Organization Summary\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Directory Structure\n\n")
        f.write(f"```\n")
        f.write(f"{organized_dir}/\n")
        for name, path in subdirs.items():
            f.write(f"├── {name}/\n")
        f.write(f"└── organization_summary.md\n")
        f.write(f"```\n\n")
        f.write(f"## Contents\n\n")
        for name, path in subdirs.items():
            if os.path.exists(path):
                files = os.listdir(path)
                f.write(f"### {name.title()}\n")
                if files:
                    for file in sorted(files):
                        f.write(f"- {file}\n")
                else:
                    f.write(f"- (empty)\n")
                f.write(f"\n")
    
    print(f"\n✅ Organization complete!")
    print(f"📁 Organized outputs saved to: {organized_dir}")
    print(f"📄 Summary: {summary_file}")
    
    return organized_dir


def clean_old_outputs(base_dir: str = '.', keep_days: int = 30):
    """
    Clean old output files to save space.
    
    Args:
        base_dir: Base directory to clean
        keep_days: Number of days to keep files
    """
    import time
    
    print(f"🧹 Cleaning outputs older than {keep_days} days...")
    
    cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
    cleaned_count = 0
    
    # Clean old CSV exports
    csv_exports_dir = os.path.join(base_dir, 'csv_exports')
    if os.path.exists(csv_exports_dir):
        for file in os.listdir(csv_exports_dir):
            file_path = os.path.join(csv_exports_dir, file)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                # Keep canonical files (without timestamps)
                if not (file.startswith('reddit_posts.csv') or 
                       file.startswith('reddit_comments.csv') or 
                       file.startswith('reddit_all_data.csv')):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        print(f"  🗑️  {file}")
                    except Exception as e:
                        print(f"  ⚠️  Could not remove {file}: {e}")
    
    # Clean old logs
    logs_dir = os.path.join(base_dir, 'logs')
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            file_path = os.path.join(logs_dir, file)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                    print(f"  🗑️  {file}")
                except Exception as e:
                    print(f"  ⚠️  Could not remove {file}: {e}")
    
    print(f"✅ Cleaned {cleaned_count} old files")


def main():
    """Main function for output organization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize Reddit analysis outputs')
    parser.add_argument('--base-dir', default='.', 
                       help='Base directory to organize (default: current directory)')
    parser.add_argument('--no-dated-folders', action='store_true',
                       help='Do not create dated subfolders')
    parser.add_argument('--clean-old', action='store_true',
                       help='Clean old output files')
    parser.add_argument('--keep-days', type=int, default=30,
                       help='Days to keep when cleaning (default: 30)')
    
    args = parser.parse_args()
    
    print("🗂️  Reddit Output Organization Tool")
    print("=" * 40)
    
    try:
        # Organize outputs
        organized_dir = organize_outputs(
            base_dir=args.base_dir,
            create_dated_folders=not args.no_dated_folders
        )
        
        # Clean old files if requested
        if args.clean_old:
            clean_old_outputs(args.base_dir, args.keep_days)
        
        print(f"\n🎉 Organization complete!")
        print(f"📁 Check the organized outputs in: {organized_dir}")
        
    except Exception as e:
        print(f"❌ Organization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
