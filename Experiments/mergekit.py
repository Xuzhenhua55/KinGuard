import os
import yaml
import tempfile
import subprocess
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.style import Style

# 初始化Rich组件
console = Console()
error_style = Style(color="red", bold=True)
success_style = Style(color="green", bold=True)
warning_style = Style(color="yellow", bold=True)

# 配置文件和参数设置
config_files = [
    #'/work/txn/mergingmodel/merge_config/ties.yml',
    #'/work/txn/mergingmodel/merge_config/dare_ties.yml'
    '/work/txn/mergingmodel/merge_config/dare_task.yml'
    
]
weight_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

def print_header():
    """打印美观的标题"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", width=20)
    table.add_column("Event", width=50)
    table.add_column("Details", width=60)
    
    console.print("\n")
    console.rule("[bold cyan]🚀 Model Fusion Processor[/bold cyan]", align="left")
    console.print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"🔧 Total Configurations: {len(config_files)}")
    console.print(f"🎚️ Weight Ratios: {', '.join(map(str, weight_ratios))}")
    console.print("\n")

def log_event(status: str, message: str, config: str = ""):
    """统一日志记录"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_colors = {
        "SUCCESS": "green",
        "ERROR": "red",
        "PROCESSING": "blue",
        "WARNING": "yellow"
    }
    
    console.print(
        f"[{timestamp}] "
        f"[{status_colors.get(status, 'white')}]{status:^12}[/] | "
        f"{config:25} | "
        f"{message}"
    )

def main():
    print_header()
    
    with Progress(transient=True) as progress:
        task = progress.add_task("[cyan]Processing configs...", total=len(config_files)*len(weight_ratios))
        
        for config_path in config_files:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                model1 = config['models'][0]['model']
                model2 = config['models'][1]['model']
                model2_name = os.path.basename(model2)
                config_name = os.path.splitext(os.path.basename(config_path))[0]
                
                log_event("PROCESSING", f"Processing config: [bold]{config_name}[/]", config_name)
                
                for weight1 in weight_ratios:
                    weight2 = round(1 - weight1, 1)
                    progress.update(task, advance=1, description=f"Merging {config_name} {weight1}:{weight2}")
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp_file:
                        new_config = config.copy()
                        new_config['models'][0]['parameters']['weight'] = float(weight1)
                        new_config['models'][1]['parameters']['weight'] = float(weight2)
                        yaml.dump(new_config, tmp_file)
                        tmp_path = tmp_file.name
                    
                    output_dir = os.path.join(model1, config_name, model2_name, f"{weight1:.1f}-{weight2:.1f}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    try:
                        result = subprocess.run(
                            ['mergekit-yaml', tmp_path, output_dir],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        log_event("SUCCESS", 
                                f"Saved to [link=file://{output_dir}]{output_dir}[/link]", 
                                config_name)
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Failed: {e.stderr.strip()}" if e.stderr else "Unknown error"
                        log_event("ERROR", 
                                f"{error_msg} [dim](ratio {weight1}:{weight2})[/]", 
                                config_name)
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                            
            except Exception as e:
                log_event("ERROR", f"Config processing failed: {str(e)}", os.path.basename(config_path))

if __name__ == '__main__':
    main()
    console.print("\n[bold cyan]✅ All tasks completed![/bold cyan]\n")


