import os
import argparse

def collect_code_to_md(root_dir, output_file):
    """
    遍历指定目录下的所有文件，并将 .go 和 .yaml 文件的内容
    按目录结构写入一个 Markdown 文件。

    :param root_dir: 要遍历的根目录。
    :param output_file: 输出的 Markdown 文件名。
    """
    # 获取绝对路径，以便 relpath 正常工作
    root_dir = os.path.abspath(root_dir)

    try:
        with open(output_file, 'w', encoding='utf-8') as md_file:
            # 1. 将根目录的名称作为一级标题
            md_file.write(f"# 索引: {os.path.basename(root_dir)}\n\n")

            # 2. 遍历整个目录树
            for dirpath, _, filenames in os.walk(root_dir):
                # 筛选出当前目录下的 .go 和 .yaml 文件
                # target_files = sorted([f for f in filenames if f.endswith('.py') or f.endswith('.yaml')])
                target_files = sorted([f for f in filenames if f.endswith('.py')])
                # 如果当前目录中没有目标文件，则跳过
                if not target_files:
                    continue

                # 3. 根据目录深度生成 Markdown 标题
                relative_dir_path = os.path.relpath(dirpath, root_dir)

                # 如果不是根目录本身，才创建目录标题
                if relative_dir_path != ".":
                    # 计算深度，根目录的下一级为 H2
                    depth = relative_dir_path.count(os.sep) + 2
                    heading = '#' * depth
                    md_file.write(f"{heading} 目录: `{relative_dir_path}`\n\n")

                # 4. 遍历并写入每个文件的内容
                for filename in target_files:
                    file_path = os.path.join(dirpath, filename)

                    # 确定文件标题的级别，比其所在目录的标题深一级
                    if relative_dir_path == ".":
                        file_heading_level = 2  # 根目录下的文件使用 H2
                    else:
                        file_heading_level = relative_dir_path.count(os.sep) + 3

                    file_heading = '#' * file_heading_level
                    md_file.write(f"{file_heading} 文件: `{filename}`\n\n")

                    # 根据文件扩展名确定代码块的语言
                    lang = ''
                    if filename.endswith('.py'):
                        lang = 'py'
                    elif filename.endswith('.yaml'):
                        lang = 'yaml'

                    md_file.write(f"```{lang}\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as code_file:
                            md_file.write(code_file.read())
                    except Exception as e:
                        md_file.write(f"Error reading file: {e}")
                    md_file.write("\n```\n\n")

        print(f"成功将代码按目录结构写入到 '{output_file}' 文件中。")

    except IOError as e:
        print(f"写入文件时出错: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将一个目录下的所有 .py 和 .yaml 文件的代码按目录结构收集到一个 Markdown 文件中。",
        epilog="例如: python collect_code_hierarchical.py my_project_code.md -d ./my_project"
    )

    parser.add_argument(
        "output_filename",
        help="要创建的 Markdown 文件的名称 (例如: 'output.md')"
    )

    parser.add_argument(
        "-d", "--directory",
        default=".",
        help="要遍历的根目录 (默认为当前目录)"
    )

    args = parser.parse_args()
    collect_code_to_md(args.directory, args.output_filename)