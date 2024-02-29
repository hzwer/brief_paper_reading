import os
import re

# 获取当前目录下的所有 markdown 文件（排除 readme.md）
markdown_files = [f for f in os.listdir('.') if f.endswith('.md') and f.lower() != 'readme.md']

markdown_files = ['LLM.md', 'ReinforcementLearning.md', 'LowLevelVision.md', 'Video.md', 'ImageGeneration.md', 'BaseModel.md', 'Others.md']

# 用于匹配三级标题的正则表达式
header_pattern = re.compile(r'^### (?!\[)(.*)', re.MULTILINE)

# 用于提取文本的正则表达式，忽略 Markdown 链接
link_pattern = re.compile(r'\[(.*?)\]\(.*?\)')

# 用于存储每个文件的三级标题的字典
file_headers = {}

# 遍历每个文件并收集三级标题
for md_file in markdown_files:
    with open(md_file, 'r', encoding='utf-8') as file:
        content = file.read()
        headers = header_pattern.findall(content)
        # 删除标题中的链接，只保留文本        
        headers = [link_pattern.sub(r'\1', header) for header in headers]
        file_headers[md_file] = headers

# 创建或清空 tmp.md 文件
with open('tmp.md', 'w', encoding='utf-8') as readme:
    # 写入 README 文件的标题
    readme.write("# 目录 \n\n")

    # 遍历收集到的每个文件和其标题
    for md_file, headers in file_headers.items():
        readme.write(f"## [{md_file}]({md_file})\n\n")
        for header in headers:
            # 将标题文本转换为锚点链接
            anchor = re.sub(r'[\s\W]+', '-', header.lower()).strip('-')
            anchor = anchor[:5] + '--' + anchor[5:]
            readme.write(f"- [{header}](./{md_file}#{anchor})\n")
        readme.write("\n")

print("目录已生成在 tmp.md 中。")
